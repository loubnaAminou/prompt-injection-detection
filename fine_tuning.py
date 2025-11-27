import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, LoggingHandler
from torch import nn
import torch
import logging
import re
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

# Disable tqdm progress bars globally
from sentence_transformers.util import batch_to_device
SentenceTransformer.show_progress_bar = False

# Setup logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


# Text preprocessing functions
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Apply full preprocessing to resume chunks and prompts"""
    text = text.lower()  # Lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Fix spaces
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

def chunk_text(text, chunk_size=15, overlap=5):
    """Chunk text into smaller pieces"""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk_words = []
    
    for sentence in sentences:
        sentence_words = sentence.split()
        
        if len(current_chunk_words) + len(sentence_words) <= chunk_size:
            current_chunk_words.extend(sentence_words)
        else:
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
            
            current_chunk_words = sentence_words
            
            while len(current_chunk_words) > chunk_size:
                chunks.append(" ".join(current_chunk_words[:chunk_size]))
                current_chunk_words = current_chunk_words[chunk_size-overlap:]
    
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
    
    return chunks

# Load and prepare data
df = pd.read_csv("dataset/combined_resume_dataset.csv")

# Sample to balance the dataset
zero_labeled = df[df['label'] == 0].sample(n=500, random_state=42)
one_labeled = df[df['label'] == 1]
balanced_df = pd.concat([zero_labeled, one_labeled]).reset_index(drop=True)

# Split train/validation
train_df, val_df = train_test_split(balanced_df, test_size=0.2, random_state=42, stratify=balanced_df["label"])

# Process text chunks for binary classification
class ChunkedTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.samples = []
        
        for text, label in zip(texts, labels):
            cleaned_text = clean_text(text)
            chunks = chunk_text(cleaned_text)
            
            for chunk in chunks:
                self.samples.append({
                    'text': chunk,
                    'label': float(label)
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

# Create datasets
train_dataset = ChunkedTextDataset(train_df["resume"].tolist(), train_df["label"].tolist())
val_dataset = ChunkedTextDataset(val_df["resume"].tolist(), val_df["label"].tolist())

# Create dataloaders
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, shuffle=True, batch_size=16
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, shuffle=False, batch_size=16
)

# Create a classifier on top of SentenceTransformer
class SentenceClassifier(nn.Module):
    def __init__(self, model_name):
        super(SentenceClassifier, self).__init__()
        self.encoder = SentenceTransformer(model_name)
        self.classifier = nn.Linear(self.encoder.get_sentence_embedding_dimension(), 1)
        
    def forward(self, texts):
        # Get embeddings from SentenceTransformer
        with torch.no_grad():
            embeddings = self.encoder.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        
        # Apply classification layer
        logits = self.classifier(embeddings)
        return logits

# Initialize model
model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
model = SentenceClassifier(model_name)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)  # Only train the classifier
num_epochs = 3

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        optimizer.zero_grad()
        
        texts = batch['text']
        labels = torch.tensor(batch['label'], dtype=torch.float32).to(device)
        
        # Forward pass
        logits = model(texts)
        loss = criterion(logits.squeeze(), labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            texts = batch['text']
            labels = torch.tensor(batch['label'], dtype=torch.float32).to(device)
            
            # Forward pass
            logits = model(texts)
            predictions = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            
            val_correct += (predictions == labels).sum().item()
            val_total += labels.size(0)
    
    val_accuracy = val_correct / val_total
    avg_loss = total_loss / len(train_dataloader)
    
    logging.info(f"Epoch {epoch+1}/{num_epochs}: Loss={avg_loss:.4f}, Val Accuracy={val_accuracy:.4f}")

# Save the classifier part (which we fine-tuned)
torch.save(model.classifier.state_dict(), "prompt_injection_classifier.pt")


class BenchmarkClassifier:
    def __init__(self, model_path=None):
        self.model_name = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        self.encoder = SentenceTransformer(self.model_name)
        self.classifier = nn.Linear(self.encoder.get_sentence_embedding_dimension(), 1)
        
        if model_path:
            self.classifier.load_state_dict(torch.load(model_path))
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier.to(self.device)
    
    def predict_resume(self, resume_text, threshold=0.5):
        """Predict if a resume contains injections with timing"""
        start_time = time.time()
        
        # Preprocess and chunk
        cleaned_text = clean_text(resume_text)
        chunks = chunk_text(cleaned_text)
        
        if not chunks:
            return 0, time.time() - start_time
        
        # Get predictions for all chunks
        self.classifier.eval()
        with torch.no_grad():
            chunk_embeddings = self.encoder.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
            logits = self.classifier(chunk_embeddings)
            probs = torch.sigmoid(logits.squeeze())
            max_prob = torch.max(probs).item()
        
        exec_time = time.time() - start_time
        return 1 if max_prob > threshold else 0, exec_time
    
    def benchmark_classifier(self, df, threshold=0.5):
        """Comprehensive benchmarking for classifier"""
        logger.info("=== Classifier Benchmarking ===")
        
        predictions = []
        times = []
        
        for resume in df['resume']:
            prediction, exec_time = self.predict_resume(resume, threshold)
            predictions.append(prediction)
            times.append(exec_time)
        
        # Calculate metrics
        precision = precision_score(df['label'], predictions, zero_division=0)
        recall = recall_score(df['label'], predictions, zero_division=0)
        f1 = f1_score(df['label'], predictions, zero_division=0)
        
        # Performance statistics
        avg_time = np.mean(times)
        throughput = len(df) / sum(times)
        
        logger.info(f"Classifier Performance:")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"Avg Inference Time: {avg_time:.4f}s per resume")
        logger.info(f"Throughput: {throughput:.2f} resumes/sec")
        logger.info(f"Total Processing Time: {sum(times):.2f}s")
        
        return {
            'f1_score': f1,
            'precision': precision, 
            'recall': recall,
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'total_time': sum(times),
            'predictions': predictions
        }