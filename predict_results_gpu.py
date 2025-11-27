import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.spatial.distance import cosine
import re
import string
import logging
from typing import List, Tuple
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load NLTK resources
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Text Preprocessing
def clean_text(text):
    """Apply full preprocessing to resume chunks and prompts"""
    text = text.lower()  # Lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Fix spaces
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatize & remove stopwords
    return text

def chunk_text(text, chunk_size=15, overlap=5):
    """Improved text chunking while maintaining sentence boundaries"""
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
                current_chunk_words = current_chunk_words[chunk_size - overlap:]

    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))

    return chunks

def detect_prompt_injections(
    text_chunks: List[str], 
    model, 
    prompt_embeddings, 
    similarity_threshold: float = 0.65
) -> List[Tuple[str, float, int]]:
    """Detect potential prompt injections in text chunks"""
    cleaned_chunks = [clean_text(chunk) for chunk in text_chunks if chunk.strip()]
    
    if not cleaned_chunks:
        return []

    results = []
    
    try:
        chunk_embeddings = model.encode(cleaned_chunks, convert_to_numpy=True, show_progress_bar=True, device=device)
        chunk_embeddings = torch.tensor(chunk_embeddings, device=device)  # Move to GPU
        
        prompt_embeddings = torch.tensor(prompt_embeddings, device=device)  # Move to GPU

        for i, chunk_emb in enumerate(chunk_embeddings):
            similarities = 1 - torch.nn.functional.cosine_similarity(chunk_emb.unsqueeze(0), prompt_embeddings)  # GPU similarity computation
            max_similarity, template_index = similarities.max(dim=0)

            if max_similarity.item() >= similarity_threshold:
                results.append((cleaned_chunks[i], max_similarity.item(), template_index.item()))

    except Exception as e:
        logger.error(f"Error in prompt injection detection: {e}")
        return []
    
    return results

def process_resume_for_injection(
    resume_text: str, 
    model, 
    prompt_embeddings, 
    similarity_threshold: float = 0.65
) -> int:
    """Determine if a resume contains prompt injections"""
    chunks = chunk_text(resume_text)
    injections = detect_prompt_injections(chunks, model, prompt_embeddings, similarity_threshold)
    return 1 if injections else 0

def evaluate_prompt_injection_detection(
    df: pd.DataFrame, 
    model, 
    prompt_embeddings, 
    similarity_threshold: float = 0.65
):
    """Evaluate prompt injection detection performance"""
    df['predicted'] = df['resume'].apply(lambda x: process_resume_for_injection(x, model, prompt_embeddings, similarity_threshold))
    
    precision = precision_score(df['label'], df['predicted'], zero_division=0)
    recall = recall_score(df['label'], df['predicted'], zero_division=0)
    f1 = f1_score(df['label'], df['predicted'], zero_division=0)
    cm = confusion_matrix(df['label'], df['predicted'])
    
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def main():
    """Main execution function"""
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(MODEL_NAME, device=device)  # Load model on GPU
    
    try:
        prompt_data = np.load('/home/aminou/prompt-injection-detection/dataset/prompts_embeddings.npz')
        prompt_embeddings = torch.tensor(prompt_data['embeddings'], device=device)  # Load embeddings to GPU
    except Exception as e:
        logger.error(f"Error loading prompt embeddings: {e}")
        return
    
    try:
        df = pd.read_csv("dataset/combined_resume_dataset.csv")
        df_0 = df[df["label"] == 0].sample(n=500, random_state=42)
        df_1 = df[df["label"] == 1].sample(n=500, random_state=42)
        df = pd.concat([df_0, df_1]).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    best_threshold = 0.68  # Predefined best threshold

    df['predicted'] = df['resume'].apply(lambda x: process_resume_for_injection(x, model, prompt_embeddings, best_threshold))

    evaluate_prompt_injection_detection(df, model, prompt_embeddings, best_threshold)

if __name__ == "__main__":
    main()

# Threshold optimization
# thresholds = np.arange(0.5, 0.9, 0.05)
# best_f1 = 0
# best_threshold = 0.65

# for t in thresholds:
#     df["predicted"] = df["resume"].apply(lambda x: process_resume(x, prompt_embeddings, model, threshold=t))
#     f1 = f1_score(df["label"], df["predicted"])
    
#     if f1 > best_f1:
#         best_f1 = f1
#         best_threshold = t

# print(f"Best Threshold: {best_threshold}, Best F1-score: {best_f1:.3f}")

# **Plot ROC Curve**
# similarity_scores = [
#     max(1 - cosine(model.encode(sent), prompt_emb) for sent in chunk_text(resume) for prompt_emb in prompt_embeddings)
#     for resume in df["resume"]
# ]

# fpr, tpr, _ = roc_curve(y_true, similarity_scores)
# roc_auc = auc(fpr, tpr)

# plt.figure(figsize=(6, 6))
# plt.plot(fpr, tpr, color="blue", label=f"AUC = {roc_auc:.3f}")
# plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
# plt.xlabel("False Positive Rate")
# plt.ylabel("True Positive Rate")
# plt.title("ROC Curve")
# plt.legend()
# plt.savefig("ROC")
