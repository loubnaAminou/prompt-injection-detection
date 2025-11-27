import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import re
import string
import re
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import precision_recall_curve

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Load the pre-trained Sentence-BERT model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# Load the stored prompt embeddings
data = np.load('/home/aminou/prompt-injection-detection/dataset/prompts_embeddings.npz')
prompt_embeddings = data["embeddings"]


# **Load Resume CSV File**
df = pd.read_csv("dataset/combined_resume_dataset.csv") 


### 📌 Function to clean Text ###
def clean_text(text):
    """Apply full preprocessing to resume chunks and prompts"""
    text = text.lower()  # Lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Fix spaces
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatize & remove stopwords
    return text



### 📌 Function to Chunk Text ###
def chunk_text(text, chunk_size=15, overlap=5):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk_words = []
    
    for sentence in sentences:
        sentence_words = sentence.split()
        
        # If adding this sentence doesn't exceed chunk size, add it
        if len(current_chunk_words) + len(sentence_words) <= chunk_size:
            current_chunk_words.extend(sentence_words)
        else:
            # If we have words in the current chunk, save it
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
            
            # Start a new chunk with this sentence
            current_chunk_words = sentence_words
            
            # If this single sentence exceeds chunk size, split it while preserving overlap
            while len(current_chunk_words) > chunk_size:
                chunks.append(" ".join(current_chunk_words[:chunk_size]))
                current_chunk_words = current_chunk_words[chunk_size-overlap:]
    
    # Don't forget the last chunk if it has content
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
    
    return chunks


# **Batch Process Resumes**
def process_resume(resume_text, threshold=0.65):
    resume_sentences = chunk_text(resume_text) 

    resume_sentences = [clean_text(chunk) for chunk in resume_sentences]
    
    resume_embeddings = model.encode(resume_sentences)

    # Compute max cosine similarity between resume and injected prompts
    max_sim = max(
        1 - cosine(resume_emb, prompt_emb)
        for resume_emb in resume_embeddings
        for prompt_emb in prompt_embeddings
    )

    return 1 if max_sim >= threshold else 0  # Classify as injected if above threshold

# **Apply Detection on All Resumes**
df["predicted"] = df["resume"].apply(lambda x: process_resume(x, threshold=0.65))

# **Evaluate Performance**
y_true = df["label"].tolist()  # Ground truth
y_pred = df["predicted"].tolist()  # Predicted labels

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")

thresholds = np.arange(0.6, 0.9, 0.01)  # Try values between 0.6 and 0.8
best_f1 = 0
best_threshold = 0.65

print("===> trying thresholds")

for t in thresholds:
    # df["predicted"] = df["resume"].apply(lambda x: process_resume(x, threshold=t))
    print("===> trying precision")
    p = precision_score(df["label"], df["predicted"])
    print("===> trying recall")
    r = recall_score(df["label"], df["predicted"])
    print("===> trying F1-score")
    f1 = f1_score(df["label"], df["predicted"])
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = t

print(f"Best Threshold: {best_threshold}, Best F1-score: {best_f1:.3f}")


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