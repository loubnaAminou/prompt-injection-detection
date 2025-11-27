import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix,
    roc_curve,
    auc
)
from scipy.spatial.distance import cosine
import re
import string
import logging
from typing import List, Tuple
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
import torch
torch.cuda.empty_cache()  # Free unused memory


# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Text Preprocessing
def clean_text(text):
    """Normalize and clean text for better embedding comparison"""
    text = text.lower()  # Lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Fix spaces
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatize & remove stopwords
    return text

def chunk_text(text, chunk_size=15, overlap=5):
    """Split text into overlapping chunks respecting sentence boundaries"""
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

def get_max_similarity_scores(df, model, prompt_embeddings):
    """Process one resume at a time to avoid OOM errors"""
    similarity_scores = []
    
    for i, resume in enumerate(df["resume"]):
        if i % 10 == 0:
            logger.info(f"Processing resume {i}/{len(df)}")
            torch.cuda.empty_cache()
        
        chunks = chunk_text(resume)
        cleaned_chunks = [clean_text(chunk) for chunk in chunks if chunk.strip()]
        
        if not cleaned_chunks:
            similarity_scores.append(0)
            continue
            
        # Process one chunk at a time
        max_sim = 0
        for chunk in cleaned_chunks:
            chunk_emb = model.encode(
                chunk, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                device="cuda"  # Explicitly use GPU
            )
            similarities = [1 - cosine(chunk_emb, prompt_emb) for prompt_emb in prompt_embeddings]
            max_sim = max(max_sim, max(similarities)) if similarities else max_sim
            del chunk_emb  # Free memory
            torch.cuda.empty_cache()
        
        similarity_scores.append(max_sim)
    
    return similarity_scores

def evaluate_thresholds(df, similarity_scores, thresholds):
    """Evaluate performance across thresholds and return metrics"""
    results = []
    
    for threshold in thresholds:
        df['predicted'] = (np.array(similarity_scores) >= threshold).astype(int)
        
        precision = precision_score(df['label'], df['predicted'], zero_division=0)
        recall = recall_score(df['label'], df['predicted'], zero_division=0)
        f1 = f1_score(df['label'], df['predicted'], zero_division=0)
        
        results.append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cm': confusion_matrix(df['label'], df['predicted'])
        })
    
    return pd.DataFrame(results)

def plot_roc_curve(y_true, similarity_scores, save_path="assets/roc_curve.png"):
    """Plot ROC curve and save to file"""
    fpr, tpr, _ = roc_curve(y_true, similarity_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
    logger.info(f"ROC curve saved to {save_path}")


def plot_threshold_analysis(results_df, save_path="assets/threshold_plot.png"):
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=results_df, x="threshold", y="precision", label="Precision", marker="o")
    sns.lineplot(data=results_df, x="threshold", y="recall", label="Recall", marker="o")
    sns.lineplot(data=results_df, x="threshold", y="f1", label="F1 Score", marker="o")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Model Performance across Thresholds")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Threshold performance plot saved to {save_path}")


def main():
    # Load model and data
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    
    try:
        prompt_data = np.load('dataset/prompts_embeddings.npz')
        prompt_embeddings = prompt_data['embeddings']
    except Exception as e:
        logger.error(f"Error loading prompt embeddings: {e}")
        return
    
    try:
        df = pd.read_csv("dataset/injected_resume_dataset.csv")
        df_0 = df[df["label"] == 0]#.sample(n=500, random_state=42)
        df_1 = df[df["label"] == 1]#.sample(n=500, random_state=42)
        df = pd.concat([df_0, df_1]).reset_index(drop=True)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Precompute similarity scores
    logger.info("Computing similarity scores...")
    similarity_scores = get_max_similarity_scores(df, model, prompt_embeddings)
    
    # Threshold optimization
    thresholds = np.arange(0.5, 0.85, 0.05)
    results_df = evaluate_thresholds(df, similarity_scores, thresholds)
    
    # Find best threshold
    best_row = results_df.loc[results_df['f1'].idxmax()]
    logger.info(f"\nBest Threshold: {best_row['threshold']:.2f}")
    logger.info(f"Precision: {best_row['precision']:.4f}")
    logger.info(f"Recall: {best_row['recall']:.4f}")
    logger.info(f"F1-Score: {best_row['f1']:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(str(best_row['cm']))

    # Display performance metrics table
    logger.info("\nPerformance metrics across thresholds:")
    logger.info(results_df[['threshold', 'precision', 'recall', 'f1']])

    # Export LaTeX table to a .tex file
    latex_table = results_df[['threshold', 'precision', 'recall', 'f1']].to_latex(index=False, float_format="%.2f")
    with open("assets/threshold_table.tex", "w") as f:
        f.write(latex_table)

    plot_threshold_analysis(results_df)

    
    # Plot ROC curve
    plot_roc_curve(df['label'], similarity_scores)
    
    # Save threshold analysis results
    results_df.to_csv("assets/threshold_analysis.csv", index=False)
    logger.info("Threshold analysis saved to assets/threshold_analysis.csv")

if __name__ == "__main__":
    main()