import pandas as pd
import numpy as np
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
import time
from functools import wraps


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


### 📌 Function to Chunk Text ###
def chunk_text(text, chunk_size=15, overlap=5):
    """
    Improved text chunking with sentence awareness.
    - Respects sentence boundaries
    - Maintains word overlap between chunks
    - chunk_size: target number of words per chunk
    - overlap: number of words to overlap between chunks
    """
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

def detect_prompt_injections(
    text_chunks: List[str], 
    model, 
    prompt_embeddings, 
    similarity_threshold: float = 0.65
) -> List[Tuple[str, float, int]]:
    """
    Detect potential prompt injections in text chunks
    
    Args:
        text_chunks: List of text chunks to analyze
        model: Sentence embedding model
        prompt_embeddings: Pre-computed prompt template embeddings
        similarity_threshold: Minimum similarity to flag as injection
    
    Returns:
        List of (chunk, max_similarity, template_index)
    """
    # Clean chunks
    cleaned_chunks = [clean_text(chunk) for chunk in text_chunks if chunk.strip()]
    
    if not cleaned_chunks:
        return []
    
    results = []
    
    try:
        # Encode text chunks
        chunk_embeddings = model.encode(cleaned_chunks, convert_to_numpy=True, show_progress_bar=False)
        
        # Compute similarities
        for i, chunk_emb in enumerate(chunk_embeddings):
            # Calculate cosine similarities with all prompt embeddings
            similarities = [1 - cosine(chunk_emb, prompt_emb) for prompt_emb in prompt_embeddings]
            
            # Find max similarity
            max_similarity = max(similarities)
            template_index = similarities.index(max_similarity)
            
            # Check against threshold
            if max_similarity >= similarity_threshold:
                results.append((
                    cleaned_chunks[i], 
                    max_similarity, 
                    template_index
                ))
    
    except Exception as e:
        logger.error(f"Error in prompt injection detection: {e}")
        return []
    
    return results

# Add this decorator to measure execution time
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

# Modify your functions to return timing information
@timing_decorator
def process_resume_for_injection(
    resume_text: str, 
    model, 
    prompt_embeddings, 
    similarity_threshold: float = 0.65
) -> int:
    """
    Determine if a resume contains prompt injections
    
    Args:
        resume_text: Full resume text
        model: Sentence embedding model
        prompt_embeddings: Pre-computed prompt template embeddings
        similarity_threshold: Threshold for injection detection
    
    Returns:
        1 if potential injection detected, 0 otherwise
    """
    # Chunk the resume
    chunks = chunk_text(resume_text)
    
    # Detect injections
    injections = detect_prompt_injections(
        chunks, 
        model, 
        prompt_embeddings, 
        similarity_threshold
    )
    
    # If any chunks are flagged, consider it a potential injection
    return 1 if injections else 0

def evaluate_prompt_injection_detection(
    df: pd.DataFrame, 
    model, 
    prompt_embeddings, 
    similarity_threshold: float = 0.65
):
    """
    Evaluate prompt injection detection performance
    
    Args:
        df: DataFrame with 'resume' and 'label' columns
        model: Sentence embedding model
        prompt_embeddings: Pre-computed prompt template embeddings
        similarity_threshold: Threshold for injection detection
    
    Returns:
        Dictionary of performance metrics
    """
    # Predict injections
    df['predicted'] = df['resume'].apply(
        lambda x: process_resume_for_injection(
            x, model, prompt_embeddings, similarity_threshold
        )
    )
    
    # Calculate metrics
    precision = precision_score(df['label'], df['predicted'], zero_division=0)
    recall = recall_score(df['label'], df['predicted'], zero_division=0)
    f1 = f1_score(df['label'], df['predicted'], zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(df['label'], df['predicted'])
    
    # Log results
    logger.info(f"Prompt Injection Detection Performance:")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(str(cm))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def main():
    # Load models and data
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(MODEL_NAME)
    
    # Load prompt embeddings
    try:
        prompt_data = np.load('/home/aminou/prompt-injection-detection/dataset/prompts_embeddings.npz')
        prompt_embeddings = prompt_data['embeddings']
    except Exception as e:
        logger.error(f"Error loading prompt embeddings: {e}")
        return
    
    # Load dataset
    try:
        df = pd.read_csv("dataset/injected_resume_dataset.csv")
        # Reduce dataset to a balanced 1:1 ratio (500 samples per class)
        df_0 = df[df["label"] == 0] # .sample(n=500, random_state=42) Sample 500 "0" resumes
        df_1 = df[df["label"] == 1]# .sample(n=500, random_state=42)  Sample 500 "1" resumes
        df = pd.concat([df_0, df_1]).reset_index(drop=True)

    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Threshold optimization
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    best_f1 = 0
    best_threshold = 0.65
    
    # Final evaluation with best threshold

    df['predicted'] = df['resume'].apply(
            lambda x: process_resume_for_injection(
                x, model, prompt_embeddings, best_threshold
            )
        )


    evaluate_prompt_injection_detection(
        df, model, prompt_embeddings, best_threshold
    )

def benchmark_guard_pipeline(df, model, prompt_embeddings, threshold=0.65):
    """Comprehensive benchmarking for GUARD"""
    logger.info("=== GUARD Pipeline Benchmarking ===")
    
    predictions = []
    times = []
    
    # Process each resume with timing
    for resume in df['resume']:
        prediction, exec_time = process_resume_for_injection(
            resume, model, prompt_embeddings, threshold
        )
        predictions.append(prediction)
        times.append(exec_time)
    
    # Calculate metrics
    precision = precision_score(df['label'], predictions, zero_division=0)
    recall = recall_score(df['label'], predictions, zero_division=0)
    f1 = f1_score(df['label'], predictions, zero_division=0)
    
    # Performance statistics
    avg_time = np.mean(times)
    throughput = len(df) / sum(times)  # resumes per second
    
    logger.info(f"GUARD Performance:")
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

if __name__ == "__main__":
    main()
