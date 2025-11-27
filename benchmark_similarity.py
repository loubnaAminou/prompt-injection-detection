import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean, cityblock
from bert_score import BERTScorer
import torch
import numpy as np
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


# Initialize BERTScore once (expensive operation)
bert_scorer = None

def initialize_bert_scorer():
    """Initialize BERTScorer (expensive, so do it once)"""
    global bert_scorer
    if bert_scorer is None:
        bert_scorer = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en")
    return bert_scorer

def calculate_similarity(embedding1, embedding2, method='cosine'):
    """
    Calculate similarity using different distance metrics
    Note: We convert distances to similarities (higher = more similar)
    """
    if method == 'cosine':
        return 1 - cosine(embedding1, embedding2)
    elif method == 'euclidean':
        # Convert distance to similarity (inverse relationship)
        distance = euclidean(embedding1, embedding2)
        return 1 / (1 + distance)  # Normalized similarity
    elif method == 'manhattan':
        distance = cityblock(embedding1, embedding2)
        return 1 / (1 + distance)
    elif method == 'dot_product':
        return np.dot(embedding1, embedding2)
    elif method == 'jaccard':
        # For embeddings, we'll use a threshold-based Jaccard
        threshold = 0.1
        set1 = set(np.where(embedding1 > threshold)[0])
        set2 = set(np.where(embedding2 > threshold)[0])
        if len(set1 | set2) == 0:
            return 0
        return len(set1 & set2) / len(set1 | set2)
    elif method == 'bertscore':
        # BERTScore requires actual text, not embeddings
        # We'll handle this separately in the detection function
        return 0  # Placeholder
    else:
        raise ValueError(f"Unknown method: {method}")
    

def detect_prompt_injections_with_metric(
    text_chunks: List[str], 
    model, 
    prompt_embeddings, 
    similarity_threshold: float = 0.65,
    metric_method: str = 'cosine'
) -> List[Tuple[str, float, int]]:
    """
    Detect injections using specified distance metric
    """
    cleaned_chunks = [clean_text(chunk) for chunk in text_chunks if chunk.strip()]
    
    if not cleaned_chunks:
        return []
    
    results = []
    
    try:
        chunk_embeddings = model.encode(cleaned_chunks, convert_to_numpy=True, show_progress_bar=False)
        
        for i, chunk_emb in enumerate(chunk_embeddings):
            # Use the specified metric method
            similarities = [calculate_similarity(chunk_emb, prompt_emb, metric_method) 
                          for prompt_emb in prompt_embeddings]
            
            max_similarity = max(similarities)
            template_index = similarities.index(max_similarity)
            
            if max_similarity >= similarity_threshold:
                results.append((cleaned_chunks[i], max_similarity, template_index))
    
    except Exception as e:
        logger.error(f"Error in detection with {metric_method}: {e}")
        return []
    
    return results

def detect_prompt_injections_with_bertscore(
    text_chunks: List[str], 
    prompt_texts: List[str],
    similarity_threshold: float = 0.65
) -> List[Tuple[str, float, int]]:
    """
    Detect injections using BERTScore (requires actual text, not embeddings)
    """
    cleaned_chunks = [clean_text(chunk) for chunk in text_chunks if chunk.strip()]
    
    if not cleaned_chunks:
        return []
    
    results = []
    
    try:
        # Initialize BERTScorer
        scorer = initialize_bert_scorer()
        
        for i, chunk_text in enumerate(cleaned_chunks):
            # BERTScore compares text-to-text (not embedding-to-embedding)
            # We need to compare this chunk against all prompt texts
            P, R, F1 = scorer.score([chunk_text] * len(prompt_texts), prompt_texts)
            
            # Use F1 score (harmonic mean of precision and recall)
            similarities = F1.numpy()
            max_similarity = np.max(similarities)
            template_index = np.argmax(similarities)
            
            if max_similarity >= similarity_threshold:
                results.append((chunk_text, max_similarity, template_index))
    
    except Exception as e:
        logger.error(f"Error in BERTScore detection: {e}")
        return []
    
    return results

@timing_decorator
def process_resume_with_bertscore(
    resume_text: str, 
    prompt_texts: List[str],
    similarity_threshold: float = 0.65
) -> int:
    """Process resume using BERTScore"""
    chunks = chunk_text(resume_text)
    injections = detect_prompt_injections_with_bertscore(
        chunks, prompt_texts, similarity_threshold
    )
    return 1 if injections else 0

@timing_decorator
def process_resume_with_metric(
    resume_text: str, 
    model, 
    prompt_embeddings, 
    similarity_threshold: float = 0.65,
    metric_method: str = 'cosine'
) -> int:
    """Process resume with specific distance metric"""
    chunks = chunk_text(resume_text)
    injections = detect_prompt_injections_with_metric(
        chunks, model, prompt_embeddings, similarity_threshold, metric_method
    )
    return 1 if injections else 0

def benchmark_all_metrics(df, model, prompt_embeddings, prompt_texts=None, threshold=0.65):
    """Benchmark all distance metrics comprehensively"""
    
    metrics_to_test = ['cosine', 'euclidean', 'manhattan', 'dot_product', 'jaccard', 'bertscore']
    results = {}
    
    for metric in metrics_to_test:
        logger.info(f"=== Testing {metric.upper()} metric ===")
        
        predictions = []
        times = []
        
        # Special handling for BERTScore
        if metric == 'bertscore':
            if prompt_texts is None:
                logger.warning("Skipping BERTScore - no prompt texts provided")
                continue
                
            for resume in df['resume']:
                prediction, exec_time = process_resume_with_bertscore(
                    resume, prompt_texts, threshold
                )
                predictions.append(prediction)
                times.append(exec_time)
        else:
            # Standard metrics
            for resume in df['resume']:
                prediction, exec_time = process_resume_with_metric(
                    resume, model, prompt_embeddings, threshold, metric
                )
                predictions.append(prediction)
                times.append(exec_time)
        
        # Calculate performance metrics
        precision = precision_score(df['label'], predictions, zero_division=0)
        recall = recall_score(df['label'], predictions, zero_division=0)
        f1 = f1_score(df['label'], predictions, zero_division=0)
        
        avg_time = np.mean(times)
        throughput = len(df) / sum(times) if times else 0
        
        results[metric] = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'total_time': sum(times)
        }
        
        logger.info(f"{metric.upper()} -> F1: {f1:.3f}, Precision: {precision:.3f}, "
                   f"Recall: {recall:.3f}, Speed: {throughput:.1f} resumes/sec")
    
    return results


def create_metric_comparison_table(results):
    """Create a beautiful comparison table"""
    comparison_data = []
    
    for metric, data in results.items():
        comparison_data.append({
            'Distance Metric': metric.upper(),
            'F1-Score': f"{data['f1_score']:.3f}",
            'Precision': f"{data['precision']:.3f}",
            'Recall': f"{data['recall']:.3f}",
            'Throughput (resumes/sec)': f"{data['throughput']:.1f}",
            'Relative F1 (%)': f"{(data['f1_score'] / results['cosine']['f1_score'] - 1) * 100:+.1f}%"
        })
    
    # Sort by F1-score descending
    comparison_data.sort(key=lambda x: float(x['F1-Score']), reverse=True)
    
    return pd.DataFrame(comparison_data)

def main():
    # Load models and data
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(MODEL_NAME)
    
    prompt_data = np.load('/home/aminou/prompt-injection-detection/dataset/prompts_embeddings.npz')
    prompt_embeddings = prompt_data['embeddings']
    
    # You'll need to load the actual prompt texts for BERTScore
    # Replace this with how you load your prompt texts
    prompt_texts = [
        "ignore previous instructions",
        "override the system", 
        "bypass security checks",
        # ... add all your actual prompt texts here
    ]
    
    df = pd.read_csv("dataset/injected_resume_dataset.csv")
    df_0 = df[df["label"] == 0]
    df_1 = df[df["label"] == 1]
    df_balanced = pd.concat([df_0, df_1]).reset_index(drop=True)
    
    # Use a smaller subset for quick testing (BERTScore is slow)
    df_test = df_balanced #.sample(n=50, random_state=42)  # Smaller due to BERTScore slowness
    
    logger.info("Starting comprehensive distance metric comparison...")
    
    # Benchmark all metrics (including BERTScore)
    results = benchmark_all_metrics(df_test, model, prompt_embeddings, prompt_texts, threshold=0.65)
    
    # Create and display comparison table
    comparison_df = create_metric_comparison_table(results)
    
    print("\n" + "="*80)
    print("DISTANCE METRIC COMPARISON RESULTS (WITH BERTSCORE)")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    # Save results for your paper
    comparison_df.to_csv("distance_metric_comparison_with_bertscore.csv", index=False)
    logger.info("Results saved to distance_metric_comparison_with_bertscore.csv")
    
    # Find the best metric
    if results:
        best_metric = max(results.items(), key=lambda x: x[1]['f1_score'])
        logger.info(f"Best performing metric: {best_metric[0].upper()} "
                   f"(F1-Score: {best_metric[1]['f1_score']:.3f})")

if __name__ == "__main__":
    main()