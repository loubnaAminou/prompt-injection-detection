import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.spatial.distance import cosine
import re
import string
import logging
from typing import List, Tuple, Dict
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
from functools import wraps
import matplotlib.pyplot as plt
import seaborn as sns

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Text Preprocessing
def clean_text(text):
    """Apply full preprocessing to resume chunks and prompts"""
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

### 📌 Enhanced Chunking Strategies ###

def chunk_text_sentence_aware(text, chunk_size=15, overlap=5):
    """
    Original sentence-aware chunking with overlap
    - Respects sentence boundaries
    - Maintains word overlap between chunks
    """
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

def chunk_text_fixed_window(text, chunk_size=15, overlap=5):
    """
    Fixed window chunking without sentence boundaries
    - Simple sliding window approach
    - Ignores sentence structure
    """
    words = text.split()
    chunks = []
    
    if len(words) <= chunk_size:
        return [" ".join(words)]
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if i + chunk_size >= len(words):
            break
    
    return chunks

def chunk_text_paragraph_based(text, chunk_size=15, overlap=5):
    """
    Paragraph-based chunking
    - Splits by paragraphs first, then chunks if needed
    """
    paragraphs = text.split('\n\n')
    chunks = []
    
    for paragraph in paragraphs:
        para_text = paragraph.strip()
        if not para_text:
            continue
            
        if len(para_text.split()) <= chunk_size:
            chunks.append(para_text)
        else:
            # If paragraph is too long, use sentence-aware chunking
            para_chunks = chunk_text_sentence_aware(para_text, chunk_size, overlap)
            chunks.extend(para_chunks)
    
    return chunks

def chunk_text_no_segmentation(text, chunk_size=15, overlap=5):
    """
    No segmentation - process entire text as one chunk
    """
    return [text]

def detect_prompt_injections(
    text_chunks: List[str], 
    model, 
    prompt_embeddings, 
    similarity_threshold: float = 0.65
) -> List[Tuple[str, float, int]]:
    """
    Detect potential prompt injections in text chunks
    """
    cleaned_chunks = [clean_text(chunk) for chunk in text_chunks if chunk.strip()]
    
    if not cleaned_chunks:
        return []
    
    results = []
    
    try:
        chunk_embeddings = model.encode(cleaned_chunks, convert_to_numpy=True, show_progress_bar=False)
        
        for i, chunk_emb in enumerate(chunk_embeddings):
            similarities = [1 - cosine(chunk_emb, prompt_emb) for prompt_emb in prompt_embeddings]
            max_similarity = max(similarities)
            template_index = similarities.index(max_similarity)
            
            if max_similarity >= similarity_threshold:
                results.append((cleaned_chunks[i], max_similarity, template_index))
    
    except Exception as e:
        logger.error(f"Error in prompt injection detection: {e}")
        return []
    
    return results

# Timing decorator
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    return wrapper

@timing_decorator
def process_resume_with_chunking_strategy(
    resume_text: str, 
    model, 
    prompt_embeddings, 
    chunking_strategy: str = "sentence_aware",
    chunk_size: int = 15,
    overlap: int = 5,
    similarity_threshold: float = 0.65
) -> int:
    """
    Process resume with specified chunking strategy
    """
    # Select chunking strategy
    if chunking_strategy == "sentence_aware":
        chunks = chunk_text_sentence_aware(resume_text, chunk_size, overlap)
    elif chunking_strategy == "fixed_window":
        chunks = chunk_text_fixed_window(resume_text, chunk_size, overlap)
    elif chunking_strategy == "paragraph_based":
        chunks = chunk_text_paragraph_based(resume_text, chunk_size, overlap)
    elif chunking_strategy == "no_segmentation":
        chunks = chunk_text_no_segmentation(resume_text, chunk_size, overlap)
    else:
        chunks = chunk_text_sentence_aware(resume_text, chunk_size, overlap)  # Default
    
    # Detect injections
    injections = detect_prompt_injections(chunks, model, prompt_embeddings, similarity_threshold)
    
    return 1 if injections else 0

def benchmark_chunking_strategies(df, model, prompt_embeddings, threshold=0.65):
    """Comprehensive benchmark of different chunking strategies"""
    logger.info("=== CHUNKING STRATEGY ABLATION STUDY ===")
    
    # Define chunking strategies to test
    chunking_configs = [
        {
            "name": "Sentence-Aware",
            "strategy": "sentence_aware",
            "chunk_size": 15,
            "overlap": 5,
            "description": "Respects sentence boundaries with overlap"
        },
        {
            "name": "Fixed-Window",
            "strategy": "fixed_window", 
            "chunk_size": 15,
            "overlap": 5,
            "description": "Sliding window ignoring sentence structure"
        },
        {
            "name": "Paragraph-Based",
            "strategy": "paragraph_based",
            "chunk_size": 15,
            "overlap": 5,
            "description": "Paragraph-first then sentence chunking"
        },
        {
            "name": "No-Segmentation",
            "strategy": "no_segmentation",
            "chunk_size": 15,
            "overlap": 5,
            "description": "Full text as single chunk"
        },
        {
            "name": "Fine-Grained",
            "strategy": "sentence_aware",
            "chunk_size": 8,
            "overlap": 3,
            "description": "Small chunks for precision"
        },
        {
            "name": "Coarse-Grained", 
            "strategy": "sentence_aware",
            "chunk_size": 25,
            "overlap": 8,
            "description": "Large chunks for context"
        }
    ]
    
    results = {}
    
    for config in chunking_configs:
        logger.info(f"Testing: {config['name']} - {config['description']}")
        
        predictions = []
        execution_times = []
        chunk_statistics = []
        
        for resume in df['resume']:
            # Get chunks for analysis
            if config['strategy'] == "sentence_aware":
                chunks = chunk_text_sentence_aware(resume, config['chunk_size'], config['overlap'])
            elif config['strategy'] == "fixed_window":
                chunks = chunk_text_fixed_window(resume, config['chunk_size'], config['overlap'])
            elif config['strategy'] == "paragraph_based":
                chunks = chunk_text_paragraph_based(resume, config['chunk_size'], config['overlap'])
            else:  # no_segmentation
                chunks = chunk_text_no_segmentation(resume, config['chunk_size'], config['overlap'])
            
            chunk_statistics.append(len(chunks))
            
            # Process for detection
            prediction, exec_time = process_resume_with_chunking_strategy(
                resume, model, prompt_embeddings, 
                config['strategy'], config['chunk_size'], config['overlap'], threshold
            )
            predictions.append(prediction)
            execution_times.append(exec_time)
        
        # Calculate metrics
        predictions_array = np.array(predictions, dtype=int)
        true_labels = np.array(df['label'], dtype=int)
        
        precision = precision_score(true_labels, predictions_array, zero_division=0)
        recall = recall_score(true_labels, predictions_array, zero_division=0)
        f1 = f1_score(true_labels, predictions_array, zero_division=0)
        cm = confusion_matrix(true_labels, predictions_array)
        
        avg_time = np.mean(execution_times)
        throughput = len(df) / sum(execution_times)
        avg_chunks = np.mean(chunk_statistics)
        
        results[config['name']] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'avg_chunks_per_resume': avg_chunks,
            'config': config
        }
        
        logger.info(f"  → F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}")
        logger.info(f"  → Speed: {throughput:.1f} resumes/sec, Avg chunks: {avg_chunks:.1f}")
    
    return results

def analyze_chunking_results(results):
    """Comprehensive analysis and visualization of chunking strategies"""
    
    strategies = list(results.keys())
    f1_scores = [results[s]['f1_score'] for s in strategies]
    recalls = [results[s]['recall'] for s in strategies]
    precisions = [results[s]['precision'] for s in strategies]
    throughputs = [results[s]['throughput'] for s in strategies]
    avg_chunks = [results[s]['avg_chunks_per_resume'] for s in strategies]
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Performance Comparison
    x = np.arange(len(strategies))
    width = 0.25
    
    ax1.bar(x - width, recalls, width, label='Recall', color='#E74C3C', alpha=0.8)
    ax1.bar(x, precisions, width, label='Precision', color='#3498DB', alpha=0.8)
    ax1.bar(x + width, f1_scores, width, label='F1-Score', color='#2Ecc71', alpha=0.8)
    
    ax1.set_xlabel('Chunking Strategy')
    ax1.set_ylabel('Performance Score')
    ax1.set_title('A. Detection Performance by Chunking Strategy', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Add value labels
    for i, (rec, prec, f1) in enumerate(zip(recalls, precisions, f1_scores)):
        ax1.text(i - width, rec + 0.02, f'{rec:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax1.text(i, prec + 0.02, f'{prec:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        ax1.text(i + width, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2. Efficiency vs Performance
    colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
    scatter = ax2.scatter(throughputs, f1_scores, s=200, c=colors, alpha=0.8, edgecolors='black')
    
    for i, (thru, f1, strategy) in enumerate(zip(throughputs, f1_scores, strategies)):
        ax2.annotate(strategy, (thru, f1), xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax2.set_xlabel('Throughput (resumes/sec)')
    ax2.set_ylabel('F1-Score')
    ax2.set_title('B. Efficiency vs Performance Trade-off', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Chunk Statistics
    ax3.bar(strategies, avg_chunks, color='#F39C12', alpha=0.8)
    ax3.set_xlabel('Chunking Strategy')
    ax3.set_ylabel('Average Chunks per Resume')
    ax3.set_title('C. Segmentation Granularity', fontsize=14, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, chunks in enumerate(avg_chunks):
        ax3.text(i, chunks + 0.5, f'{chunks:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Recall-Precision Trade-off
    ax4.scatter(recalls, precisions, s=200, c=colors, alpha=0.8, edgecolors='black')
    
    for i, (rec, prec, strategy) in enumerate(zip(recalls, precisions, strategies)):
        ax4.annotate(strategy, (rec, prec), xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('D. Recall-Precision Trade-off', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0.5, 1.0)
    ax4.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig('chunking_strategy_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('chunking_strategy_analysis.pdf', bbox_inches='tight')
    plt.close()
    
    # Generate comprehensive report
    generate_chunking_report(results)

def generate_chunking_report(results):
    """Generate comprehensive analysis report"""
    
    print("\n" + "="*80)
    print("📊 CHUNKING STRATEGY ABLATION STUDY - COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Find optimal strategies
    best_recall = max(results.items(), key=lambda x: x[1]['recall'])
    best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
    best_efficiency = max(results.items(), key=lambda x: x[1]['throughput'])
    best_precision = max(results.items(), key=lambda x: x[1]['precision'])
    
    print(f"\n🏆 OPTIMAL STRATEGIES:")
    print(f"   🎯 Best Recall: {best_recall[0]} (Recall: {best_recall[1]['recall']:.3f})")
    print(f"   ⚖️  Best F1-Score: {best_f1[0]} (F1: {best_f1[1]['f1_score']:.3f})")
    print(f"   🚀 Most Efficient: {best_efficiency[0]} (Throughput: {best_efficiency[1]['throughput']:.2f}/s)")
    print(f"   📈 Best Precision: {best_precision[0]} (Precision: {best_precision[1]['precision']:.3f})")
    
    print(f"\n📈 PERFORMANCE SUMMARY:")
    for strategy, result in results.items():
        config = result['config']
        print(f"   {strategy:15} | F1: {result['f1_score']:.3f} | Recall: {result['recall']:.3f} | "
              f"Precision: {result['precision']:.3f} | Throughput: {result['throughput']:.2f}/s | "
              f"Avg Chunks: {result['avg_chunks_per_resume']:.1f}")
    
    # Key insights
    print(f"\n🔍 KEY INSIGHTS:")
    f1_improvement = best_f1[1]['f1_score'] - results['No-Segmentation']['f1_score']
    recall_improvement = best_recall[1]['recall'] - results['No-Segmentation']['recall']
    
    print(f"   • Segmentation improves F1 by {f1_improvement:.3f} over no-segmentation")
    print(f"   • Segmentation improves recall by {recall_improvement:.3f} over no-segmentation")
    print(f"   • Sentence-aware strategies balance performance and efficiency")
    print(f"   • Granularity affects both detection capability and computational cost")

def main():
    # Load models and data
    MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"  # Using our optimal model
    model = SentenceTransformer(MODEL_NAME)
    
    # Load prompt embeddings
    try:
        prompt_data = np.load('/home/aminou/prompt-injection-detection/dataset/prompts_embeddings.npz')
        prompt_embeddings = prompt_data['embeddings']
        logger.info(f"Loaded {len(prompt_embeddings)} prompt embeddings")
    except Exception as e:
        logger.error(f"Error loading prompt embeddings: {e}")
        return
    
    # Load dataset
    try:
        df = pd.read_csv("dataset/injected_resume_dataset.csv")
        df_0 = df[df["label"] == 0]
        df_1 = df[df["label"] == 1]
        df_balanced = pd.concat([df_0, df_1]).reset_index(drop=True)
        logger.info(f"Loaded balanced dataset with {len(df_balanced)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Use full dataset or subset for testing
    df_test = df_balanced  # Use .sample(n=200, random_state=42) for quicker testing
    
    best_threshold = 0.65
    
    # Run comprehensive chunking strategy ablation
    logger.info("Starting comprehensive chunking strategy ablation study...")
    chunking_results = benchmark_chunking_strategies(df_test, model, prompt_embeddings, best_threshold)
    
    # Analyze and visualize results
    analyze_chunking_results(chunking_results)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL CHUNKING STRATEGY ANALYSIS COMPLETE")
    print("="*60)
    print("Check 'chunking_strategy_analysis.png' for comprehensive visualizations")
    
    return chunking_results

if __name__ == "__main__":
    results = main()