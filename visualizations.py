import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, auc, precision_recall_curve)
from scipy.spatial.distance import cosine
import re
import string
import logging
from typing import List, Tuple, Dict
import time
from functools import wraps
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# ==================== SECURITY-OPTIMIZED CONFIGURATION ====================
SECURITY_OPTIMAL_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
SECURITY_THRESHOLD = 0.65
# =========================================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

class Q1ResultsGenerator:
    """Generate Q1-quality visualizations for main results"""
    
    def __init__(self, results_dir="q1_results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
    def create_main_results_figures(self, guard_results: Dict, baseline_results: Dict = None):
        """
        Create exactly 4 Q1-quality figures for main results
        """
        # Figure 1: ROC Curve (Essential for detection papers)
        self._create_roc_curves(guard_results, baseline_results)
        
        # Figure 2: Performance Comparison Bar Chart
        self._create_performance_comparison(guard_results, baseline_results)
        
        # Figure 3: Security-Performance Trade-off Analysis
        self._create_threshold_analysis(guard_results)
        
        # Figure 4: Confusion Matrix with Security Context
        self._create_security_confusion_matrix(guard_results)
    
    def _create_roc_curves(self, guard_results: Dict, baseline_results: Dict = None):
        """Figure 1: ROC curves showing detection capability"""
        plt.figure(figsize=(8, 6))
        
        # Plot GUARD ROC curve
        if 'roc_data' in guard_results:
            fpr, tpr, _ = guard_results['roc_data']
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='#E74C3C', lw=2, 
                    label=f'GUARD (AUC = {roc_auc:.3f})')
        
        # Plot baseline ROC curves if available
        if baseline_results:
            colors = ['#3498DB', '#2Ecc71', '#F39C12', '#9B59B6']
            for i, (baseline_name, baseline_data) in enumerate(baseline_results.items()):
                if 'roc_data' in baseline_data:
                    fpr, tpr, _ = baseline_data['roc_data']
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                            label=f'{baseline_name} (AUC = {roc_auc:.3f})')
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')
        
        # Formatting
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves: Prompt Injection Detection Performance', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/roc_curves.pdf", bbox_inches='tight')
        plt.close()
        
        logger.info("Created ROC curves figure")
    
    def _create_performance_comparison(self, guard_results: Dict, baseline_results: Dict = None):
        """Figure 2: Performance metrics comparison"""
        metrics = ['Recall', 'Precision', 'F1-Score']
        
        plt.figure(figsize=(10, 6))
        
        # Set up bar positions
        if baseline_results:
            n_methods = 1 + len(baseline_results)
            x = np.arange(len(metrics))
            width = 0.8 / n_methods
            
            # Plot GUARD performance
            guard_values = [guard_results['recall'], guard_results['precision'], guard_results['f1_score']]
            plt.bar(x - width*(n_methods-1)/2, guard_values, width, 
                   label='GUARD', color='#E74C3C', alpha=0.9, edgecolor='black')
            
            # Plot baseline performances
            colors = ['#3498DB', '#2Ecc71', '#F39C12', '#9B59B6']
            for i, (baseline_name, baseline_data) in enumerate(baseline_results.items()):
                baseline_values = [baseline_data['recall'], baseline_data['precision'], baseline_data['f1_score']]
                plt.bar(x - width*(n_methods-1)/2 + width*(i+1), baseline_values, width,
                       label=baseline_name, color=colors[i % len(colors)], alpha=0.8, edgecolor='black')
        else:
            # Only GUARD results
            guard_values = [guard_results['recall'], guard_results['precision'], guard_results['f1_score']]
            bars = plt.bar(metrics, guard_values, color='#E74C3C', alpha=0.9, edgecolor='black')
            
            # Add value labels
            for bar, value in zip(bars, guard_values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Formatting
        plt.xlabel('Performance Metrics', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('Detection Performance Comparison', fontsize=14, fontweight='bold', pad=15)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Highlight recall as primary security metric
        plt.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, linewidth=1)
        plt.text(len(metrics)-0.5, 0.91, 'Security Target (90% Recall)', 
                fontsize=9, color='red', ha='right')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/performance_comparison.pdf", bbox_inches='tight')
        plt.close()
        
        logger.info("Created performance comparison figure")
    
    def _create_threshold_analysis(self, guard_results: Dict):
        """Figure 3: Security-performance trade-off across thresholds"""
        if 'threshold_analysis' not in guard_results:
            logger.warning("Threshold analysis data not available")
            return
            
        threshold_data = guard_results['threshold_analysis']
        thresholds = threshold_data['thresholds']
        recalls = threshold_data['recalls']
        precisions = threshold_data['precisions']
        f1_scores = threshold_data['f1_scores']
        
        plt.figure(figsize=(10, 6))
        
        # Plot metrics vs thresholds
        plt.plot(thresholds, recalls, 'o-', color='#E74C3C', linewidth=2, 
                label='Recall (Attack Detection)', markersize=4)
        plt.plot(thresholds, precisions, 's-', color='#3498DB', linewidth=2,
                label='Precision (Accuracy)', markersize=4)
        plt.plot(thresholds, f1_scores, '^-', color='#2Ecc71', linewidth=2,
                label='F1-Score (Balance)', markersize=4)
        
        # Highlight chosen threshold
        chosen_threshold = guard_results.get('threshold', 0.65)
        chosen_idx = np.argmin(np.abs(np.array(thresholds) - chosen_threshold))
        plt.axvline(x=chosen_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Chosen Threshold ({chosen_threshold})')
        
        # Annotate chosen threshold performance
        plt.annotate(f'Recall: {recalls[chosen_idx]:.3f}\nPrecision: {precisions[chosen_idx]:.3f}',
                    xy=(chosen_threshold, recalls[chosen_idx]),
                    xytext=(10, 30), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Formatting
        plt.xlabel('Similarity Threshold', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title('Security-Performance Trade-off Analysis', fontsize=14, fontweight='bold', pad=15)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/threshold_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/threshold_analysis.pdf", bbox_inches='tight')
        plt.close()
        
        logger.info("Created threshold analysis figure")
    
    def _create_security_confusion_matrix(self, guard_results: Dict):
        """Figure 4: Confusion matrix with security context"""
        cm = np.array(guard_results['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                   xticklabels=['Predicted Clean', 'Predicted Malicious'],
                   yticklabels=['Actual Clean', 'Actual Malicious'],
                   cbar_kws={'label': 'Count', 'shrink': 0.8})
        
        # Calculate and display key security metrics
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        recall = guard_results['recall']
        precision = guard_results['precision']
        
        # Add security context annotation
        plt.text(0.5, -0.15, 
                f'Security Assessment: {recall:.1%} Attack Detection | {fn} Attacks Missed',
                ha='center', va='center', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3),
                fontweight='bold')
        
        plt.title('Confusion Matrix: GUARD Detection Performance\n'
                 f'Recall: {recall:.3f} | Precision: {precision:.3f}', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        plt.savefig(f"{self.results_dir}/security_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/security_confusion_matrix.pdf", bbox_inches='tight')
        plt.close()
        
        logger.info("Created security confusion matrix figure")


class SecurityOptimizedGUARD:
    """Security-optimized GUARD pipeline - STREAMLINED for Q1 results"""
    
    def __init__(self, model_name=SECURITY_OPTIMAL_MODEL, similarity_threshold=SECURITY_THRESHOLD):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.prompt_embeddings = None
        self.results_dir = "security_guard_results"
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Initialized SECURITY-OPTIMIZED GUARD with model: {model_name}")
        
    def load_prompt_embeddings(self, embeddings_path):
        """Load pre-computed prompt embeddings"""
        try:
            prompt_data = np.load(embeddings_path)
            self.prompt_embeddings = prompt_data['embeddings']
            logger.info(f"Loaded {len(self.prompt_embeddings)} prompt embeddings")
            return True
        except Exception as e:
            logger.error(f"Error loading prompt embeddings: {e}")
            return False

    def clean_text(self, text):
        """Apply full preprocessing to resume chunks and prompts"""
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text

    def chunk_text(self, text, chunk_size=15, overlap=5):
        """Improved text chunking with sentence awareness"""
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

    def detect_prompt_injections(self, text_chunks: List[str]) -> List[Tuple[str, float, int]]:
        """Detect potential prompt injections in text chunks"""
        if self.prompt_embeddings is None:
            raise ValueError("Prompt embeddings not loaded. Call load_prompt_embeddings first.")
        
        cleaned_chunks = [self.clean_text(chunk) for chunk in text_chunks if chunk.strip()]
        
        if not cleaned_chunks:
            return []
        
        results = []
        
        try:
            chunk_embeddings = self.model.encode(cleaned_chunks, convert_to_numpy=True, show_progress_bar=False)
            
            for i, chunk_emb in enumerate(chunk_embeddings):
                similarities = [1 - cosine(chunk_emb, prompt_emb) for prompt_emb in self.prompt_embeddings]
                max_similarity = max(similarities)
                template_index = similarities.index(max_similarity)
                
                if max_similarity >= self.similarity_threshold:
                    results.append((cleaned_chunks[i], max_similarity, template_index))
        
        except Exception as e:
            logger.error(f"Error in prompt injection detection: {e}")
            return []
        
        return results

    def timing_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            return result, execution_time
        return wrapper

    @timing_decorator
    def process_resume_for_injection(self, resume_text: str) -> Tuple[int, float]:
        """Process resume and return (prediction, execution_time)"""
        chunks = self.chunk_text(resume_text)
        injections = self.detect_prompt_injections(chunks)
        prediction = 1 if injections else 0
        return prediction

    def evaluate_on_dataset(self, df: pd.DataFrame, dataset_name="main") -> Dict:
        """
        Comprehensive evaluation on dataset with Q1-quality results
        """
        logger.info(f"Evaluating SECURITY-GUARD on {dataset_name} dataset...")
        
        predictions = []
        execution_times = []
        similarity_scores = []  # Store for ROC analysis
        
        for idx, row in df.iterrows():
            resume_text = row['resume']
            true_label = row['label']
            
            prediction, exec_time = self.process_resume_for_injection(resume_text)
            predictions.append(prediction)
            execution_times.append(exec_time)
            
            # For ROC analysis, we need similarity scores instead of binary predictions
            # This is a simplified approach - you might need to modify based on your actual scoring
            max_similarity = self._get_max_similarity(resume_text)
            similarity_scores.append(max_similarity)
        
        # Calculate basic metrics
        precision = precision_score(df['label'], predictions, zero_division=0)
        recall = recall_score(df['label'], predictions, zero_division=0)
        f1 = f1_score(df['label'], predictions, zero_division=0)
        cm = confusion_matrix(df['label'], predictions)
        
        # Performance statistics
        avg_time = np.mean(execution_times)
        throughput = len(df) / sum(execution_times)
        
        # Generate ROC data
        fpr, tpr, thresholds = roc_curve(df['label'], similarity_scores)
        roc_data = (fpr, tpr, thresholds)
        
        # Generate threshold analysis data
        threshold_analysis = self._analyze_thresholds(df['label'], similarity_scores)
        
        results = {
            'dataset': dataset_name,
            'model': self.model_name,
            'threshold': self.similarity_threshold,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'avg_inference_time': avg_time,
            'throughput': throughput,
            'total_samples': len(df),
            'roc_data': roc_data,
            'threshold_analysis': threshold_analysis,
            'similarity_scores': similarity_scores
        }
        
        # Generate Q1-quality visualizations
        q1_viz = Q1ResultsGenerator()
        q1_viz.create_main_results_figures(results)
        
        # Save results
        self._save_results(results, dataset_name)
        
        return results

    def _get_max_similarity(self, resume_text: str) -> float:
        """Get maximum similarity score for ROC analysis"""
        chunks = self.chunk_text(resume_text)
        cleaned_chunks = [self.clean_text(chunk) for chunk in chunks if chunk.strip()]
        
        if not cleaned_chunks or self.prompt_embeddings is None:
            return 0.0
        
        try:
            chunk_embeddings = self.model.encode(cleaned_chunks, convert_to_numpy=True, show_progress_bar=False)
            max_similarities = []
            
            for chunk_emb in chunk_embeddings:
                similarities = [1 - cosine(chunk_emb, prompt_emb) for prompt_emb in self.prompt_embeddings]
                max_similarities.append(max(similarities))
            
            return max(max_similarities) if max_similarities else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating max similarity: {e}")
            return 0.0

    def _analyze_thresholds(self, true_labels, similarity_scores, n_thresholds=20):
        """Analyze performance across different thresholds"""
        thresholds = np.linspace(0, 1, n_thresholds)
        recalls = []
        precisions = []
        f1_scores = []
        
        for threshold in thresholds:
            predictions = [1 if score >= threshold else 0 for score in similarity_scores]
            recalls.append(recall_score(true_labels, predictions, zero_division=0))
            precisions.append(precision_score(true_labels, predictions, zero_division=0))
            f1_scores.append(f1_score(true_labels, predictions, zero_division=0))
        
        return {
            'thresholds': thresholds.tolist(),
            'recalls': recalls,
            'precisions': precisions,
            'f1_scores': f1_scores
        }

    def _save_results(self, results: Dict, dataset_name: str):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/security_guard_results_{dataset_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Security results saved to {filename}")

def run_security_evaluation():
    """Run comprehensive evaluation of security-optimized GUARD"""
    
    # Initialize security-optimized GUARD
    guard = SecurityOptimizedGUARD()
    
    # Load prompt embeddings
    if not guard.load_prompt_embeddings('/home/aminou/prompt-injection-detection/dataset/prompts_embeddings.npz'):
        return
    
    # Load dataset
    try:
        df = pd.read_csv("dataset/injected_resume_dataset.csv")
        df_0 = df[df["label"] == 0]
        df_1 = df[df["label"] == 1]
        df = pd.concat([df_0, df_1]).reset_index(drop=True)
        logger.info(f"Loaded dataset with {len(df)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Run comprehensive evaluation
    results = guard.evaluate_on_dataset(df, "security_evaluation")
    
    return results

if __name__ == "__main__":
    # Run the security-focused evaluation
    security_results = run_security_evaluation()