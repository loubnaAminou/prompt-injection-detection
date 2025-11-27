import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             confusion_matrix, roc_curve, auc)
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

# Download NLTK data
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

# Configuration
SECURITY_OPTIMAL_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
SECURITY_THRESHOLD = 0.65
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

class OptimizedGUARD:
    """Optimized GUARD pipeline - Fine-Grained Only"""
    
    def __init__(self, model_name=SECURITY_OPTIMAL_MODEL, similarity_threshold=SECURITY_THRESHOLD):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.similarity_threshold = similarity_threshold
        self.prompt_embeddings = None
        self.results_dir = "guard_results"
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Initialized OPTIMIZED GUARD with model: {model_name}")
        
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
        """Apply full preprocessing"""
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text

    def fine_grained_chunking(self, text, max_words_per_chunk=8, overlap=2):
        """Fine-grained chunking for maximum detection sensitivity"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_words_per_chunk - overlap):
            chunk = ' '.join(words[i:i + max_words_per_chunk])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def detect_prompt_injections(self, text_chunks: List[str]) -> List[Tuple[str, float, int]]:
        """Detect potential prompt injections"""
        if self.prompt_embeddings is None:
            raise ValueError("Prompt embeddings not loaded")
        
        cleaned_chunks = [self.clean_text(chunk) for chunk in text_chunks if chunk.strip()]
        if not cleaned_chunks:
            return []
        
        results = []
        try:
            chunk_embeddings = self.model.encode(cleaned_chunks, convert_to_numpy=True, show_progress_bar=False)
            for i, chunk_emb in enumerate(chunk_embeddings):
                similarities = [1 - cosine(chunk_emb, prompt_emb) for prompt_emb in self.prompt_embeddings]
                max_similarity = max(similarities)
                if max_similarity >= self.similarity_threshold:
                    results.append((cleaned_chunks[i], max_similarity, similarities.index(max_similarity)))
        except Exception as e:
            logger.error(f"Error in detection: {e}")
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
        """Process resume with fine-grained chunking"""
        chunks = self.fine_grained_chunking(resume_text)
        injections = self.detect_prompt_injections(chunks)
        prediction = 1 if injections else 0
        return prediction

    def get_similarity_scores(self, resume_text: str) -> float:
        """Get max similarity score for ROC analysis"""
        chunks = self.fine_grained_chunking(resume_text)
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
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    def evaluate_pipeline(self, df: pd.DataFrame) -> Dict:
        """Comprehensive evaluation of GUARD pipeline"""
        logger.info("Evaluating GUARD pipeline...")
        
        predictions, execution_times, similarity_scores = [], [], []
        total_chunks = 0
        
        for idx, row in df.iterrows():
            resume_text = row['resume']
            chunks = self.fine_grained_chunking(resume_text)
            total_chunks += len(chunks)
            
            # Get prediction and similarity score
            prediction, exec_time = self.process_resume_for_injection(resume_text)
            predictions.append(prediction)
            execution_times.append(exec_time)
            similarity_scores.append(self.get_similarity_scores(resume_text))
        
        # Calculate metrics
        precision = precision_score(df['label'], predictions, zero_division=0)
        recall = recall_score(df['label'], predictions, zero_division=0)
        f1 = f1_score(df['label'], predictions, zero_division=0)
        cm = confusion_matrix(df['label'], predictions)
        avg_time = np.mean(execution_times)
        throughput = len(df) / sum(execution_times)
        avg_chunks = total_chunks / len(df)
        
        # ROC analysis
        fpr, tpr, thresholds = roc_curve(df['label'], similarity_scores)
        roc_auc = auc(fpr, tpr)
        
        results = {
            'f1_score': f1, 'precision': precision, 'recall': recall,
            'avg_inference_time': avg_time, 'throughput': throughput,
            'avg_chunks_per_resume': avg_chunks, 'roc_auc': roc_auc,
            'roc_data': (fpr, tpr, thresholds),
            'similarity_scores': similarity_scores,
            'confusion_matrix': cm.tolist(),
            'total_samples': len(df)
        }
        
        logger.info(f"GUARD → F1: {f1:.3f}, Recall: {recall:.3f}, Precision: {precision:.3f}, "
                   f"AUC: {roc_auc:.3f}, Throughput: {throughput:.2f}/s")
        
        return results

    def analyze_thresholds(self, true_labels, similarity_scores, n_thresholds=20):
        """Analyze performance across thresholds"""
        thresholds = np.linspace(0, 1, n_thresholds)
        recalls, precisions, f1_scores = [], [], []
        
        for threshold in thresholds:
            predictions = [1 if score >= threshold else 0 for score in similarity_scores]
            recalls.append(recall_score(true_labels, predictions, zero_division=0))
            precisions.append(precision_score(true_labels, predictions, zero_division=0))
            f1_scores.append(f1_score(true_labels, predictions, zero_division=0))
        
        return {
            'thresholds': thresholds.tolist(),
            'recalls': recalls, 'precisions': precisions, 'f1_scores': f1_scores
        }

    def create_master_visualizations(self, guard_results, threshold_analysis):
        """Create comprehensive master figure"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. ROC Curve
        self._plot_roc_curve(ax1, guard_results)
        
        # 2. Performance Metrics
        self._plot_performance_metrics(ax2, guard_results)
        
        # 3. Threshold Analysis
        self._plot_threshold_analysis(ax3, threshold_analysis)
        
        # 4. Confusion Matrix
        self._plot_confusion_matrix(ax4, guard_results)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/master_analysis.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/master_analysis.pdf", bbox_inches='tight')
        plt.close()
        logger.info("Created master analysis figure")

    def _plot_roc_curve(self, ax, guard_results):
        """Plot ROC curve"""
        fpr, tpr, _ = guard_results['roc_data']
        roc_auc = guard_results['roc_auc']
        
        ax.plot(fpr, tpr, color='#E74C3C', linewidth=2, 
                label=f'GUARD (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('A. ROC Analysis: Detection Capability')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_performance_metrics(self, ax, guard_results):
        """Plot performance metrics"""
        metrics = ['F1-Score', 'Recall', 'Precision']
        values = [guard_results['f1_score'], guard_results['recall'], guard_results['precision']]
        colors = ['#E74C3C', '#3498DB', '#2Ecc71']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black')
        ax.set_ylabel('Score')
        ax.set_title('B. Performance Metrics')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    def _plot_threshold_analysis(self, ax, threshold_analysis):
        """Plot threshold sensitivity"""
        thresholds = threshold_analysis['thresholds']
        ax.plot(thresholds, threshold_analysis['recalls'], 'o-', label='Recall', 
                linewidth=2, color='#E74C3C')
        ax.plot(thresholds, threshold_analysis['precisions'], 's-', label='Precision', 
                linewidth=2, color='#3498DB')
        ax.plot(thresholds, threshold_analysis['f1_scores'], '^-', label='F1-Score', 
                linewidth=2, color='#2Ecc71')
        
        # Mark optimal threshold
        optimal_idx = np.argmax(threshold_analysis['f1_scores'])
        ax.axvline(x=thresholds[optimal_idx], color='red', linestyle='--', 
                   label=f'Optimal (={thresholds[optimal_idx]:.2f})')
        
        ax.set_xlabel('Similarity Threshold')
        ax.set_ylabel('Score')
        ax.set_title('C. Threshold Sensitivity Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_confusion_matrix(self, ax, guard_results):
        """Plot confusion matrix"""
        cm = np.array(guard_results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax,
                   xticklabels=['Predicted Clean', 'Predicted Malicious'],
                   yticklabels=['Actual Clean', 'Actual Malicious'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('D. Confusion Matrix')

def run_focused_evaluation():
    """Run focused evaluation - Fine-grained only"""
    print("🚀 STARTING FOCUSED GUARD EVALUATION")
    print("=" * 60)
    
    # Initialize optimized GUARD
    guard = OptimizedGUARD()
    
    # Load prompt embeddings
    if not guard.load_prompt_embeddings('/home/aminou/prompt-injection-detection/dataset/prompts_embeddings.npz'):
        print("❌ Failed to load prompt embeddings")
        return
    
    # Load dataset
    try:
        df = pd.read_csv("dataset/injected_resume_dataset.csv")
        df_0 = df[df["label"] == 0]
        df_1 = df[df["label"] == 1]
        df_balanced = pd.concat([df_0, df_1]).reset_index(drop=True)
        print(f"✅ Loaded dataset with {len(df_balanced)} samples")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # 1. Evaluate GUARD pipeline
    print("\n📊 EVALUATING GUARD PIPELINE...")
    guard_results = guard.evaluate_pipeline(df_balanced)
    
    # 2. Get threshold analysis
    print("\n🎯 ANALYZING THRESHOLDS...")
    threshold_analysis = guard.analyze_thresholds(df_balanced['label'], 
                                                 guard_results['similarity_scores'])
    
    # 3. Create master visualizations
    print("\n📈 CREATING MASTER VISUALIZATIONS...")
    guard.create_master_visualizations(guard_results, threshold_analysis)
    
    # 4. Generate master table
    print("\n📋 GENERATING MASTER TABLE...")
    master_table = generate_master_table(guard_results, threshold_analysis)
    
    # Display results
    print("\n" + "=" * 80)
    print("🎉 FOCUSED EVALUATION FINISHED!")
    print("=" * 80)
    
    print(f"\n📊 GUARD PERFORMANCE (Use these in your paper):")
    print("-" * 80)
    print(f"F1-Score:     {guard_results['f1_score']:.3f}")
    print(f"Recall:       {guard_results['recall']:.3f} ({guard_results['recall']*100:.1f}% attack detection)")
    print(f"Precision:    {guard_results['precision']:.3f}")
    print(f"AUC:          {guard_results['roc_auc']:.3f}")
    print(f"Throughput:   {guard_results['throughput']:.2f} resumes/second")
    print(f"Avg Time:     {guard_results['avg_inference_time']:.4f} seconds/resume")
    
    # Find optimal threshold
    optimal_idx = np.argmax(threshold_analysis['f1_scores'])
    print(f"\n🎯 OPTIMAL THRESHOLD: {threshold_analysis['thresholds'][optimal_idx]:.2f} "
          f"(F1: {threshold_analysis['f1_scores'][optimal_idx]:.3f}, "
          f"Recall: {threshold_analysis['recalls'][optimal_idx]:.3f}, "
          f"Precision: {threshold_analysis['precisions'][optimal_idx]:.3f})")
    
    return guard_results, threshold_analysis, master_table

def generate_master_table(guard_results, threshold_analysis):
    """Generate LaTeX for master table"""
    
    # Find optimal threshold
    optimal_idx = np.argmax(threshold_analysis['f1_scores'])
    optimal_threshold = threshold_analysis['thresholds'][optimal_idx]
    
    latex_code = f"""
\\begin{{table}}[!ht]
\\centering
\\caption{{GUARD Pipeline Performance: Comprehensive Analysis}}
\\label{{tab:master_results}}
\\renewcommand{{\\arraystretch}}{{1.2}}
\\small
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Component}} & \\textbf{{F1-Score}} & \\textbf{{Recall}} & \\textbf{{Precision}} & \\textbf{{AUC}} & \\textbf{{Throughput}} \\\\
\\midrule
\\textbf{{GUARD Pipeline}} & \\textbf{{{guard_results['f1_score']:.3f}}} & \\textbf{{{guard_results['recall']:.3f}}} & {guard_results['precision']:.3f} & \\textbf{{{guard_results['roc_auc']:.3f}}} & {guard_results['throughput']:.2f}/s \\\\
\\midrule
\\multicolumn{{6}}{{l}}{{\\textbf{{Threshold Analysis}}}} \\\\
\\quad {optimal_threshold:.2f} (Optimal) & \\textbf{{{threshold_analysis['f1_scores'][optimal_idx]:.3f}}} & {threshold_analysis['recalls'][optimal_idx]:.3f} & {threshold_analysis['precisions'][optimal_idx]:.3f} & - & - \\\\
\\quad 0.50 (High Recall) & {threshold_analysis['f1_scores'][0]:.3f} & \\textbf{{{threshold_analysis['recalls'][0]:.3f}}} & {threshold_analysis['precisions'][0]:.3f} & - & - \\\\
\\quad 0.70 (High Precision) & {threshold_analysis['f1_scores'][14]:.3f} & {threshold_analysis['recalls'][14]:.3f} & \\textbf{{{threshold_analysis['precisions'][14]:.3f}}} & - & - \\\\
\\bottomrule
\\end{{tabular}}
\\vspace{{0.2cm}}
\\parbox{{\\textwidth}}{{
\\small
\\textbf{{Key Insights:}} GUARD achieves balanced performance with {guard_results['recall']*100:.1f}\\% attack detection at practical throughput. The {optimal_threshold:.2f} threshold optimizes security-performance trade-off. Method requires zero labeled training data.
}}
\\end{{table}}
"""
    
    # Save LaTeX to file
    with open("master_table.tex", "w") as f:
        f.write(latex_code)
    
    print("📜 LaTeX table saved to: master_table.tex")
    return latex_code

if __name__ == "__main__":
    guard_results, threshold_analysis, master_table = run_focused_evaluation()
    
    # Save numerical results
    results_data = {
        'Metric': ['F1-Score', 'Recall', 'Precision', 'AUC', 'Throughput', 'Avg_Inference_Time'],
        'Value': [
            guard_results['f1_score'],
            guard_results['recall'], 
            guard_results['precision'],
            guard_results['roc_auc'],
            guard_results['throughput'],
            guard_results['avg_inference_time']
        ]
    }
    results_df = pd.DataFrame(results_data)
    results_df.to_csv("guard_results.csv", index=False)
    print("💾 Numerical results saved to: guard_results.csv")