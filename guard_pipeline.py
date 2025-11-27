import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
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
# Based on security-first analysis - using the best recall model
SECURITY_OPTIMAL_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"  # 91.7% recall - BEST for security
SECURITY_THRESHOLD = 0.65  # Can be tuned for optimal security recall
# =========================================================================

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

class ErrorAnalyzer:
    """Comprehensive error analysis for security pipeline"""
    
    def __init__(self, guard_pipeline):
        self.guard = guard_pipeline
        self.error_results = {}
        
    def analyze_errors(self, df: pd.DataFrame, predictions: List[int], dataset_name: str):
        """Comprehensive error analysis"""
        
        # Basic error identification
        df_analysis = df.copy()
        df_analysis['prediction'] = predictions
        df_analysis['correct'] = df_analysis['label'] == df_analysis['prediction']
        df_analysis['error_type'] = self._categorize_errors(df_analysis)
        
        # Detailed error analysis
        error_analysis = {
            'false_negatives': self._analyze_false_negatives(df_analysis),
            'false_positives': self._analyze_false_positives(df_analysis),
            'error_patterns': self._identify_error_patterns(df_analysis),
            'confidence_analysis': self._analyze_confidence_scores(df_analysis),
            'text_characteristics': self._analyze_text_characteristics(df_analysis)
        }
        
        self._save_error_analysis(error_analysis, dataset_name)
        self._create_error_visualizations(error_analysis, dataset_name)
        
        return error_analysis
    
    def _categorize_errors(self, df: pd.DataFrame) -> List[str]:
        """Categorize different types of errors"""
        error_types = []
        for idx, row in df.iterrows():
            if row['correct']:
                error_types.append('correct')
            elif row['label'] == 1 and row['prediction'] == 0:
                error_types.append('false_negative')
            else:
                error_types.append('false_positive')
        return error_types
    
    def _analyze_false_negatives(self, df: pd.DataFrame) -> Dict:
        """Analyze missed attacks (false negatives)"""
        fn_df = df[df['error_type'] == 'false_negative']
        
        analysis = {
            'count': len(fn_df),
            'examples': [],
            'common_patterns': self._extract_common_patterns(fn_df['resume'].tolist()),
            'length_analysis': self._analyze_text_lengths(fn_df['resume'].tolist()),
            'complexity_metrics': self._analyze_text_complexity(fn_df['resume'].tolist())
        }
        
        # Sample problematic cases for detailed examination
        for idx, row in fn_df.head(5).iterrows():
            analysis['examples'].append({
                'text_preview': row['resume'][:200] + "...",
                'full_text': row['resume'],
                'index': idx
            })
        
        return analysis
    
    def _analyze_false_positives(self, df: pd.DataFrame) -> Dict:
        """Analyze false alarms (false positives)"""
        fp_df = df[df['error_type'] == 'false_positive']
        
        analysis = {
            'count': len(fp_df),
            'examples': [],
            'common_clean_patterns': self._extract_common_patterns(fp_df['resume'].tolist()),
            'potential_triggers': self._identify_false_positive_triggers(fp_df['resume'].tolist())
        }
        
        for idx, row in fp_df.head(5).iterrows():
            analysis['examples'].append({
                'text_preview': row['resume'][:200] + "...",
                'full_text': row['resume'],
                'index': idx
            })
        
        return analysis
    
    def _identify_error_patterns(self, df: pd.DataFrame) -> Dict:
        """Identify systematic patterns in errors"""
        error_df = df[df['error_type'].isin(['false_negative', 'false_positive'])]
        
        patterns = {
            'length_correlation': self._correlate_length_with_errors(df),
            'keyword_analysis': self._analyze_error_keywords(error_df),
            'structural_patterns': self._analyze_structural_patterns(error_df)
        }
        
        return patterns
    
    def _analyze_confidence_scores(self, df: pd.DataFrame) -> Dict:
        """Analyze model confidence for different prediction types"""
        # This would require modifying your pipeline to return confidence scores
        confidence_analysis = {
            'confidence_distribution': {
                'correct_predictions': {'mean': 0.0, 'std': 0.0},
                'false_negatives': {'mean': 0.0, 'std': 0.0},
                'false_positives': {'mean': 0.0, 'std': 0.0}
            },
            'threshold_analysis': self._analyze_threshold_impact(df)
        }
        
        return confidence_analysis
    
    def _analyze_text_characteristics(self, df: pd.DataFrame) -> Dict:
        """Analyze text characteristics correlation with errors"""
        characteristics = {
            'text_lengths': self._analyze_text_length_correlation(df),
            'vocabulary_richness': self._analyze_vocabulary_complexity(df),
            'syntactic_complexity': self._analyze_syntactic_patterns(df),
            'semantic_patterns': self._analyze_semantic_patterns(df)
        }
        
        return characteristics
    
    def _extract_common_patterns(self, texts: List[str]) -> List[Dict]:
        """Extract common patterns from text samples"""
        patterns = []
        
        # Analyze common n-grams in error cases
        from collections import Counter
        all_words = []
        for text in texts:
            words = self.guard.clean_text(text).split()
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        common_words = word_freq.most_common(20)
        
        for word, freq in common_words:
            if freq > 1:  # Only include patterns that appear multiple times
                patterns.append({
                    'pattern': word,
                    'frequency': freq,
                    'percentage': (freq / len(texts)) * 100
                })
        
        return patterns
    
    def _analyze_text_lengths(self, texts: List[str]) -> Dict:
        """Analyze text length characteristics"""
        lengths = [len(text.split()) for text in texts]
        
        return {
            'mean_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'min_length': np.min(lengths),
            'max_length': np.max(lengths),
            'length_distribution': lengths
        }
    
    def _correlate_length_with_errors(self, df: pd.DataFrame) -> Dict:
        """Correlate text length with error types"""
        df['text_length'] = df['resume'].apply(lambda x: len(x.split()))
        
        length_stats = {
            'all_samples': df['text_length'].describe().to_dict(),
            'correct_predictions': df[df['correct']]['text_length'].describe().to_dict(),
            'false_negatives': df[df['error_type'] == 'false_negative']['text_length'].describe().to_dict(),
            'false_positives': df[df['error_type'] == 'false_positive']['text_length'].describe().to_dict()
        }
        
        return length_stats
    
    def create_comprehensive_error_report(self, error_analysis: Dict, dataset_name: str):
        """Generate comprehensive error analysis report"""
        
        # Create error analysis visualization
        self._create_error_breakdown_chart(error_analysis, dataset_name)
        self._create_pattern_analysis_chart(error_analysis, dataset_name)
        self._create_case_studies_visualization(error_analysis, dataset_name)
        
        # Print detailed report
        self._print_error_analysis_report(error_analysis)
    
    def _create_error_breakdown_chart(self, error_analysis: Dict, dataset_name: str):
        """Create visualization of error breakdown"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Error type distribution
        error_types = ['Correct', 'False Negatives', 'False Positives']
        error_counts = [
            error_analysis.get('total_correct', 0),
            error_analysis['false_negatives']['count'],
            error_analysis['false_positives']['count']
        ]
        
        ax1.bar(error_types, error_counts, color=['green', 'red', 'orange'])
        ax1.set_title('Error Type Distribution')
        ax1.set_ylabel('Count')
        
        # Text length correlation
        fn_lengths = error_analysis['error_patterns']['length_correlation']['false_negatives']
        fp_lengths = error_analysis['error_patterns']['length_correlation']['false_positives']
        
        ax2.boxplot([fn_lengths['length_distribution'], fp_lengths['length_distribution']], 
                   labels=['False Negatives', 'False Positives'])
        ax2.set_title('Text Length Distribution by Error Type')
        ax2.set_ylabel('Text Length (words)')
        
        # Common patterns in false negatives
        fn_patterns = error_analysis['false_negatives']['common_patterns'][:10]
        patterns = [p['pattern'] for p in fn_patterns]
        frequencies = [p['frequency'] for p in fn_patterns]
        
        ax3.barh(patterns, frequencies, color='red', alpha=0.7)
        ax3.set_title('Top Patterns in False Negatives')
        ax3.set_xlabel('Frequency')
        
        # Common patterns in false positives
        fp_patterns = error_analysis['false_positives']['common_clean_patterns'][:10]
        patterns = [p['pattern'] for p in fp_patterns]
        frequencies = [p['frequency'] for p in fp_patterns]
        
        ax4.barh(patterns, frequencies, color='orange', alpha=0.7)
        ax4.set_title('Top Patterns in False Positives')
        ax4.set_xlabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{self.guard.results_dir}/error_analysis_breakdown_{dataset_name}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_error_analysis_report(self, error_analysis: Dict):
        """Print comprehensive error analysis report"""
        print("\n" + "="*80)
        print("🔍 COMPREHENSIVE ERROR ANALYSIS REPORT")
        print("="*80)
        
        fn_count = error_analysis['false_negatives']['count']
        fp_count = error_analysis['false_positives']['count']
        total_errors = fn_count + fp_count
        
        print(f"\n📊 ERROR BREAKDOWN:")
        print(f"   ❌ False Negatives (Missed Attacks): {fn_count}")
        print(f"   ⚠️  False Positives (False Alarms): {fp_count}")
        print(f"   📈 Total Errors: {total_errors}")
        
        print(f"\n🔍 FALSE NEGATIVE ANALYSIS (MISSED ATTACKS):")
        if fn_count > 0:
            fn_patterns = error_analysis['false_negatives']['common_patterns'][:5]
            print(f"   Top patterns in missed attacks:")
            for pattern in fn_patterns:
                print(f"     - '{pattern['pattern']}': {pattern['frequency']} occurrences")
            
            print(f"\n   Text characteristics:")
            length_stats = error_analysis['false_negatives']['length_analysis']
            print(f"     Average length: {length_stats['mean_length']:.1f} words")
            print(f"     Length range: {length_stats['min_length']} - {length_stats['max_length']} words")
        
        print(f"\n⚠️  FALSE POSITIVE ANALYSIS (FALSE ALARMS):")
        if fp_count > 0:
            fp_patterns = error_analysis['false_positives']['common_clean_patterns'][:5]
            print(f"   Top patterns triggering false alarms:")
            for pattern in fp_patterns:
                print(f"     - '{pattern['pattern']}': {pattern['frequency']} occurrences")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if fn_count > fp_count:
            print("   Primary concern: Too many attacks are being missed")
            print("   Suggested action: Lower similarity threshold or improve attack pattern detection")
        else:
            print("   Primary concern: Too many false alarms")
            print("   Suggested action: Increase similarity threshold or improve clean text understanding")
        
        print(f"\n💡 SPECIFIC INSIGHTS:")
        # Add specific insights based on patterns found
        if any('ignore' in p['pattern'] for p in error_analysis['false_negatives']['common_patterns']):
            print("   - 'Ignore' commands are frequently missed - consider enhancing detection for obedience keywords")
        
        if any('previous' in p['pattern'] for p in error_analysis['false_positives']['common_clean_patterns']):
            print("   - Historical context phrases trigger false alarms - adjust for common resume language")


class SecurityOptimizedGUARD:
    """Security-optimized GUARD pipeline with best recall embedding model"""
    
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
        """
        Improved text chunking with sentence awareness.
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

    
    def create_unified_analysis_figure(self, results: Dict, dataset_name: str):
    
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Define the grid layout
        gs = fig.add_gridspec(3, 3)
        
        # 1. Performance Metrics (Top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_performance_metrics(ax1, results)
        
        # 2. Confusion Matrix (Top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_confusion_matrix(ax2, results)
        
        # 3. Security Impact (Top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_security_impact(ax3, results)
        
        # 4. Radar Chart (Middle row - span all columns)
        ax4 = fig.add_subplot(gs[1, :])
        self._plot_radar_comparison(ax4, results)
        
        # 5. Efficiency Trade-off (Bottom-left)
        ax5 = fig.add_subplot(gs[2, 0])
        self._plot_efficiency_tradeoff(ax5, results)
        
        # 6. Performance vs Ideal (Bottom-middle + right)
        ax6 = fig.add_subplot(gs[2, 1:])
        self._plot_performance_gap(ax6, results)
        
        # Main title
        fig.suptitle('Comprehensive Security Analysis: GUARD Pipeline Performance', 
                    fontsize=24, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"{self.results_dir}/unified_analysis_{dataset_name}.png", 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(f"{self.results_dir}/unified_analysis_{dataset_name}.pdf", 
                    bbox_inches='tight', facecolor='white')
        plt.close()

    def _plot_performance_metrics(self, ax, results: Dict):
        """Plot performance metrics bar chart"""
        metrics = ['Recall', 'Precision', 'F1-Score']
        values = [results['recall'], results['precision'], results['f1_score']]
        colors = ['#E74C3C', '#3498DB', '#2Ecc71']  # Red, Blue, Green
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('A. Core Performance Metrics', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # Add performance thresholds
        ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.7, linewidth=1, label='Excellent')
        ax.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, linewidth=1, label='Good')
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, linewidth=1, label='Minimum')

    def _plot_confusion_matrix(self, ax, results: Dict):
        """Plot confusion matrix heatmap"""
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Clean', 'Malicious'],
                yticklabels=['Clean', 'Malicious'],
                cbar_kws={'shrink': 0.6})
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
        ax.set_title('B. Confusion Matrix', fontsize=12, fontweight='bold', pad=10)

    def _plot_security_impact(self, ax, results: Dict):
        """Plot security impact donut chart"""
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        labels = ['Safe Passed', 'Safe Flagged', 'Attack Missed', 'Attack Blocked']
        sizes = [tn, fp, fn, tp]
        colors = ['#2Ecc71', '#F39C12', '#E74C3C', '#3498DB']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 9})
        
        # Draw circle for donut
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        ax.add_artist(centre_circle)
        ax.axis('equal')
        ax.set_title('C. Security Impact Distribution', fontsize=12, fontweight='bold', pad=10)
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(8)

    def _plot_radar_comparison(self, ax, results: Dict):
        """Plot radar chart comparing performance dimensions"""
        categories = ['Recall\n(Detection)', 'Precision\n(Accuracy)', 'F1-Score\n(Balance)', 
                    'Throughput\n(Speed)', 'Efficiency\n(Resources)']
        
        # GUARD values
        guard_recall = results['recall']
        guard_precision = results['precision']
        guard_f1 = results['f1_score']
        guard_throughput = results['throughput'] / 2.0  # Normalized
        guard_efficiency = 1 - (results['avg_inference_time'] / 2.0)  # Normalized
        
        guard_values = [guard_recall, guard_precision, guard_f1, guard_throughput, guard_efficiency]
        
        # Ideal values
        ideal_values = [0.95, 0.85, 0.85, 0.8, 0.9]
        
        # Complete the circle
        guard_values += guard_values[:1]
        ideal_values += ideal_values[:1]
        categories_radar = categories + [categories[0]]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        # Plot
        ax.plot(angles, guard_values, 'o-', linewidth=3, label='GUARD Performance', 
                color='#E74C3C', markersize=8)
        ax.fill(angles, guard_values, alpha=0.25, color='#E74C3C')
        
        ax.plot(angles, ideal_values, 'o-', linewidth=2, label='Ideal Targets', 
                color='#2Ecc71', linestyle='--', markersize=6)
        ax.fill(angles, ideal_values, alpha=0.1, color='#2Ecc71')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories_radar[:-1], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        ax.set_title('D. Multi-Dimensional Performance Assessment', fontsize=12, fontweight='bold', pad=10)

    def _plot_efficiency_tradeoff(self, ax, results: Dict):
        """Plot efficiency vs security trade-off"""
        # Comparison points
        models = {
            'GUARD': {'recall': results['recall'], 'time': results['avg_inference_time'], 'color': '#E74C3C'},
            'Baseline': {'recall': 0.6031, 'time': 0.6612, 'color': '#95A5A6'},
            'Precision-Opt': {'recall': 0.6446, 'time': 0.6209, 'color': '#3498DB'},
            'Ideal': {'recall': 0.95, 'time': 0.5, 'color': '#2Ecc71'}
        }
        
        for name, data in models.items():
            ax.scatter(data['time'], data['recall'], s=150, c=data['color'], 
                    label=name, alpha=0.8, edgecolors='black', linewidth=1)
            ax.annotate(name, (data['time'], data['recall']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Inference Time (s)', fontweight='bold')
        ax.set_ylabel('Recall', fontweight='bold')
        ax.set_title('E. Efficiency vs Security Trade-off', fontsize=12, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.4, 0.8)
        ax.set_ylim(0.5, 1.0)
        
        # Performance regions
        ax.axhspan(0.9, 1.0, alpha=0.1, color='green', label='Excellent')
        ax.axhspan(0.8, 0.9, alpha=0.1, color='yellow', label='Good')
        ax.axhspan(0.5, 0.8, alpha=0.1, color='red', label='Poor')

    def _plot_performance_gap(self, ax, results: Dict):
        """Plot performance gap analysis"""
        metrics = ['Recall', 'Precision', 'F1-Score']
        guard_values = [results['recall'], results['precision'], results['f1_score']]
        ideal_values = [0.95, 0.85, 0.85]
        gaps = [ideal - actual for ideal, actual in zip(ideal_values, guard_values)]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, guard_values, width, label='GUARD Achieved', 
                    color='#3498DB', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, ideal_values, width, label='Ideal Target', 
                    color='#2Ecc71', alpha=0.6, edgecolor='black')
        
        # Add gap annotations
        for i, (actual, ideal, gap) in enumerate(zip(guard_values, ideal_values, gaps)):
            ax.text(i, (actual + ideal)/2, f'Gap: {gap:.3f}', 
                ha='center', va='center', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Performance Metrics')
        ax.set_ylabel('Score')
        ax.set_title('F. Performance Gap Analysis', fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)

    def evaluate_on_dataset(self, df: pd.DataFrame, dataset_name="main") -> Dict:
        """
        Comprehensive evaluation on dataset with error analysis
        """
        logger.info(f"Evaluating SECURITY-GUARD on {dataset_name} dataset...")
        
        predictions = []
        execution_times = []
        detailed_results = []
        
        for idx, row in df.iterrows():
            resume_text = row['resume']
            true_label = row['label']
            
            prediction, exec_time = self.process_resume_for_injection(resume_text)
            predictions.append(prediction)
            execution_times.append(exec_time)
            
            detailed_results.append({
                'index': idx,
                'true_label': true_label,
                'predicted_label': prediction,
                'inference_time': exec_time,
                'correct': prediction == true_label
            })
        
        # Calculate metrics
        precision = precision_score(df['label'], predictions, zero_division=0)
        recall = recall_score(df['label'], predictions, zero_division=0)
        f1 = f1_score(df['label'], predictions, zero_division=0)
        cm = confusion_matrix(df['label'], predictions)
        
        # Performance statistics
        avg_time = np.mean(execution_times)
        throughput = len(df) / sum(execution_times)
        
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
            'total_time': sum(execution_times),
            'total_samples': len(df),
            'detailed_predictions': detailed_results
        }
        
        # ADD ERROR ANALYSIS HERE
        error_analyzer = ErrorAnalyzer(self)
        error_analysis = error_analyzer.analyze_errors(df, predictions, dataset_name)
        results['error_analysis'] = error_analysis
        
        # Save results
        self._save_results(results, dataset_name)
        self._create_visualizations(results, df['label'], predictions, dataset_name)
        self.create_unified_analysis_figure(results, dataset_name)
        
        # Generate error analysis report
        error_analyzer.create_comprehensive_error_report(error_analysis, dataset_name)
        
        return results

    def _save_results(self, results: Dict, dataset_name: str):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.results_dir}/security_guard_results_{dataset_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Security results saved to {filename}")

    def _create_visualizations(self, results: Dict, true_labels, predictions, dataset_name: str):
        """Create visualization plots for results"""
        # Confusion Matrix Heatmap
        plt.figure(figsize=(8, 6))
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', 
                   xticklabels=['Predicted Clean', 'Predicted Malicious'],
                   yticklabels=['Actual Clean', 'Actual Malicious'])
        plt.title(f'SECURITY-GUARD Confusion Matrix - {dataset_name}\n(Recall: {results["recall"]:.1%})')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/security_confusion_matrix_{dataset_name}.png", dpi=300)
        plt.close()
        
        # Security-focused Metrics
        metrics = ['Recall', 'Precision', 'F1-Score']
        values = [results['recall'], results['precision'], results['f1_score']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red for recall (security focus)
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(metrics, values, color=colors, alpha=0.8)
        plt.ylim(0, 1.0)
        plt.title(f'SECURITY-GUARD Performance - {dataset_name}\nModel: {self.model_name.split("/")[-1]}')
        plt.ylabel('Score')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/security_performance_{dataset_name}.png", dpi=300)
        plt.close()

    def create_comprehensive_visualizations(self, results: Dict, dataset_name: str):
        """Create comprehensive publication-ready visualizations for main results"""
        
        # 1. Enhanced Performance Metrics Radar Chart
        self._create_radar_chart(results, dataset_name)
        
        # 2. Security Impact Donut Chart
        self._create_security_donut_chart(results, dataset_name)
        
        # 3. Comparative Performance Bar Chart
        self._create_comparative_barchart(results, dataset_name)
        
        # 4. Efficiency vs Performance Scatter Plot
        self._create_efficiency_scatter(results, dataset_name)

    def _create_radar_chart(self, results: Dict, dataset_name: str):
        """Create radar chart for multi-dimensional performance assessment"""
        categories = ['Recall', 'Precision', 'F1-Score', 'Throughput', 'Efficiency']
        
        # Normalize values for radar chart (0-1 scale)
        recall = results['recall']
        precision = results['precision'] 
        f1 = results['f1_score']
        throughput = results['throughput'] / 2.0  # Normalize assuming max 2 resumes/sec
        efficiency = 1 - (results['avg_inference_time'] / 2.0)  # Normalize assuming max 2s
        
        values = [recall, precision, f1, throughput, efficiency]
        
        # Complete the circle
        values += values[:1]
        categories += categories[:1]
        
        angles = np.linspace(0, 2*np.pi, len(categories)-1, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label='GUARD Performance', color='#FF6B6B')
        ax.fill(angles, values, alpha=0.25, color='#FF6B6B')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories[:-1])
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        plt.title('GUARD Multi-Dimensional Performance Assessment', size=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/radar_performance_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_security_donut_chart(self, results: Dict, dataset_name: str):
        """Create donut chart showing security impact distribution"""
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        
        labels = ['Safe Passed\n(True Negative)', 'Safe Flagged\n(False Positive)', 
                'Attack Missed\n(False Negative)', 'Attack Blocked\n(True Positive)']
        sizes = [tn, fp, fn, tp]
        colors = ['#2Ecc71', '#F39C12', '#E74C3C', '#3498DB']
        explode = (0, 0.1, 0.1, 0)  # Emphasize errors
        
        fig, ax = plt.subplots(figsize=(10, 8))
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90,
                                        textprops={'fontsize': 10})
        
        # Draw circle for donut chart
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig.gca().add_artist(centre_circle)
        
        # Equal aspect ratio ensures pie is drawn as circle
        ax.axis('equal')  
        plt.title('Security Impact Distribution: Resume Processing Outcomes', 
                fontsize=14, fontweight='bold', pad=20)
        
        # Enhance text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/security_donut_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_comparative_barchart(self, results: Dict, dataset_name: str):
        """Create comparative bar chart showing performance vs ideal targets"""
        metrics = ['Recall', 'Precision', 'F1-Score']
        guard_values = [results['recall'], results['precision'], results['f1_score']]
        ideal_values = [0.95, 0.90, 0.85]  # Ideal targets for security systems
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, guard_values, width, label='GUARD Performance', 
                    color='#3498DB', alpha=0.8)
        bars2 = ax.bar(x + width/2, ideal_values, width, label='Ideal Security Targets', 
                    color='#2Ecc71', alpha=0.6)
        
        ax.set_xlabel('Performance Metrics')
        ax.set_ylabel('Score')
        ax.set_title('GUARD Performance vs Ideal Security Targets', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.0)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/comparative_performance_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_efficiency_scatter(self, results: Dict, dataset_name: str):
        """Create scatter plot showing efficiency vs performance trade-off"""
        # Simulate some comparison points (you can replace with actual baseline data)
        models_data = {
            'GUARD (Current)': {'recall': results['recall'], 'time': results['avg_inference_time']},
            'Baseline (mpnet)': {'recall': 0.6031, 'time': 0.6612},
            'High-Precision': {'recall': 0.6446, 'time': 0.6209},
            'Ideal Target': {'recall': 0.95, 'time': 0.5}
        }
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'GUARD (Current)': '#E74C3C', 'Baseline (mpnet)': '#95A5A6', 
                'High-Precision': '#3498DB', 'Ideal Target': '#2Ecc71'}
        markers = {'GUARD (Current)': 'o', 'Baseline (mpnet)': 's', 
                'High-Precision': '^', 'Ideal Target': '*'}
        
        for model, data in models_data.items():
            ax.scatter(data['time'], data['recall'], 
                    c=colors[model], marker=markers[model], s=200, 
                    label=model, alpha=0.8, edgecolors='black', linewidth=1)
            
            # Add annotations
            ax.annotate(model, (data['time'], data['recall']),
                    xytext=(10, 10), textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Average Inference Time (seconds)')
        ax.set_ylabel('Recall (Attack Detection Rate)')
        ax.set_title('Efficiency vs Security Performance Trade-off', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.4, 0.8)
        ax.set_ylim(0.5, 1.0)
        
        # Add performance regions
        ax.axhspan(0.9, 1.0, alpha=0.1, color='green', label='Excellent Security')
        ax.axhspan(0.8, 0.9, alpha=0.1, color='yellow', label='Good Security')
        ax.axhspan(0.5, 0.8, alpha=0.1, color='red', label='Poor Security')
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/efficiency_tradeoff_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    def generate_security_report(self, results: Dict):
        """Generate a security-focused performance report"""
        print("\n" + "="*80)
        print("🛡️  SECURITY-GUARD COMPREHENSIVE PERFORMANCE REPORT")
        print("="*80)
        
        print(f"\n📊 DATASET: {results['dataset']}")
        print(f"🤖 SECURITY MODEL: {results['model']}")
        print(f"⚡ DETECTION THRESHOLD: {results['threshold']}")
        
        print(f"\n🔒 SECURITY METRICS (Primary Focus):")
        print(f"   ✅ Recall:           {results['recall']:.4f}  ← ATTACK DETECTION RATE")
        print(f"   ❌ False Negative Rate: {1 - results['recall']:.4f}  ← MISSED ATTACKS")
        print(f"   🎯 F1-Score:        {results['f1_score']:.4f}")
        print(f"   📈 Precision:       {results['precision']:.4f}")
        
        print(f"\n🚀 EFFICIENCY METRICS:")
        print(f"   Avg Inference Time: {results['avg_inference_time']:.4f}s")
        print(f"   Throughput:        {results['throughput']:.2f} resumes/sec")
        
        cm = results['confusion_matrix']
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        print(f"\n📈 SECURITY ANALYSIS:")
        print(f"   ✅ True Positives:  {tp} attacks correctly detected")
        print(f"   ❌ False Negatives: {fn} attacks MISSED ({(fn/(tp+fn))*100:.1f}% of attacks)")
        print(f"   ⚠️  False Positives: {fp} clean resumes flagged ({(fp/(tn+fp))*100:.1f}% of clean)")
        print(f"   ✅ True Negatives:  {tn} clean resumes correctly passed")
        
        print(f"\n🎯 SECURITY ASSESSMENT:")
        detection_rate = results['recall'] * 100
        if detection_rate >= 90:
            print(f"   🟢 EXCELLENT: {detection_rate:.1f}% of attacks detected")
        elif detection_rate >= 80:
            print(f"   🟡 GOOD: {detection_rate:.1f}% of attacks detected")
        else:
            print(f"   🔴 POOR: Only {detection_rate:.1f}% of attacks detected")

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
    
    # Generate security-focused report
    guard.generate_security_report(results)
    
    return results

if __name__ == "__main__":
    # Run the security-focused evaluation
    security_results = run_security_evaluation()