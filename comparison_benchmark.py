import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from predict_results import benchmark_guard_pipeline
from fine_tuning import BenchmarkClassifier
from sentence_transformers import SentenceTransformer
import logging
import os

# Configure logging to be less verbose
logging.basicConfig(level=logging.WARNING)  # Reduced from INFO to WARNING
logger = logging.getLogger(__name__)

class CleanComparisonVisualizer:
    """Create clean, evidence-based visualizations"""
    
    def __init__(self, results_dir="comparison_results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid')
        self.colors = ['#E74C3C', '#3498DB', '#2Ecc71']  # GUARD, Classifier, WithSecure
    
    def create_evidence_based_figures(self, guard_results, classifier_results, withsecure_data):
        """Create figures based only on empirical evidence"""
        
        # Figure 1: Core Performance Metrics
        self._create_performance_comparison(guard_results, classifier_results, withsecure_data)
        
        # Figure 2: Precision-Recall Trade-off
        self._create_pr_tradeoff(guard_results, classifier_results, withsecure_data)
        
        # Figure 3: Training Progress (WithSecure only)
        self._create_training_progress(withsecure_data)
    
    def _create_performance_comparison(self, guard_results, classifier_results, withsecure_data):
        """Figure 1: Core performance metrics comparison"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        
        methods = ['GUARD', 'Classifier', 'WithSecure']
        best_withsecure = withsecure_data.loc[withsecure_data['f1'].idxmax()]
        
        # F1-Scores
        f1_scores = [guard_results['f1_score'], classifier_results['f1_score'], best_withsecure['f1']]
        bars1 = ax1.bar(methods, f1_scores, color=self.colors, alpha=0.8, edgecolor='black')
        ax1.set_ylabel('F1-Score', fontweight='bold', fontsize=12)
        ax1.set_title('F1-Score Comparison', fontweight='bold', fontsize=12)
        ax1.set_ylim(0, 1.0)
        self._add_value_labels(ax1, bars1)
        
        # Precision
        precision_scores = [guard_results['precision'], classifier_results['precision'], best_withsecure['precision']]
        bars2 = ax2.bar(methods, precision_scores, color=self.colors, alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Precision', fontweight='bold', fontsize=12)
        ax2.set_title('Precision Comparison', fontweight='bold', fontsize=12)
        ax2.set_ylim(0, 1.0)
        self._add_value_labels(ax2, bars2)
        
        # Recall
        recall_scores = [guard_results['recall'], classifier_results['recall'], best_withsecure['recall']]
        bars3 = ax3.bar(methods, recall_scores, color=self.colors, alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Recall', fontweight='bold', fontsize=12)
        ax3.set_title('Recall Comparison', fontweight='bold', fontsize=12)
        ax3.set_ylim(0, 1.0)
        self._add_value_labels(ax3, bars3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/performance_comparison.pdf", bbox_inches='tight')
        plt.close()
        
        logger.info("Created performance comparison figure")
    
    def _create_pr_tradeoff(self, guard_results, classifier_results, withsecure_data):
        """Figure 2: Precision-Recall trade-off"""
        plt.figure(figsize=(8, 6))
        
        # GUARD and Classifier
        plt.scatter(guard_results['precision'], guard_results['recall'], 
                   s=200, c=self.colors[0], label='GUARD', alpha=0.8, 
                   edgecolors='black', linewidth=2, zorder=5)
        plt.scatter(classifier_results['precision'], classifier_results['recall'], 
                   s=200, c=self.colors[1], label='Classifier', alpha=0.8,
                   edgecolors='black', linewidth=2, zorder=5)
        
        # WithSecure epochs
        plt.scatter(withsecure_data['precision'], withsecure_data['recall'], 
                   s=80, c=self.colors[2], alpha=0.6, label='WithSecure (All Epochs)')
        
        # Best WithSecure
        best_withsecure = withsecure_data.loc[withsecure_data['f1'].idxmax()]
        plt.scatter(best_withsecure['precision'], best_withsecure['recall'], 
                   s=200, c=self.colors[2], marker='*', label='WithSecure (Best)',
                   edgecolors='black', linewidth=2, zorder=6)
        
        # Ideal point
        plt.scatter(1.0, 1.0, s=150, c='gold', marker='*', label='Ideal',
                   edgecolors='black', linewidth=2, zorder=7)
        
        plt.xlabel('Precision', fontweight='bold', fontsize=12)
        plt.ylabel('Recall', fontweight='bold', fontsize=12)
        plt.title('Precision-Recall Trade-off Analysis', fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0.3, 1.05)
        plt.ylim(0.3, 1.05)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/pr_tradeoff.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/pr_tradeoff.pdf", bbox_inches='tight')
        plt.close()
        
        logger.info("Created Precision-Recall trade-off figure")
    
    def _create_training_progress(self, withsecure_data):
        """Figure 3: WithSecure training progress"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(withsecure_data['epoch'], withsecure_data['precision'], 
                'o-', color='#3498DB', linewidth=2, label='Precision', markersize=6)
        plt.plot(withsecure_data['epoch'], withsecure_data['recall'], 
                's-', color='#E74C3C', linewidth=2, label='Recall', markersize=6)
        plt.plot(withsecure_data['epoch'], withsecure_data['f1'], 
                '^-', color='#2Ecc71', linewidth=2, label='F1-Score', markersize=6)
        
        # Highlight best epoch
        best_epoch = withsecure_data.loc[withsecure_data['f1'].idxmax()]
        plt.axvline(x=best_epoch['epoch'], color='red', linestyle='--', alpha=0.7,
                   label=f'Best Epoch ({best_epoch["epoch"]})')
        
        plt.xlabel('Training Epoch', fontweight='bold', fontsize=12)
        plt.ylabel('Score', fontweight='bold', fontsize=12)
        plt.title('WithSecure: Training Progress Across Epochs', fontweight='bold', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.6, 1.0)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/withsecure_training.png", dpi=300, bbox_inches='tight')
        plt.savefig(f"{self.results_dir}/withsecure_training.pdf", bbox_inches='tight')
        plt.close()
        
        logger.info("Created WithSecure training progress figure")
    
    def _add_value_labels(self, ax, bars):
        """Add value labels on bars"""
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

def run_clean_comparison():
    """Run comparison with clean output"""
    
    # Load data
    df = pd.read_csv("dataset/injected_resume_dataset.csv")
    df_0 = df[df["label"] == 0]
    df_1 = df[df["label"] == 1] 
    df_balanced = pd.concat([df_0, df_1]).reset_index(drop=True)
    
    logger.warning("Starting GUARD evaluation...")
    model = SentenceTransformer("sentence-transformers/multi-qa-mpnet-base-dot-v1")
    prompt_embeddings = np.load('dataset/prompts_embeddings.npz')['embeddings']
    guard_results = benchmark_guard_pipeline(df_balanced, model, prompt_embeddings, threshold=0.65)
    
    logger.warning("Starting Classifier evaluation...")
    classifier = BenchmarkClassifier(model_path="prompt_injection_classifier.pt")
    classifier_results = classifier.benchmark_classifier(df_balanced, threshold=0.5)
    
    # WithSecure data from blogpost
    withsecure_data = pd.DataFrame({
        'epoch': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'accuracy': [0.8875, 0.7625, 0.8750, 0.8125, 0.8375, 0.8125, 0.8375, 0.8250, 0.8375, 0.8250],
        'precision': [0.969697, 0.723404, 0.894737, 0.878788, 0.935484, 0.837838, 0.909091, 0.882353, 0.885714, 0.861111],
        'recall': [0.800, 0.850, 0.850, 0.725, 0.725, 0.775, 0.750, 0.750, 0.775, 0.775],
        'f1': [0.876712, 0.781609, 0.871795, 0.794521, 0.816901, 0.805195, 0.821918, 0.810811, 0.826667, 0.815789]
    })
    
    best_withsecure = withsecure_data.loc[withsecure_data['f1'].idxmax()]
    
    # Create comprehensive results table
    results_table = {
        'Method': ['GUARD (Ours)', 'Supervised Classifier', 'WithSecure (Best)'],
        'F1-Score': [guard_results['f1_score'], classifier_results['f1_score'], best_withsecure['f1']],
        'Precision': [guard_results['precision'], classifier_results['precision'], best_withsecure['precision']],
        'Recall': [guard_results['recall'], classifier_results['recall'], best_withsecure['recall']],
        'Accuracy': [guard_results.get('accuracy', 'N/A'), classifier_results.get('accuracy', 'N/A'), best_withsecure['accuracy']],
        'Throughput (resumes/sec)': [guard_results['throughput'], classifier_results['throughput'], 'Not Reported'],
        'Avg Inference Time (s)': [guard_results['avg_inference_time'], classifier_results['avg_inference_time'], 'Not Reported'],
        'Approach': ['Zero-shot Similarity', 'Supervised Fine-tuning', 'Supervised Fine-tuning'],
        'Data Requirements': ['Low (Templates)', 'High (Labeled Data)', 'High (Labeled Data)'],
        'Best Configuration': ['Threshold: 0.65', 'Threshold: 0.50', f'Epoch: {int(best_withsecure["epoch"])}']
    }
    
    results_df = pd.DataFrame(results_table)
    
    # Generate clean visualizations
    viz = CleanComparisonVisualizer()
    viz.create_evidence_based_figures(guard_results, classifier_results, withsecure_data)
    
    return guard_results, classifier_results, withsecure_data, results_df

if __name__ == "__main__":
    print("🚀 Starting comprehensive comparison...")
    print("This may take a few minutes. Terminal output will be minimal.")
    print("-" * 50)
    
    guard_results, classifier_results, withsecure_data, results_df = run_clean_comparison()
    
    # Save results
    results_df.to_csv("clean_comparison_results.csv", index=False)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    # Display clean results table
    print("\nRESULTS SUMMARY:")
    print("-" * 60)
    display_df = results_df[['Method', 'F1-Score', 'Precision', 'Recall', 'Throughput (resumes/sec)']].copy()
    display_df['F1-Score'] = display_df['F1-Score'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    display_df['Precision'] = display_df['Precision'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    display_df['Recall'] = display_df['Recall'].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    print(display_df.to_string(index=False))
    
    print(f"\n📊 Results saved to: clean_comparison_results.csv")
    print("📈 Visualizations saved to: comparison_results/")
    print(f"🎯 Best WithSecure: Epoch {withsecure_data.loc[withsecure_data['f1'].idxmax()]['epoch']} "
          f"(F1: {withsecure_data['f1'].max():.3f})")