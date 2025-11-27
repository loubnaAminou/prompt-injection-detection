import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import re

# Set a non-GTK backend to avoid segmentation faults
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Create directory for figures
os.makedirs("paper_figures", exist_ok=True)

def safe_parse_confusion_matrix(cm_str):
    """Safely parse confusion matrix string from CSV"""
    try:
        # Try direct evaluation first
        return ast.literal_eval(cm_str)
    except:
        try:
            # If that fails, use regex to extract numbers
            numbers = re.findall(r'\d+', cm_str)
            if len(numbers) == 4:
                return [[int(numbers[0]), int(numbers[1])], [int(numbers[2]), int(numbers[3])]]
            else:
                print(f"Warning: Could not parse confusion matrix: {cm_str}")
                return [[0, 0], [0, 0]]
        except:
            print(f"Warning: Failed to parse confusion matrix: {cm_str}")
            return [[0, 0], [0, 0]]

def create_simple_visualizations(results_df, output_dir="paper_figures"):
    """Create robust, publication-ready visualizations without interactive backend"""
    
    # Set style for academic papers
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 14
    })
    
    # Clean model names
    results_df['model_short'] = results_df['model_name'].str.replace('sentence-transformers/', '')
    short_names = [name.split('/')[-1] for name in results_df['model_short']]
    
    # Create individual plots instead of subplots to avoid complexity
    create_performance_plot(results_df, short_names, output_dir)
    create_tradeoff_plot(results_df, short_names, output_dir)
    create_throughput_plot(results_df, short_names, output_dir)
    create_precision_recall_plot(results_df, short_names, output_dir)

def create_performance_plot(results_df, short_names, output_dir):
    """Create main performance comparison bar chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(results_df))
    width = 0.25
    
    # Plot bars for each metric
    metrics = ['f1_score', 'precision', 'recall']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    labels = ['F1-Score', 'Precision', 'Recall']
    
    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        values = results_df[metric].values
        bars = ax.bar(x + i*width, values, width, label=label, color=color, alpha=0.8)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('Embedding Models')
    ax.set_ylabel('Performance Score')
    ax.set_title('Detection Performance Across Embedding Models')
    ax.set_xticks(x + width)
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/performance_comparison.pdf', bbox_inches='tight')
    plt.close()

def create_tradeoff_plot(results_df, short_names, output_dir):
    """Create speed vs accuracy trade-off plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_df)))
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.scatter(row['avg_inference_time'], row['f1_score'], 
                  c=[colors[i]], s=150, alpha=0.7, edgecolors='black', linewidth=1)
        ax.annotate(short_names[i], 
                   (row['avg_inference_time'], row['f1_score']),
                   xytext=(10, 10), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Average Inference Time (seconds)')
    ax.set_ylabel('F1-Score')
    ax.set_title('Performance vs Efficiency Trade-off')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speed_accuracy_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/speed_accuracy_tradeoff.pdf', bbox_inches='tight')
    plt.close()

def create_throughput_plot(results_df, short_names, output_dir):
    """Create throughput comparison plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(results_df)))
    bars = ax.bar(short_names, results_df['throughput'], color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar, throughput in zip(bars, results_df['throughput']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{throughput:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_xticklabels(short_names, rotation=45, ha='right')
    ax.set_ylabel('Throughput (resumes/second)')
    ax.set_title('Processing Throughput Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/throughput_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/throughput_comparison.pdf', bbox_inches='tight')
    plt.close()

def create_precision_recall_plot(results_df, short_names, output_dir):
    """Create precision-recall trade-off plot"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    colors = plt.cm.plasma(np.linspace(0, 1, len(results_df)))
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        ax.scatter(row['recall'], row['precision'], 
                  c=[colors[i]], s=150, alpha=0.7, edgecolors='black', linewidth=1)
        ax.annotate(short_names[i], 
                   (row['recall'], row['precision']),
                   xytext=(10, 10), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Trade-off')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 1.0)
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_tradeoff.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/precision_recall_tradeoff.pdf', bbox_inches='tight')
    plt.close()

def create_confusion_matrix_heatmaps(results_df, output_dir="paper_figures"):
    """Create confusion matrix heatmaps for each model"""
    n_models = len(results_df)
    fig, axes = plt.subplots(1, n_models, figsize=(4*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (_, row) in enumerate(results_df.iterrows()):
        model_name = row['model_short'].split('/')[-1]
        cm = safe_parse_confusion_matrix(row['confusion_matrix'])
        
        # Create heatmap
        im = axes[idx].imshow(cm, cmap='Blues', aspect='auto', vmin=0, vmax=800)
        axes[idx].set_title(f'{model_name}', fontsize=10)
        axes[idx].set_xticks([0, 1])
        axes[idx].set_yticks([0, 1])
        axes[idx].set_xticklabels(['Pred 0', 'Pred 1'])
        axes[idx].set_yticklabels(['True 0', 'True 1'])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[idx].text(j, i, f'{cm[i][j]}', 
                             ha='center', va='center', 
                             fontsize=12, fontweight='bold',
                             color='white' if cm[i][j] > 400 else 'black')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/confusion_matrices.pdf', bbox_inches='tight')
    plt.close()

def create_comprehensive_analysis(results_df):
    """Generate comprehensive analysis without visualization dependencies"""
    
    print("GUARD ABLATION STUDY: COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    # Basic results table
    print("\n📊 PERFORMANCE RESULTS SUMMARY")
    print("-" * 50)
    
    table_data = []
    for _, row in results_df.iterrows():
        model_name = row['model_name'].replace('sentence-transformers/', '').split('/')[-1]
        table_data.append({
            'Model': model_name,
            'F1': f"{row['f1_score']:.4f}",
            'Precision': f"{row['precision']:.4f}", 
            'Recall': f"{row['recall']:.4f}",
            'Time (s)': f"{row['avg_inference_time']:.4f}",
            'Throughput': f"{row['throughput']:.2f}/s"
        })
    
    results_table = pd.DataFrame(table_data)
    print(results_table.to_string(index=False))
    
    # Key findings
    print("\n🔍 KEY FINDINGS")
    print("-" * 30)
    
    best_f1_model = results_df.loc[results_df['f1_score'].idxmax()]
    best_precision_model = results_df.loc[results_df['precision'].idxmax()]
    best_recall_model = results_df.loc[results_df['recall'].idxmax()]
    fastest_model = results_df.loc[results_df['avg_inference_time'].idxmin()]
    
    print(f"• Best F1-Score: {best_f1_model['model_short'].split('/')[-1]} ({best_f1_model['f1_score']:.4f})")
    print(f"• Highest Precision: {best_precision_model['model_short'].split('/')[-1]} ({best_precision_model['precision']:.4f})")
    print(f"• Highest Recall: {best_recall_model['model_short'].split('/')[-1]} ({best_recall_model['recall']:.4f})")
    print(f"• Fastest Inference: {fastest_model['model_short'].split('/')[-1]} ({fastest_model['avg_inference_time']:.4f}s)")
    
    # Statistical improvements
    baseline = results_df[results_df['model_name'].str.contains('all-mpnet-base-v2')].iloc[0]
    print(f"\n• Baseline (all-mpnet-base-v2) F1: {baseline['f1_score']:.4f}")
    print(f"• Best improvement: +{(best_f1_model['f1_score'] - baseline['f1_score'])/baseline['f1_score']*100:.1f}%")
    
    # Confusion matrix analysis
    print("\n📈 CONFUSION MATRIX ANALYSIS")
    print("-" * 35)
    
    for _, row in results_df.iterrows():
        model_name = row['model_short'].split('/')[-1]
        cm = safe_parse_confusion_matrix(row['confusion_matrix'])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total_samples = tn + fp + fn + tp
        
        print(f"\n{model_name}:")
        print(f"  True Negatives: {tn} (clean correctly classified)")
        print(f"  False Positives: {fp} ({fp/(tn+fp)*100:.1f}% of clean resumes)")
        print(f"  False Negatives: {fn} ({fn/(fn+tp)*100:.1f}% of malicious resumes)") 
        print(f"  True Positives: {tp} (attacks correctly detected)")
        
        # Calculate additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        print(f"  Specificity: {specificity:.3f}")
        print(f"  False Negative Rate: {fnr:.3f}")
        print(f"  False Positive Rate: {fpr:.3f}")
    
    # Error pattern analysis
    print("\n🔎 ERROR PATTERN ANALYSIS")
    print("-" * 30)
    
    # Find model with most false positives
    max_fp_model = None
    max_fp = -1
    max_fn_model = None  
    max_fn = -1
    
    for _, row in results_df.iterrows():
        cm = safe_parse_confusion_matrix(row['confusion_matrix'])
        fp = cm[0][1]
        fn = cm[1][0]
        
        if fp > max_fp:
            max_fp = fp
            max_fp_model = row['model_short'].split('/')[-1]
        if fn > max_fn:
            max_fn = fn
            max_fn_model = row['model_short'].split('/')[-1]
    
    print(f"• Most False Positives: {max_fp_model} ({max_fp} clean resumes misclassified)")
    print(f"• Most False Negatives: {max_fn_model} ({max_fn} attacks missed)")
    
    # Deployment recommendations
    print("\n🎯 DEPLOYMENT RECOMMENDATIONS")
    print("-" * 35)
    
    recommendations = [
        ("High-Security", "all-MiniLM-L6-v2", "94% precision minimizes false alarms"),
        ("Balanced", "multi-qa-mpnet-base-dot-v1", "Best overall F1 (77.0%)"),
        ("Fast", "all-MiniLM-L6-v2", "Fastest + high precision"),
        ("Catch-All", "paraphrase-multilingual", "93% recall but many false positives")
    ]
    
    for use_case, model, rationale in recommendations:
        print(f"• {use_case}: {model} - {rationale}")
    
    # Save detailed analysis
    with open("paper_figures/ablation_analysis.txt", "w") as f:
        f.write("GUARD Ablation Study Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(results_table.to_string(index=False))
        f.write("\n\nKey Insights:\n")
        f.write(f"- Best F1: {best_f1_model['model_short'].split('/')[-1]} ({best_f1_model['f1_score']:.4f})\n")
        f.write(f"- Best Precision: {best_precision_model['model_short'].split('/')[-1]} ({best_precision_model['precision']:.4f})\n")
        f.write(f"- Best Recall: {best_recall_model['model_short'].split('/')[-1]} ({best_recall_model['recall']:.4f})\n")
        f.write(f"- Fastest: {fastest_model['model_short'].split('/')[-1]} ({fastest_model['avg_inference_time']:.4f}s)\n")

# Load and process your data
try:
    results_df = pd.read_csv("ablation_results/ablation_study_results.csv")
    
    # Generate visualizations
    create_simple_visualizations(results_df)
    create_confusion_matrix_heatmaps(results_df)
    
    # Generate comprehensive analysis
    create_comprehensive_analysis(results_df)
    
    print(f"\n✅ Analysis complete! Check the 'paper_figures' directory for:")
    print("   - performance_comparison.png/pdf")
    print("   - speed_accuracy_tradeoff.png/pdf") 
    print("   - throughput_comparison.png/pdf")
    print("   - precision_recall_tradeoff.png/pdf")
    print("   - confusion_matrices.png/pdf")
    print("   - ablation_analysis.txt")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure your CSV file is in the correct location.")
    import traceback
    traceback.print_exc()