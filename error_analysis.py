#!/usr/bin/env python3
"""
Lightweight Error Analysis for GUARD Pipeline
Run this separately to analyze errors without re-running detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import json
import os

class GuardErrorAnalyzer:
    """Lightweight error analyzer for GUARD pipeline results"""
    
    def __init__(self, results_dir="security_guard_results"):
        self.results_dir = results_dir
        
    def load_latest_results(self):
        """Load the most recent results file"""
        result_files = [f for f in os.listdir(self.results_dir) if f.startswith('security_guard_results') and f.endswith('.json')]
        if not result_files:
            raise FileNotFoundError("No results files found. Run GUARD evaluation first.")
        
        # Get the most recent file
        latest_file = sorted(result_files)[-1]
        file_path = os.path.join(self.results_dir, latest_file)
        
        with open(file_path, 'r') as f:
            results = json.load(f)
        
        print(f"📁 Loaded results from: {latest_file}")
        return results
    
    def load_dataset_with_predictions(self, results):
        """Load dataset and match with predictions"""
        df = pd.read_csv("dataset/injected_resume_dataset.csv")
        df_0 = df[df["label"] == 0]
        df_1 = df[df["label"] == 1]
        df_balanced = pd.concat([df_0, df_1]).reset_index(drop=True)
        
        # Add predictions from results
        detailed_predictions = results['detailed_predictions']
        predictions = [pred['predicted_label'] for pred in detailed_predictions]
        df_balanced['predicted'] = predictions
        
        return df_balanced
    
    def perform_comprehensive_error_analysis(self, df, results):
        """Main error analysis function"""
        print("\n" + "="*60)
        print("🔍 GUARD ERROR ANALYSIS")
        print("="*60)
        
        # Basic error statistics
        self._analyze_error_statistics(df, results)
        
        # Detailed error analysis
        false_positives, false_negatives = self._categorize_errors(df)
        
        # Pattern analysis
        self._analyze_false_positive_patterns(false_positives)
        self._analyze_false_negative_patterns(false_negatives)
        
        # Create visualizations
        self._create_error_visualizations(df, results, false_positives, false_negatives)
        
        return false_positives, false_negatives
    
    def _analyze_error_statistics(self, df, results):
        """Calculate basic error statistics"""
        cm = np.array(results['confusion_matrix'])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        total = tn + fp + fn + tp
        
        print(f"\n📊 ERROR STATISTICS:")
        print(f"   Total Samples: {total}")
        print(f"   False Positives: {fp} ({fp/total*100:.1f}%)")
        print(f"   False Negatives: {fn} ({fn/total*100:.1f}%)")
        print(f"   True Positives: {tp} ({tp/total*100:.1f}%)")
        print(f"   True Negatives: {tn} ({tn/total*100:.1f}%)")
        
        # Error rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        print(f"\n📈 ERROR RATES:")
        print(f"   False Positive Rate: {fpr:.3f}")
        print(f"   False Negative Rate: {fnr:.3f}")
        print(f"   Precision: {results['precision']:.3f}")
        print(f"   Recall: {results['recall']:.3f}")
    
    def _categorize_errors(self, df):
        """Categorize errors into false positives and false negatives"""
        false_positives = df[(df['predicted'] == 1) & (df['label'] == 0)]
        false_negatives = df[(df['predicted'] == 0) & (df['label'] == 1)]
        
        print(f"\n🎯 ERROR CATEGORIZATION:")
        print(f"   False Positives: {len(false_positives)} samples")
        print(f"   False Negatives: {len(false_negatives)} samples")
        
        return false_positives, false_negatives
    
    def _analyze_false_positive_patterns(self, false_positives, sample_size=10):
        """Analyze patterns in false positives"""
        print(f"\n📈 FALSE POSITIVE ANALYSIS:")
        print(f"   Analyzing {min(len(false_positives), sample_size)} samples...")
        
        technical_keywords = [
            'python', 'java', 'javascript', 'sql', 'api', 'framework', 'system',
            'database', 'server', 'cloud', 'docker', 'kubernetes', 'aws', 'azure',
            'algorithm', 'optimize', 'implement', 'deploy', 'configure', 'debug'
        ]
        
        fp_patterns = {}
        
        for idx, row in false_positives.head(sample_size).iterrows():
            resume_text = row['resume'].lower()
            
            # Count technical keywords
            found_keywords = [kw for kw in technical_keywords if kw in resume_text]
            
            if found_keywords:
                for keyword in found_keywords:
                    fp_patterns[keyword] = fp_patterns.get(keyword, 0) + 1
            
            # Sample analysis for first few FPs
            if idx < 3:
                print(f"\n   FP Sample {idx}:")
                print(f"      Length: {len(resume_text)} chars")
                print(f"      Technical terms: {found_keywords[:5]}")
                # Show a snippet of problematic text
                snippet = resume_text[:200] + "..." if len(resume_text) > 200 else resume_text
                print(f"      Snippet: {snippet}")
        
        # Print most common FP patterns
        if fp_patterns:
            print(f"\n   Most common FP triggers:")
            for keyword, count in sorted(fp_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"      {keyword}: {count} occurrences")
    
    def _analyze_false_negative_patterns(self, false_negatives, sample_size=10):
        """Analyze patterns in false negatives"""
        print(f"\n📉 FALSE NEGATIVE ANALYSIS:")
        print(f"   Analyzing {min(len(false_negatives), sample_size)} samples...")
        
        injection_keywords = [
            'ignore', 'disregard', 'previous instructions', 'system prompt',
            'as an ai', 'as a language model', 'you are', 'your purpose',
            'do not', 'never', 'always', 'from now on', 'override'
        ]
        
        fn_patterns = {}
        
        for idx, row in false_negatives.head(sample_size).iterrows():
            resume_text = row['resume'].lower()
            
            # Check for injection keywords
            found_keywords = [kw for kw in injection_keywords if kw in resume_text]
            
            if found_keywords:
                for keyword in found_keywords:
                    fn_patterns[keyword] = fn_patterns.get(keyword, 0) + 1
            else:
                # No obvious keywords - might be obfuscated
                fn_patterns['obfuscated'] = fn_patterns.get('obfuscated', 0) + 1
            
            # Sample analysis for first few FNs
            if idx < 3:
                print(f"\n   FN Sample {idx}:")
                print(f"      Length: {len(resume_text)} chars")
                if found_keywords:
                    print(f"      Injection keywords: {found_keywords}")
                else:
                    print(f"      No obvious keywords - likely obfuscated")
                
                # Try to find the injection part
                injection_indicators = ['ignore', 'disregard', 'as an ai', 'system']
                for indicator in injection_indicators:
                    if indicator in resume_text:
                        start = max(0, resume_text.find(indicator) - 50)
                        end = min(len(resume_text), resume_text.find(indicator) + 100)
                        snippet = resume_text[start:end]
                        print(f"      Injection context: ...{snippet}...")
                        break
    
    def _create_error_visualizations(self, df, results, false_positives, false_negatives):
        """Create error analysis visualizations"""
        print(f"\n📊 Creating error visualizations...")
        
        # 1. Error Distribution Pie Chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Error type distribution
        cm = np.array(results['confusion_matrix'])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        
        error_labels = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
        error_sizes = [tn, fp, fn, tp]
        error_colors = ['#2Ecc71', '#F39C12', '#E74C3C', '#3498DB']
        
        ax1.pie(error_sizes, labels=error_labels, colors=error_colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('A. Error Type Distribution', fontweight='bold')
        
        # 2. Precision-Recall-F1 Comparison
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [results['precision'], results['recall'], results['f1_score']]
        colors = ['#3498DB', '#E74C3C', '#2Ecc71']
        
        bars = ax2.bar(metrics, values, color=colors, alpha=0.8)
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel('Score')
        ax2.set_title('B. Performance Metrics', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Error Rates
        error_rates = ['FPR', 'FNR']
        rate_values = [
            fp / (fp + tn) if (fp + tn) > 0 else 0,
            fn / (fn + tp) if (fn + tp) > 0 else 0
        ]
        
        bars = ax3.bar(error_rates, rate_values, color=['#F39C12', '#E74C3C'], alpha=0.8)
        ax3.set_ylim(0, max(rate_values) + 0.1)
        ax3.set_ylabel('Rate')
        ax3.set_title('C. Error Rates', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars, rate_values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01, 
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Security Impact
        security_labels = ['Attacks\nBlocked', 'Attacks\nMissed', 'Clean\nFlagged', 'Clean\nPassed']
        security_sizes = [tp, fn, fp, tn]
        security_colors = ['#3498DB', '#E74C3C', '#F39C12', '#2Ecc71']
        
        ax4.bar(security_labels, security_sizes, color=security_colors, alpha=0.8)
        ax4.set_ylabel('Count')
        ax4.set_title('D. Security Impact Analysis', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        for i, value in enumerate(security_sizes):
            ax4.text(i, value + 10, f'{value}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{self.results_dir}/error_analysis.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"   ✅ Visualizations saved to: {self.results_dir}/error_analysis.png")
    
    def generate_error_report(self, false_positives, false_negatives):
        """Generate comprehensive error analysis report"""
        report_path = f'{self.results_dir}/error_analysis_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("GUARD ERROR ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("FALSE POSITIVE ANALYSIS:\n")
            f.write(f"Total False Positives: {len(false_positives)}\n\n")
            
            f.write("Common FP Patterns:\n")
            # Analyze FP patterns in detail
            technical_terms_count = {}
            for _, row in false_positives.iterrows():
                text = row['resume'].lower()
                technical_terms = ['python', 'java', 'system', 'api', 'database', 'server', 'cloud']
                for term in technical_terms:
                    if term in text:
                        technical_terms_count[term] = technical_terms_count.get(term, 0) + 1
            
            for term, count in sorted(technical_terms_count.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {term}: {count} occurrences\n")
            
            f.write("\nFALSE NEGATIVE ANALYSIS:\n")
            f.write(f"Total False Negatives: {len(false_negatives)}\n\n")
            
            f.write("Sample FN Cases:\n")
            for idx, row in false_negatives.head(5).iterrows():
                f.write(f"\nFN Sample {idx}:\n")
                # Look for injection patterns
                text = row['resume'].lower()
                injection_indicators = [ind for ind in ['ignore', 'disregard', 'as an ai'] if ind in text]
                if injection_indicators:
                    f.write(f"  Contains: {injection_indicators}\n")
                else:
                    f.write("  No obvious injection patterns - likely sophisticated obfuscation\n")
        
        print(f"   📄 Detailed report saved to: {report_path}")

def main():
    """Main function to run error analysis"""
    print("🚀 Starting GUARD Error Analysis...")
    
    analyzer = GuardErrorAnalyzer()
    
    try:
        # Load existing results
        results = analyzer.load_latest_results()
        
        # Load dataset with predictions
        df = analyzer.load_dataset_with_predictions(results)
        
        # Perform comprehensive analysis
        false_positives, false_negatives = analyzer.perform_comprehensive_error_analysis(df, results)
        
        # Generate detailed report
        analyzer.generate_error_report(false_positives, false_negatives)
        
        print(f"\n🎉 Error analysis complete!")
        print(f"   Check {analyzer.results_dir}/ for visualizations and report")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("   Make sure you've run the main GUARD evaluation first")

if __name__ == "__main__":
    main()