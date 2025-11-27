import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.spatial.distance import cosine
import re
import string
import logging
from typing import List, Tuple, Dict
import time
from functools import wraps
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Alternative embedding models to test
MODEL_CONFIGS = {
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",  # Original
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",    # Smaller, faster
    "all-distilroberta-v1": "sentence-transformers/all-distilroberta-v1",  # Distilled version
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Multilingual
    "multi-qa-mpnet-base-dot-v1": "sentence-transformers/multi-qa-mpnet-base-dot-v1",  # QA optimized
}

class AlternativeEmbeddingGUARD:
    def __init__(self, model_name: str):
        """Initialize GUARD with alternative embedding model"""
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.prompt_embeddings = None
        
    def clean_text(self, text: str) -> str:
        """Apply full preprocessing to resume chunks and prompts"""
        text = text.lower()
        text = re.sub(r"\s+", " ", text).strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        # Simple word-based cleaning (removing nltk dependencies for robustness)
        words = text.split()
        cleaned_words = [word for word in words if len(word) > 2]  # Basic filtering
        return " ".join(cleaned_words)
    
    def chunk_text(self, text: str, chunk_size: int = 15, overlap: int = 5) -> List[str]:
        """Simple text chunking without nltk dependencies"""
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
    
    def compute_prompt_embeddings(self, prompts: List[str]) -> np.ndarray:
        """Compute embeddings for adversarial prompts"""
        cleaned_prompts = [self.clean_text(prompt) for prompt in prompts]
        embeddings = self.model.encode(cleaned_prompts, convert_to_numpy=True, show_progress_bar=True)
        return embeddings
    
    def detect_prompt_injections(
        self, 
        text_chunks: List[str], 
        similarity_threshold: float = 0.65
    ) -> List[Tuple[str, float, int]]:
        """Detect potential prompt injections in text chunks"""
        if self.prompt_embeddings is None:
            raise ValueError("Prompt embeddings not computed. Call compute_prompt_embeddings first.")
        
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
                
                if max_similarity >= similarity_threshold:
                    results.append((
                        cleaned_chunks[i], 
                        max_similarity, 
                        template_index
                    ))
        
        except Exception as e:
            logger.error(f"Error in detection with {self.model_name}: {e}")
            return []
        
        return results
    
    def process_resume_for_injection(
        self, 
        resume_text: str, 
        similarity_threshold: float = 0.65
    ) -> Tuple[int, float]:
        """Process resume and return (prediction, execution_time)"""
        start_time = time.time()
        
        chunks = self.chunk_text(resume_text)
        injections = self.detect_prompt_injections(chunks, similarity_threshold)
        
        execution_time = time.time() - start_time
        prediction = 1 if injections else 0
        
        return prediction, execution_time

def load_adversarial_prompts(csv_path: str) -> List[str]:
    """Load adversarial prompts from CSV file"""
    try:
        df = pd.read_csv(csv_path)
        # Assuming the prompts are in a column named 'prompt' or 'text'
        if 'prompt' in df.columns:
            prompts = df['prompt'].tolist()
        elif 'text' in df.columns:
            prompts = df['text'].tolist()
        else:
            # Use the first column
            prompts = df.iloc[:, 0].tolist()
        
        logger.info(f"Loaded {len(prompts)} adversarial prompts from {csv_path}")
        return prompts
    
    except Exception as e:
        logger.error(f"Error loading adversarial prompts: {e}")
        return []

def evaluate_model_performance(
    model: AlternativeEmbeddingGUARD,
    df: pd.DataFrame,
    similarity_threshold: float = 0.65
) -> Dict:
    """Evaluate model performance on the dataset"""
    predictions = []
    execution_times = []
    
    for resume in df['resume']:
        pred, exec_time = model.process_resume_for_injection(resume, similarity_threshold)
        predictions.append(pred)
        execution_times.append(exec_time)
    
    # Calculate metrics
    precision = precision_score(df['label'], predictions, zero_division=0)
    recall = recall_score(df['label'], predictions, zero_division=0)
    f1 = f1_score(df['label'], predictions, zero_division=0)
    cm = confusion_matrix(df['label'], predictions)
    
    # Performance statistics
    avg_time = np.mean(execution_times)
    throughput = len(df) / sum(execution_times)
    
    return {
        'model_name': model.model_name,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'avg_inference_time': avg_time,
        'throughput': throughput,
        'total_time': sum(execution_times),
        'predictions': predictions
    }

def run_ablation_study(
    dataset_path: str,
    prompts_path: str,
    output_dir: str = "ablation_results",
    similarity_threshold: float = 0.65
):
    """Run comprehensive ablation study with alternative embedding models"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    try:
        df = pd.read_csv(dataset_path)
        # Balance dataset if needed
        df_0 = df[df["label"] == 0]
        df_1 = df[df["label"] == 1]
        df = pd.concat([df_0, df_1]).reset_index(drop=True)
        logger.info(f"Loaded dataset with {len(df)} samples")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return
    
    # Load adversarial prompts
    adversarial_prompts = load_adversarial_prompts(prompts_path)
    if not adversarial_prompts:
        logger.error("No adversarial prompts loaded. Aborting.")
        return
    
    results = []
    
    for model_key, model_path in MODEL_CONFIGS.items():
        logger.info(f"=== Testing model: {model_key} ===")
        
        try:
            # Initialize model
            guard_model = AlternativeEmbeddingGUARD(model_path)
            
            # Compute prompt embeddings for this model
            logger.info(f"Computing prompt embeddings for {model_key}...")
            prompt_embeddings = guard_model.compute_prompt_embeddings(adversarial_prompts)
            guard_model.prompt_embeddings = prompt_embeddings
            
            # Evaluate performance
            model_results = evaluate_model_performance(guard_model, df, similarity_threshold)
            results.append(model_results)
            
            # Log results
            logger.info(f"Results for {model_key}:")
            logger.info(f"  F1-Score: {model_results['f1_score']:.4f}")
            logger.info(f"  Precision: {model_results['precision']:.4f}")
            logger.info(f"  Recall: {model_results['recall']:.4f}")
            logger.info(f"  Avg Time: {model_results['avg_inference_time']:.4f}s")
            logger.info(f"  Throughput: {model_results['throughput']:.2f} resumes/sec")
            
        except Exception as e:
            logger.error(f"Error testing model {model_key}: {e}")
            continue
    
    # Save comprehensive results
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(output_dir, "ablation_study_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Results saved to {results_csv_path}")
    
    # Generate summary report
    generate_summary_report(results_df, output_dir)
    
    return results_df

def generate_summary_report(results_df: pd.DataFrame, output_dir: str):
    """Generate a detailed summary report of the ablation study"""
    report_path = os.path.join(output_dir, "ablation_study_summary.md")
    
    with open(report_path, 'w') as f:
        f.write("# GUARD Ablation Study: Alternative Embedding Models\n\n")
        f.write("## Performance Summary\n\n")
        f.write("| Model | F1-Score | Precision | Recall | Avg Inference Time (s) | Throughput (resumes/sec) |\n")
        f.write("|-------|----------|-----------|--------|------------------------|--------------------------|\n")
        
        for _, row in results_df.iterrows():
            f.write(f"| {row['model_name']} | {row['f1_score']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} | {row['avg_inference_time']:.4f} | {row['throughput']:.2f} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # Find best performing models
        best_f1 = results_df.loc[results_df['f1_score'].idxmax()]
        fastest = results_df.loc[results_df['avg_inference_time'].idxmin()]
        best_throughput = results_df.loc[results_df['throughput'].idxmax()]
        
        f.write(f"- **Best F1-Score**: {best_f1['model_name']} (F1: {best_f1['f1_score']:.4f})\n")
        f.write(f"- **Fastest Inference**: {fastest['model_name']} ({fastest['avg_inference_time']:.4f}s per resume)\n")
        f.write(f"- **Highest Throughput**: {best_throughput['model_name']} ({best_throughput['throughput']:.2f} resumes/sec)\n")
        
        # Analysis
        f.write("\n## Analysis\n\n")
        f.write("1. **Performance vs Efficiency Trade-off**: Compare larger models (mpnet) vs smaller models (MiniLM)\n")
        f.write("2. **Multilingual Capabilities**: Assess paraphrase-multilingual model performance\n")
        f.write("3. **Domain Specialization**: Evaluate QA-optimized model effectiveness\n")
        f.write("4. **Practical Recommendations**: Suggest optimal models based on use case requirements\n")

def optimize_thresholds_for_models(
    dataset_path: str,
    prompts_path: str,
    models_to_test: List[str] = None
):
    """Optional: Test different similarity thresholds for each model"""
    if models_to_test is None:
        models_to_test = ["all-mpnet-base-v2", "all-MiniLM-L6-v2", "all-distilroberta-v1"]
    
    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    threshold_results = []
    
    # Load data
    df = pd.read_csv(dataset_path)
    df_0 = df[df["label"] == 0]
    df_1 = df[df["label"] == 1]
    df = pd.concat([df_0, df_1]).reset_index(drop=True)
    
    adversarial_prompts = load_adversarial_prompts(prompts_path)
    
    for model_key in models_to_test:
        if model_key not in MODEL_CONFIGS:
            continue
            
        guard_model = AlternativeEmbeddingGUARD(MODEL_CONFIGS[model_key])
        guard_model.prompt_embeddings = guard_model.compute_prompt_embeddings(adversarial_prompts)
        
        for threshold in thresholds:
            results = evaluate_model_performance(guard_model, df, threshold)
            results['threshold'] = threshold
            threshold_results.append(results)
            
            logger.info(f"Model: {model_key}, Threshold: {threshold}, F1: {results['f1_score']:.4f}")
    
    # Save threshold optimization results
    threshold_df = pd.DataFrame(threshold_results)
    threshold_df.to_csv("ablation_results/threshold_optimization.csv", index=False)
    
    return threshold_df

if __name__ == "__main__":
    # Paths to your files
    DATASET_PATH = "dataset/injected_resume_dataset.csv"  # Update path as needed
    PROMPTS_PATH = "dataset/adversarial_prompts.csv"  # Update path as needed
    
    # Run the main ablation study
    results = run_ablation_study(
        dataset_path=DATASET_PATH,
        prompts_path=PROMPTS_PATH,
        similarity_threshold=0.65  # You can adjust this
    )
    
    # Optional: Run threshold optimization
    threshold_results = optimize_thresholds_for_models(DATASET_PATH, PROMPTS_PATH)