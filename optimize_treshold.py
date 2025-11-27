import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, f1_score
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# Load SBERT model
print("Loading SBERT model...")
model = SentenceTransformer('all-mpnet-base-v2')

# Load your dataset
print("Loading dataset...")
df = pd.read_csv('labeled_resume_chunks.csv')

# Assuming your CSV has these columns: 'text' for resume chunks and 'label' (1 for injection, 0 for clean)
texts = df['text'].tolist()
labels = df['label'].tolist()

# Load your prompt injection examples
print("Loading prompt injection examples...")
injection_df = pd.read_csv('dataset/resume_adversarial_prompts.csv')  # Or however you have your examples stored
injection_examples = injection_df['Adversarial Prompt'].tolist()  # Adjust column name as needed

# Encode all texts and injections
print("Encoding resume chunks...")
resume_embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

print("Encoding injection examples...")
injection_embeddings = model.encode(injection_examples, convert_to_numpy=True, show_progress_bar=True)

# Compute similarity scores for each resume chunk against all injections
print("Computing similarity scores...")
similarity_scores = []

for resume_emb in tqdm(resume_embeddings):
    # Calculate cosine similarity with each injection example
    similarities = np.dot(injection_embeddings, resume_emb) / (
            np.linalg.norm(injection_embeddings, axis=1) * np.linalg.norm(resume_emb)
    )
    # Take the maximum similarity as the score
    max_similarity = np.max(similarities)
    similarity_scores.append(max_similarity)

# Make sure lengths match
assert len(labels) == len(similarity_scores), "Lengths of labels and scores don't match"

# Calculate precision-recall curve
print("Calculating precision-recall curve...")
precisions, recalls, thresholds = precision_recall_curve(labels, similarity_scores)

# Calculate F1 scores for each threshold
f1_scores = []
for i in range(len(thresholds)):
    f1 = 2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]) if (precisions[i] + recalls[i]) > 0 else 0
    f1_scores.append(f1)

# Find optimal threshold (max F1 score)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]
optimal_precision = precisions[optimal_idx]
optimal_recall = recalls[optimal_idx]
optimal_f1 = f1_scores[optimal_idx]

# Plot precision-recall curve
plt.figure(figsize=(10, 8))
plt.plot(recalls, precisions, marker='.', label=f'PR curve (AUC={auc(recalls, precisions):.3f})')
plt.scatter([optimal_recall], [optimal_precision], color='red', s=100, zorder=10,
            label=f'Optimal threshold={optimal_threshold:.3f} (F1={optimal_f1:.3f})')

# Add reference point for 0.75 threshold
threshold_075_idx = np.argmin(np.abs(thresholds - 0.75))
plt.scatter([recalls[threshold_075_idx]], [precisions[threshold_075_idx]], color='green', s=100, zorder=5,
            label=f'Threshold=0.75 (F1={f1_scores[threshold_075_idx]:.3f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Prompt Injection Detection')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('precision_recall_curve.png')

# Print results
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"Optimal precision: {optimal_precision:.4f}")
print(f"Optimal recall: {optimal_recall:.4f}")
print(f"Optimal F1 score: {optimal_f1:.4f}")
print(f"\nPerformance at threshold=0.75:")
print(f"Precision: {precisions[threshold_075_idx]:.4f}")
print(f"Recall: {recalls[threshold_075_idx]:.4f}")
print(f"F1 score: {f1_scores[threshold_075_idx]:.4f}")

# Output threshold vs metrics as a table
thresholds_to_check = np.arange(0.5, 1.0, 0.05)
results = []

for threshold in thresholds_to_check:
    idx = np.argmin(np.abs(thresholds - threshold))
    results.append({
        'threshold': threshold,
        'precision': precisions[idx],
        'recall': recalls[idx],
        'f1': f1_scores[idx]
    })

results_df = pd.DataFrame(results)
print("\nThreshold comparison table:")
print(results_df.to_string(index=False, float_format='{:.4f}'.format))
# results_df.to_csv('threshold_metrics.csv', index=False)

# Plot the curve
plt.figure(figsize=(8, 5))
plt.plot(recalls, precisions, marker='o', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Prompt Injection Detection')
plt.legend()
plt.grid()
plt.savefig("plot.png")