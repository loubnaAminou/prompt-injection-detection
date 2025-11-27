import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns

conf_matrix = np.array([[695,76],[290,481]])


# Confusion Matrix Heatmap
# plt.figure(figsize=(6, 6))
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Injection", "Injection"], yticklabels=["Non-Injection", "Injection"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.title("Confusion Matrix")
# plt.savefig("assets/confusion_matrix.png")
# plt.close()

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

# Load prompt embeddings
try:
    prompt_data = np.load('/home/aminou/prompt-injection-detection/dataset/prompts_embeddings.npz')
    prompt_embeddings = prompt_data['embeddings']
except Exception as e:
    logger.error(f"Error loading prompt embeddings: {e}")
    exit()

# Load dataset
try:
    df = pd.read_csv("dataset/injected_resume_dataset.csv")
    df = df.sample(n=min(len(df[df["label"] == 0]), len(df[df["label"] == 1])), random_state=42)  # Balance dataset
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    exit()

# Compute resume embeddings only once
logger.info("Computing resume embeddings...")
df['resume_embeddings'] = df['resume'].apply(lambda x: model.encode(x, convert_to_numpy=True))

# Define thresholds to test
thresholds = np.arange(0.5, 0.85, 0.05)  # 0.50 to 0.80

# Store evaluation results
f1_scores, precisions, recalls = [], [], []

# Compute similarity in batch
resume_embeddings = np.stack(df['resume_embeddings'])
similarities = 1 - cdist(resume_embeddings, prompt_embeddings, metric='cosine')
max_similarities = similarities.max(axis=1)

# Evaluate different thresholds
for threshold in thresholds:
    df['predicted'] = (max_similarities >= threshold).astype(int)
    
    precision = precision_score(df['label'], df['predicted'], zero_division=0)
    recall = recall_score(df['label'], df['predicted'], zero_division=0)
    f1 = f1_score(df['label'], df['predicted'], zero_division=0)

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

    logger.info(f"Threshold: {threshold:.2f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")

# Find best threshold
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
logger.info(f"Best threshold: {best_threshold:.2f} with F1-score: {f1_scores[best_idx]:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_scores, label="F1-score", marker="o", linestyle="-")
plt.plot(thresholds, precisions, label="Precision", marker="s", linestyle="--")
plt.plot(thresholds, recalls, label="Recall", marker="d", linestyle=":")
plt.axvline(x=best_threshold, color='red', linestyle="--", label=f"Best Threshold ({best_threshold:.2f})")

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1-score vs Threshold")
plt.legend()
plt.grid()
plt.savefig("assets/image.png")
print("Plots saved in assets folder.")
