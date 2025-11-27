# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import (
#     roc_curve, roc_auc_score, confusion_matrix, classification_report,
#     RocCurveDisplay, ConfusionMatrixDisplay
# )
# import pandas as pd

# # Dummy example (replace with your own data)
# y_true = np.array([0, 1, 1, 0, 1, 0, 1])  # Ground truth
# y_scores = np.array([0.1, 0.9, 0.8, 0.3, 0.85, 0.4, 0.95])  # Predicted probabilities

# # ROC Curve and AUC
# fpr, tpr, thresholds = roc_curve(y_true, y_scores)
# auc_score = roc_auc_score(y_true, y_scores)

# plt.figure(figsize=(6, 4))
# plt.plot(fpr, tpr, color='orange', label=f'ROC curve (AUC = {auc_score:.2f})')
# plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc='lower right')
# plt.tight_layout()
# plt.savefig("roc_curve.png", dpi=300)
# plt.close()

# # Classification Report and Confusion Matrix
# threshold = 0.5
# y_pred = (y_scores >= threshold).astype(int)

# # Confusion matrix as table
# cm = confusion_matrix(y_true, y_pred)
# cm_df = pd.DataFrame(cm, index=["Actual Neg", "Actual Pos"], columns=["Predicted Neg", "Predicted Pos"])
# cm_df.to_latex("confusion_matrix.tex", caption="Confusion Matrix at 0.5 Threshold", label="tab:conf_matrix")

# # Classification report as table
# report_dict = classification_report(y_true, y_pred, output_dict=True)
# report_df = pd.DataFrame(report_dict).transpose()
# report_df = report_df.round(2)
# report_df.to_latex("classification_report.tex", caption="Classification Report", label="tab:class_report")

# # AUC score summary (for LaTeX)
# with open("auc_score.tex", "w") as f:
#     f.write(f"AUC score = {auc_score:.2f}")
import matplotlib.pyplot as plt
import numpy as np

# Data
methods = [
    "Paragraph-Based",
    "Sentence-Aware",
    "Fine-Grained"
]
f1_scores = [0.720721, 0.737234, 0.753703]
recalls = [0.830091, 0.898833, 0.891051]
precisions = [0.636816, 0.624887, 0.653042]

# Settings
x = np.arange(len(methods))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
bar1 = ax.bar(x - width, f1_scores, width, label='F1-Score')
bar2 = ax.bar(x, recalls, width, label='Recall')
bar3 = ax.bar(x + width, precisions, width, label='Precision')

# Labels and style
ax.set_xlabel("Chunking Strategy", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Performance Comparison of Chunking Strategies", fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Annotate bars
for bars in [bar1, bar2, bar3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
