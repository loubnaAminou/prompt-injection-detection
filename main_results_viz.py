# ===============================
#   Visualization for GUARD Results
#   Author: [Your Name]
#   Description: Generates Q1-level figures for main results section
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Aesthetics configuration
# -------------------------------
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# -------------------------------
# Load data
# -------------------------------
threshold_df = pd.read_csv("assets/threshold_analysis.csv")

# Optional: round numeric precision for clarity
threshold_df = threshold_df.round(3)

# -------------------------------
# Figure 1: Overall Performance (Precision–Recall–F1 + Confusion Matrix)
# -------------------------------
metrics_overall = {
    "Precision": threshold_df.loc[threshold_df["threshold"] == 0.65, "precision"].values[0],
    "Recall": threshold_df.loc[threshold_df["threshold"] == 0.65, "recall"].values[0],
    "F1-score": threshold_df.loc[threshold_df["threshold"] == 0.65, "f1"].values[0],
}

# Example confusion matrix parsing (string to array)
import ast
cm_str = threshold_df.loc[threshold_df["threshold"] == 0.65, "cm"].values[0]
cm = np.array(ast.literal_eval(cm_str))

# --- Subplot Layout ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# Bar chart
sns.barplot(x=list(metrics_overall.keys()), y=list(metrics_overall.values()), ax=axes[0], palette="viridis")
axes[0].set_ylim(0, 1)
axes[0].set_title("Overall Detection Performance at Threshold = 0.65")
axes[0].set_ylabel("Score")

# Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[1])
axes[1].set_title("Confusion Matrix")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")

plt.tight_layout()
plt.savefig("fig1_overall_performance.png", bbox_inches="tight")
plt.close()

# -------------------------------
# Figure 2: Threshold Sensitivity
# -------------------------------
plt.figure(figsize=(7, 5))
plt.plot(threshold_df["threshold"], threshold_df["precision"], marker='o', label="Precision")
plt.plot(threshold_df["threshold"], threshold_df["recall"], marker='s', label="Recall")
plt.plot(threshold_df["threshold"], threshold_df["f1"], marker='^', label="F1-score")
plt.xlabel("Similarity Threshold")
plt.ylabel("Score")
plt.title("Impact of Threshold on Detection Performance")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("fig2_threshold_sensitivity.png", bbox_inches="tight")
plt.close()

# -------------------------------
# Figure 3: Efficiency vs Accuracy Trade-off
# -------------------------------
# Example dictionary results from your benchmark (replace by actual metrics)
results_with = {"avg_inference_time": 0.45, "recall": 0.84, "f1_score": 0.74}
results_without = {"avg_inference_time": 0.31, "recall": 0.65, "f1_score": 0.63}

df_eff = pd.DataFrame([
    {"Method": "With Segmentation", "Avg Time (s)": results_with["avg_inference_time"],
     "Recall": results_with["recall"], "F1-score": results_with["f1_score"]},
    {"Method": "Without Segmentation", "Avg Time (s)": results_without["avg_inference_time"],
     "Recall": results_without["recall"], "F1-score": results_without["f1_score"]}
])

fig, ax1 = plt.subplots(figsize=(7, 5))
ax2 = ax1.twinx()

df_eff.plot(x="Method", y="Avg Time (s)", kind="bar", ax=ax1, color="lightblue", width=0.4, position=1)
df_eff.plot(x="Method", y="Recall", kind="line", ax=ax2, color="darkgreen", marker="o", linewidth=2, label="Recall")
df_eff.plot(x="Method", y="F1-score", kind="line", ax=ax2, color="orange", marker="s", linewidth=2, label="F1-score")

ax1.set_ylabel("Average Inference Time (s)")
ax2.set_ylabel("Performance Score")
ax1.set_title("Efficiency vs Accuracy Trade-off")
ax1.set_xlabel("")
ax1.grid(False)
ax2.legend(loc="lower right")
plt.tight_layout()
plt.savefig("fig3_efficiency_tradeoff.png", bbox_inches="tight")
plt.close()

# -------------------------------
# (Optional) Figure 4: Comparative Radar Plot
# -------------------------------
import math

labels = np.array(["Precision", "Recall", "F1-score", "Speed"])
with_seg = np.array([metrics_overall["Precision"], metrics_overall["Recall"],
                     metrics_overall["F1-score"], 1/results_with["avg_inference_time"]])
without_seg = np.array([0.85, 0.65, 0.63, 1/results_without["avg_inference_time"]])  # Example

angles = np.linspace(0, 2 * math.pi, len(labels), endpoint=False).tolist()
with_seg = np.concatenate((with_seg, [with_seg[0]]))
without_seg = np.concatenate((without_seg, [without_seg[0]]))
angles += angles[:1]

fig = plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)
ax.plot(angles, with_seg, 'o-', linewidth=2, label='With Segmentation')
ax.fill(angles, with_seg, alpha=0.25)
ax.plot(angles, without_seg, 'o-', linewidth=2, label='Without Segmentation')
ax.fill(angles, without_seg, alpha=0.25)
ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_title("Comparative Evaluation of GUARD Variants", y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.tight_layout()
plt.savefig("fig4_radar_comparison.png", bbox_inches="tight")
plt.close()

print("✅ All figures generated successfully.")
