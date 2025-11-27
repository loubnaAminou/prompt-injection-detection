# GUARD Ablation Study: Alternative Embedding Models

## Performance Summary

| Model | F1-Score | Precision | Recall | Avg Inference Time (s) | Throughput (resumes/sec) |
|-------|----------|-----------|--------|------------------------|--------------------------|
| sentence-transformers/all-mpnet-base-v2 | 0.7159 | 0.8807 | 0.6031 | 0.6612 | 1.51 |
| sentence-transformers/all-MiniLM-L6-v2 | 0.7640 | 0.9377 | 0.6446 | 0.6209 | 1.61 |
| sentence-transformers/all-distilroberta-v1 | 0.7479 | 0.8907 | 0.6446 | 0.6313 | 1.58 |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 0.7244 | 0.5923 | 0.9326 | 0.6228 | 1.61 |
| sentence-transformers/multi-qa-mpnet-base-dot-v1 | 0.7697 | 0.6632 | 0.9170 | 0.6537 | 1.53 |

## Key Findings

- **Best F1-Score**: sentence-transformers/multi-qa-mpnet-base-dot-v1 (F1: 0.7697)
- **Fastest Inference**: sentence-transformers/all-MiniLM-L6-v2 (0.6209s per resume)
- **Highest Throughput**: sentence-transformers/all-MiniLM-L6-v2 (1.61 resumes/sec)

## Analysis

1. **Performance vs Efficiency Trade-off**: Compare larger models (mpnet) vs smaller models (MiniLM)
2. **Multilingual Capabilities**: Assess paraphrase-multilingual model performance
3. **Domain Specialization**: Evaluate QA-optimized model effectiveness
4. **Practical Recommendations**: Suggest optimal models based on use case requirements
