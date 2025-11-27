import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # Ensure sentence tokenizer is available

resumes = pd.read_csv("hf://datasets/Lakshmi12/Resume_Dataset/Resume.csv")['Resume_str'].tolist()

injected_prompts = pd.read_csv('dataset/resume_adversarial_prompts.csv')['Adversarial Prompt'].tolist()

# Process resumes: Chunk into sentences & label as legitimate (0)
resume_chunks = []
for resume in resumes:
    sentences = sent_tokenize(resume.strip())
    resume_chunks.extend([(sent, 0) for sent in sentences])  # Label as 0

# Process injected prompts: Label as 1
prompt_chunks = [(prompt.strip(), 1) for prompt in injected_prompts]

# Combine datasets
labeled_data = resume_chunks + prompt_chunks

# Convert to DataFrame
labeled_df = pd.DataFrame(labeled_data, columns=["text", "label"])

# Save to CSV for future use
labeled_df.to_csv("labeled_resume_chunks.csv", index=False, encoding="utf-8")

print("Labeled dataset prepared and saved as 'labeled_resume_data.csv'.")
