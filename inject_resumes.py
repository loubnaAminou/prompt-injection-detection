import pandas as pd
import random
from textattack.transformations import WordSwapRandomCharacterDeletion
from textattack.augmentation import Augmenter

# ✅ Load Dataset
resume_df = pd.read_csv("hf://datasets/kvsrkc/Resume/Resume.csv")  # Change if needed
malicious_prompts = pd.read_csv('dataset/adversarial_prompts.csv')['Adversarial Prompt'].tolist()

# ✅ Ensure Equal Distribution
num_injected_resumes = len(malicious_prompts)  # Keep both classes equal

# ✅ Select Resumes for Injection
selected_resumes = random.sample(resume_df['Resume_str'].tolist(), num_injected_resumes)

# ✅ Define NLP-Based Augmentation (for prompt paraphrasing only)
augmenter = Augmenter(transformation=WordSwapRandomCharacterDeletion())

def paraphrase_prompt(prompt):
    """Paraphrase an adversarial prompt for slight diversity."""
    augmented = augmenter.augment(prompt)
    return augmented[0] if augmented else prompt  # Return paraphrased or original

def inject_prompt(resume, prompt):
    """Injects a paraphrased adversarial prompt into a resume at a random position."""
    words = resume.split()
    insert_position = random.randint(0, len(words))
    words.insert(insert_position,prompt )# paraphrase_prompt()
    return " ".join(words)

# ✅ Inject Malicious Prompts into Selected Resumes
injected_resumes = [inject_prompt(resume, prompt) for resume, prompt in zip(selected_resumes, malicious_prompts)]

# ✅ Select an Equal Number of Legitimate Resumes
remaining_resumes = list(set(resume_df['Resume_str'].tolist()) - set(selected_resumes))
legitimate_resumes = random.sample(remaining_resumes, num_injected_resumes)  # Equal to injected resumes

# ✅ Create Final Dataset
data = {
    'resume': legitimate_resumes + injected_resumes,
    'label': [0] * len(legitimate_resumes) + [1] * len(injected_resumes)  # 0 = legitimate, 1 = injected
}

# ✅ Save to CSV
combined_df = pd.DataFrame(data)
combined_df.to_csv('dataset/injected_resume_dataset.csv', index=False)

# ✅ Check Class Distribution
print("Final Class Distribution:", combined_df['label'].value_counts().to_dict())
