import pandas as pd
import numpy as np
import time
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
import re
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

print('==> Step 1: Load prompts dataset')
dataset = pd.read_csv('/home/aminou/prompt-injection-detection/dataset/adversarial_prompts.csv')
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

# dataset_train = dataset.iloc[:int(0.8 * len(dataset))]
# dataset_test = dataset.iloc[int(0.8 * len(dataset)):] 

# print('==> Step 2: Load SBERT model')
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print('==> Step 2: Load Multi QA MPNet model')
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-dot-v1')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """Apply full preprocessing to resume chunks and prompts"""
    text = text.lower()  # Lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Fix spaces
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatize & remove stopwords
    return text

train_texts = dataset['Adversarial Prompt'].values
# test_texts = dataset_test['Adversarial Prompt'].values

# train_texts = [clean_text(text) for text in train_texts]


# Step 3: Encode dataset using SBERT
print('==> Step 3: Encode dataset')
train_embeddings = model.encode(list(train_texts), convert_to_numpy=True, show_progress_bar=True)
# test_embeddings = model.encode(list(test_texts), convert_to_numpy=True, show_progress_bar=True)

# Step 4: Create dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.embeddings[idx]

    def __len__(self):
        return len(self.embeddings)

train_dataset = CustomDataset(train_embeddings)
# test_dataset = CustomDataset(test_embeddings)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Save train embeddings
np.savez_compressed("/home/aminou/prompt-injection-detection/dataset/prompts_embeddings.npz", 
                    embeddings=train_embeddings)

# Save test embeddings
# np.savez_compressed("/home/aminou/prompt-injection-detection/dataset/test_embeddings.npz", embeddings=test_embeddings)

print("Embeddings saved successfully!")
