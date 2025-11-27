import pandas as pd
import numpy as np
import time
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer

print('==> Step 1: Load prompts dataset')
dataset = pd.read_csv('/home/aminou/prompt-injection-detection/dataset/combined_resume_dataset.csv')
dataset = dataset.sample(frac=1, random_state=42).reset_index(drop=True)

dataset = dataset[dataset['label'] == 1]

dataset_train = dataset.iloc[:int(0.8 * len(dataset))]
dataset_test = dataset.iloc[int(0.8 * len(dataset)):] 

# print('==> Step 2: Load SBERT model')
# model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

print('==> Step 2: Load MiniLM model')
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


train_texts = dataset_train['resume'].values
test_texts = dataset_test['resume'].values

# Step 3: Encode dataset using SBERT
print('==> Step 3: Encode dataset')
train_embeddings = model.encode(list(train_texts), convert_to_numpy=True, show_progress_bar=True)
test_embeddings = model.encode(list(test_texts), convert_to_numpy=True, show_progress_bar=True)

# Step 4: Create dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, embeddings):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)

    def __getitem__(self, idx):
        return self.embeddings[idx]

    def __len__(self):
        return len(self.embeddings)

train_dataset = CustomDataset(train_embeddings)
test_dataset = CustomDataset(test_embeddings)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Save train embeddings
np.savez_compressed("/home/aminou/prompt-injection-detection/dataset/resume_train_embeddings.npz", 
                    embeddings=train_embeddings)

# Save test embeddings
np.savez_compressed("/home/aminou/prompt-injection-detection/dataset/resume_test_embeddings.npz", 
                    embeddings=test_embeddings)

print("Embeddings saved successfully!")
