import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from bert_score import score
import pandas as pd
from nltk.tokenize import sent_tokenize
import re
import string
# Load the pre-trained Sentence-BERT model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)


df = pd.read_csv('dataset/resume_adversarial_prompts.csv')
prompt_texts = df["Adversarial Prompt"].dropna().tolist()


# Load the stored prompt embeddings
data = np.load('/home/aminou/prompt-injection-detection/dataset/prompts_embeddings.npz')
prompt_embeddings = data["embeddings"]

# Cosine similarity threshold (tune this for better precision-recall balance)
SIMILARITY_THRESHOLD = 0.6


### 📌 Function to Process PDF and TXT Files ###
def extract_text(file):
    """Extract text from PDF or TXT file"""
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")  # Read text file
    elif file.name.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        return "".join([pdf_reader.pages[i].extract_text() for i in range(len(pdf_reader.pages))])
    else:
        return None

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    return text


### 📌 Function to Chunk Text ###
def chunk_text(text, chunk_size=15, overlap=5):
    """
    Improved text chunking with sentence awareness.
    - Respects sentence boundaries
    - Maintains word overlap between chunks
    - chunk_size: target number of words per chunk
    - overlap: number of words to overlap between chunks
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk_words = []
    
    for sentence in sentences:
        sentence_words = sentence.split()
        
        # If adding this sentence doesn't exceed chunk size, add it
        if len(current_chunk_words) + len(sentence_words) <= chunk_size:
            current_chunk_words.extend(sentence_words)
        else:
            # If we have words in the current chunk, save it
            if current_chunk_words:
                chunks.append(" ".join(current_chunk_words))
            
            # Start a new chunk with this sentence
            current_chunk_words = sentence_words
            
            # If this single sentence exceeds chunk size, split it while preserving overlap
            while len(current_chunk_words) > chunk_size:
                chunks.append(" ".join(current_chunk_words[:chunk_size]))
                current_chunk_words = current_chunk_words[chunk_size-overlap:]
    
    # Don't forget the last chunk if it has content
    if current_chunk_words:
        chunks.append(" ".join(current_chunk_words))
    
    return chunks


### 📌 Function to Detect Prompt Injections ###
def detect_prompt_injections(text_chunks):
    """
    Encode text chunks and compare with stored prompt embeddings using cosine similarity.
    """

    text_chunks = [clean_text(chunk) for chunk in text_chunks]
    prompt_texts = [clean_text(chunk) for chunk in prompt_texts]


    print(text_chunks)
    results = []
    # chunk_embeddings = model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=False)

    print("=====> calculating score")
    for i, chunk in enumerate(text_chunks):
        similarities = []  # Store similarity scores for this chunk

        print("=====> calculating each chunk with all prompts")

        for prompt in prompt_texts:  # Compare each chunk with all prompts
            P, R, F1 = score([chunk], [prompt], lang="en", model_type="microsoft/deberta-xlarge-mnli")
            similarities.append(F1.item())  # Convert tensor to float

        max_f1 = max(similarities)  # Take the highest similarity score

        print("=====> highest similarity")

        if max_f1 >= SIMILARITY_THRESHOLD:
            results.append((text_chunks[i], max_f1))

    return results


### 📌 Streamlit UI ###
st.title("📄 Resume Prompt Injection Detection")
st.write("Upload a resume (TXT or PDF) to check for prompt injections.")

print("=====> hello world")


# File uploader
uploaded_file = st.file_uploader("Choose a TXT or PDF file", type=["txt", "pdf"])

if uploaded_file:
    # Extract text from the uploaded file
    resume_text = extract_text(uploaded_file)

    if resume_text:
        st.subheader("📜 Extracted Resume Text")
        st.text_area("Resume Content", resume_text[:1000] + "...", height=200)  # Show preview

        # Chunk the text
        text_chunks = chunk_text(resume_text)

        print("=====> detecting injections")


        # Detect prompt injections
        detections = detect_prompt_injections(text_chunks)

        # Show results
        st.subheader("🚨 Detection Results")
        if detections:
            st.warning(f"⚠️ {len(detections)} potential prompt injection(s) found!")
            for chunk, score in detections:
                st.write(f"**Similarity:** {score:.2f}")
                st.code(chunk, language="text")
        else:
            st.success("✅ No prompt injections detected!")
    else:
        st.error("Could not process the file. Please upload a valid TXT or PDF document.")
