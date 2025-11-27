import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cityblock
from nltk.tokenize import sent_tokenize

# Load the pre-trained Sentence-BERT model
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(MODEL_NAME)

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
    print(text_chunks)
    results = []
    chunk_embeddings = model.encode(text_chunks, convert_to_numpy=True, show_progress_bar=False)

    for i, chunk_embedding in enumerate(chunk_embeddings):
        distances = [cityblock(chunk_embedding, prompt_embedding) for prompt_embedding in prompt_embeddings]
        min_distance = min(distances)

        if min_distance <= SIMILARITY_THRESHOLD:
            results.append((text_chunks[i], min_distance))

    return results


### 📌 Streamlit UI ###
st.title("📄 Resume Prompt Injection Detection")
st.write("Upload a resume (TXT or PDF) to check for prompt injections.")

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
