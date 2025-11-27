import streamlit as st
import numpy as np
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import re
import string
import re
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

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


# def clean_text(text):
#     text = text.lower()  # Convert to lowercase
#     text = re.sub(r"\s+", " ", text)  # Remove extra spaces
#     text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
#     return text


def clean_text(text):
    """Apply full preprocessing to resume chunks and prompts"""
    text = text.lower()  # Lowercase
    text = re.sub(r"\s+", " ", text).strip()  # Fix spaces
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatize & remove stopwords
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
def detect_prompt_injections(text_chunks, batch_size=32):
    """
    Enhanced prompt injection detection with batch processing and optimizations.
    
    Args:
        text_chunks: List of text chunks to check for prompt injections
        batch_size: Number of chunks to process at once (to manage memory usage)
        
    Returns:
        List of tuples containing (flagged_chunk, similarity_score, prompt_template_matched)
    """
    text_chunks = [clean_text(chunk) for chunk in text_chunks]

    print(text_chunks)
    if not text_chunks:
        return []
    
    results = []
    
    # Process in batches to manage memory usage with large texts
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        
        # Skip empty chunks
        batch = [chunk for chunk in batch if chunk.strip()]
        if not batch:
            continue
            
        # Encode text chunks
        try:
            chunk_embeddings = model.encode(batch, convert_to_numpy=True, 
                                          show_progress_bar=False)
            
            # Process each embedding in the batch
            for j, chunk_embedding in enumerate(chunk_embeddings):
                # Calculate similarities
                similarities = cosine_similarity([chunk_embedding], prompt_embeddings)[0]
                
                # Find max similarity and which prompt template it matched
                max_similarity = max(similarities)
                template_index = similarities.argmax()
                
                # Check if above threshold
                if max_similarity >= SIMILARITY_THRESHOLD:
                    # Store the chunk, similarity score, and which template matched
                    prompt_template_matched = f"Template #{template_index+1}"
                    results.append((batch[j], max_similarity, prompt_template_matched))
                    
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {str(e)}")
            continue
                
    # Sort results by similarity score (highest first)
    results.sort(key=lambda x: x[1], reverse=True)
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
            
            # Create a container with a nice border
            with st.container():
                for i, (chunk, score, template) in enumerate(detections):
                    # Create an expandable section for each detection
                    with st.expander(f"Detection #{i+1} - Similarity: {score:.2f}"):
                        st.markdown(f"**Matched Template:** {template}")
                        st.markdown(f"**Similarity Score:** {score:.2f}")
                        st.code(chunk, language="text")
                        
                        # Add severity indicator based on similarity score
                        if score >= 0.85:
                            st.error("⚠️ High confidence match")
                        elif score >= 0.75:
                            st.warning("⚠️ Medium confidence match")
                        else:
                            st.info("ℹ️ Low confidence match")
            
            # Add suggestion for high scoring matches
            if any(score >= 0.85 for _, score, _ in detections):
                st.error("⚠️ High-confidence matches detected. Review and filter these inputs immediately.")
        else:
            st.success("✅ No prompt injections detected!")
            
        # Add metrics summary if there are detections
        if detections:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Detections", len(detections))
            with col2:
                high_conf = sum(1 for _, score, _ in detections if score >= 0.85)
                st.metric("High Confidence", high_conf)
            with col3:
                avg_score = sum(score for _, score, _ in detections) / len(detections)
                st.metric("Average Score", f"{avg_score:.2f}")
    else:
        st.error("Could not process the file. Please upload a valid TXT or PDF document.")
