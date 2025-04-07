import streamlit as st
import ollama
import faiss
import numpy as np
import time
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer
import re
import os
import tempfile
from bs4 import BeautifulSoup
from typing import List, Dict
import html2text

# --- Configuration ---
MODEL_CACHE_PATH = "./local_models"
os.makedirs(MODEL_CACHE_PATH, exist_ok=True)

# --- Improved Document Processing ---
def load_embedding_model():
    """Load embedding model with offline fallback"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2', cache_folder=MODEL_CACHE_PATH)
    except Exception as e:
        st.error(f"""Model loading failed. Please:
                 1. Connect to internet for first-time download, or
                 2. Place pre-downloaded model in {MODEL_CACHE_PATH}/all-MiniLM-L6-v2
                 Error: {str(e)}""")
        st.stop()

def enhanced_web_loader(url: str) -> List[Dict]:
    """Load and clean web documents with BeautifulSoup"""
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=BeautifulSoup.SoupStrainer(
                    ["p", "h1", "h2", "h3", "h4", "h5", "li"]
                )
            ),
        )
        docs = loader.load()
        
        # Enhanced cleaning with html2text
        h = html2text.HTML2Text()
        h.ignore_links = True
        h.ignore_images = True
        
        for doc in docs:
            doc.page_content = h.handle(doc.page_content).strip()
        return docs
    except Exception as e:
        st.error(f"Failed to load web document: {str(e)}")
        return []

# --- RAG Core Components ---
class EnhancedRAG:
    def __init__(self):
        self.embedder = load_embedding_model()
        self.index = None
        self.documents = []
    
    def process_documents(self, pages: List[Dict]) -> None:
        """Process documents with semantic chunking"""
        # Header-aware splitting for better context
        headers = [("#", "Header 1"), ("##", "Header 2")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers)
        
        split_docs = []
        for doc in pages:
            try:
                splits = markdown_splitter.split_text(doc.page_content)
                split_docs.extend(splits)
            except:
                # Fallback to recursive splitting
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000, 
                    chunk_overlap=200
                )
                splits = text_splitter.split_documents([doc])
                split_docs.extend(splits)
        
        # Create FAISS index
        document_texts = [doc.page_content for doc in split_docs]
        embeddings = self.embedder.encode(document_texts)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        self.documents = document_texts
    
    def retrieve_context(self, query: str, k: int = 3) -> List[str]:
        """Enhanced retrieval with score thresholding"""
        query_embedding = self.embedder.encode([query])
        distances, indices = self.index.search(query_embedding.astype(np.float32), k)
        
        # Filter by similarity score (cosine distance < 0.3)
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if dist < 0.3:  # Adjust threshold as needed
                results.append(self.documents[i])
        return results or ["No relevant context found."]

# --- Streamlit UI ---
st.set_page_config(page_title="AI Document Chatbot", page_icon="📄", layout="wide")

# Initialize RAG system
if "rag" not in st.session_state:
    st.session_state.rag = EnhancedRAG()

# Document Upload Section
input_method = st.sidebar.radio("Input Method:", ("Upload File", "Enter URL"))
show_think = st.sidebar.checkbox("Show reasoning process")

if input_method == "Upload File":
    uploaded_file = st.file_uploader("Upload PDF/TXT", type=["pdf", "txt"])
    doc_url = None
else:
    doc_url = st.text_input("Enter URL", placeholder="https://example.com/document")
    uploaded_file = None

# Process Documents
if uploaded_file or doc_url:
    with st.spinner("Processing documents..."):
        try:
            if uploaded_file:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    if uploaded_file.name.endswith(".pdf"):
                        pages = PyPDFLoader(tmp_file.name).load()
                    else:
                        pages = TextLoader(tmp_file.name).load()
                    os.unlink(tmp_file.name)
            else:
                pages = enhanced_web_loader(doc_url)
            
            st.session_state.rag.process_documents(pages)
            st.success("Documents processed successfully!")
        except Exception as e:
            st.error(f"Document processing failed: {str(e)}")

# Query Interface
query = st.text_input("Ask about the document:", placeholder="Type your question...")
if st.button("Get Answer") and query:
    if not hasattr(st.session_state.rag, 'index'):
        st.error("Please load documents first")
    else:
        with st.spinner("Searching for answers..."):
            context = st.session_state.rag.retrieve_context(query)
            
            if show_think:
                with st.expander("Retrieved Context"):
                    st.write(context)
            
            prompt = f"""Answer this question based on the context:
            Question: {query}
            Context: {' '.join(context)}
            Answer:"""
            
            response = ollama.generate(
                model='deepseek-r1:1.5b',
                prompt=prompt,
                options={'temperature': 0.3}
            )
            
            st.markdown("### Answer")
            st.write(response['response'])
