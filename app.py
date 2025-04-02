import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer
import requests
import os

# Original LSTM model
class LSTMSummarizer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        out = self.fc(lstm_out)
        return out, hidden

# Download model if not present
model_path = "lstm_summarizer.pth"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=YOUR_FILE_ID&export=download"
    r = requests.get(url, allow_redirects=True)
    with open(model_path, "wb") as f:
        f.write(r.content)

# Load tokenizer and model (CPU only)
try:
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    device = "cpu"
    model = LSTMSummarizer(vocab_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except Exception as e:
    st.error(f"Failed to load model/tokenizer: {e}")
    st.stop()

# Rest of the code (st.markdown, UI, generation) remains the same
st.markdown("""
    <style>
    .title {font-size: 36px; color: #2c3e50; text-align: center;}
    .subtitle {font-size: 18px; color: #7f8c8d; text-align: center;}
    .stTextArea textarea {background-color: #ecf0f1; border-radius: 10px;}
    .stButton button {background-color: #3498db; color: white; border-radius: 5px;}
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">Literature Summarizer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Generate text continuations from classic books</p>', unsafe_allow_html=True)

user_input = st.text_area("Enter text to continue", height=150, placeholder="e.g., 'Mr. Darcy, a wealthy but aloof gentleman,'")
if st.button("Generate"):
    with st.spinner("Generating..."):
        try:
            inputs = tokenizer(user_input, return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                hidden = None
                generated_ids = inputs
                for _ in range(50):
                    outputs, hidden = model(generated_ids, hidden)
                    next_token = torch.argmax(outputs[:, -1, :], dim=-1).unsqueeze(0)
                    generated_ids = torch.cat((generated_ids, next_token), dim=1)
                response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            st.success("Generated continuation:")
            st.write(response)
        except Exception as e:
            st.error(f"Generation failed: {e}")

st.sidebar.header("About")
st.sidebar.write("This app uses a custom LSTM model trained on classic literature for text generation (CPU-only).")
