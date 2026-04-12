import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# Set local cache directory for sentence transformers
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

# Initialize components
print("Loading embedding model (this may take a moment on first run)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded successfully!")

llm = OllamaLLM(
    model="llama2:13b",
    base_url="http://localhost:11434",
    temperature=0.0
)