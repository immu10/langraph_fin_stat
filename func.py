import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import embeddings

vectorstore = None

# ============= Document Loading =============

def load_documents(data_dir: str = "./data") -> List[str]:
    """Load documents from directory"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created {data_dir} directory. Add your documents there.")

    documents = []

    # Load text files
    try:
        txt_loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
    except Exception as e:
        print(f"Error loading text files: {e}")

    # Load PDF files
    try:
        pdf_loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
    except Exception as e:
        print(f"Error loading PDF files: {e}")

    return [doc.page_content for doc in documents]

# ============= Vector Store Management =============

def create_vectorstore(documents: List[str]) -> Chroma:
    """Create vector store from documents"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.create_documents(documents)
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

def get_vectorstore() -> Chroma:
    """Get or create vectorstore"""
    global vectorstore
    if vectorstore is None:
        print("Vectorstore is None, trying to load or create...")
        try:
            print("Attempting to load existing vectorstore...")
            vectorstore = Chroma(
                persist_directory="./chroma_db",
                embedding_function=embeddings
            )
            # Check if vectorstore has any documents
            if vectorstore._collection.count() == 0:
                print("Vectorstore exists but is empty, will create new one")
                raise Exception("Empty vectorstore")
            print("Successfully loaded existing vectorstore")
        except Exception as e:
            print(f"Failed to load existing vectorstore: {e}")
            print("Creating new vectorstore from documents...")
            documents = load_documents()
            print(f"Loaded {len(documents)} documents")
            if documents:
                try:
                    vectorstore = create_vectorstore(documents)
                    print("Successfully created vectorstore")
                except Exception as e:
                    print(f"Failed to create vectorstore: {e}")
                    raise e
            else:
                raise Exception("No documents found. Please add documents to ./data directory.")
    
    return vectorstore
