import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import embeddings
from funcs.func2 import vector_store_init

vectorstore = None

def create_doc_split():
    """Creates Vector store for RAG"""
    vector_store,split = vector_store_init()
    return vector_store,split


def split_summary():
    """ create summary of each section for site"""
    vector_store,split = create_doc_split()
    return vector_store,split

# ============= Document Loading =============

# def load_documents(data_dir: str = "./data") -> List[str]:
#     """Load documents from directory"""
    
#     if not os.path.exists(data_dir):
#         os.makedirs(data_dir)
#         print(f"Created {data_dir} directory. Add your documents there.")
    
#     documents = []
    
#     # Load text files
#     try:
#         txt_loader = DirectoryLoader(data_dir, glob="*.txt", loader_cls=TextLoader)
#         txt_docs = txt_loader.load()
#         documents.extend(txt_docs)
#     except Exception as e:
#         print(f"Error loading text files: {e}")

#     # Load PDF files
#     try:
#         pdf_loader = DirectoryLoader(data_dir, glob="*.pdf", loader_cls=PyPDFLoader)
#         pdf_docs = pdf_loader.load()
#         documents.extend(pdf_docs)
#     except Exception as e:
#         print(f"Error loading PDF files: {e}")
#     documents = [doc.page_content for doc in documents]
#     try:
#         if not doc_validity(documents):
#             raise ValueError("Document validity check failed. Please ensure all documents are relevant and non-empty.")
#     except Exception as e:
#         print(f"Document validity check failed: {e}")
#         raise e

#     return documents

# def doc_validity(documents: List[str]) -> bool:
#     """Check if documents are valid (non-empty)"""
#     print("Validating documents...")
#     mandatory_terms = ["balance sheet", "cash flow"]
#     optional_terms = ["profit and loss", "income statement"]  # at least one required
#     for doc in documents:
#         if not doc.strip():
#             print("⚠️  Found empty document. Please ensure all documents have content.")
#             return False
        
#         doc_lower = doc.lower()
        
#         if not all(term in doc_lower for term in mandatory_terms):
#             print("⚠️  Document may not be relevant. Ensure it contains financial data like balance sheets, P&L, cash flow, etc.")
#             return False
        
#         if not any(term in doc_lower for term in optional_terms):
#             print("⚠️  Document may not be relevant. Ensure it contains financial data like balance sheets, P&L, cash flow, etc.")
#             return False
    
#     return True
# def docs_split():
#     """Split documents into chunks"""
   
#     documents =load_documents()
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     splits = text_splitter.create_documents(documents)
#     return splits

# # ============= Vector Store Management =============

# def create_vectorstore(documents: List[str]) -> Chroma:
#     """Create vector store from documents"""
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=200
#     )
#     splits = text_splitter.create_documents(documents)
#     vectorstore = Chroma.from_documents(
#         documents=splits,
#         embedding=embeddings,
#         persist_directory="./chroma_db"
#     )
#     return vectorstore

# def validate_vectorstore() -> Chroma:
#     """Sanity check and validate vectorstore - handles all error cases"""
#     global vectorstore
    
#     print("Vectorstore validation starting...")
    
#     # Try to load existing vectorstore
#     try:
#         print("Attempting to load existing vectorstore...")
#         vs = Chroma(
#             persist_directory="./chroma_db",
#             embedding_function=embeddings
#         )
        
#         # Check if vectorstore has any documents
#         try:
#             doc_count = vs._collection.count()
#             if doc_count == 0:
#                 print("⚠️  Vectorstore exists but is empty, creating new one...")
#                 raise Exception("Empty vectorstore")
#             print(f"✓ Successfully loaded existing vectorstore with {doc_count} documents")
#             vectorstore = vs
#             return vectorstore
#         except Exception as e:
#             print(f" Vectorstore collection check failed: {e}")
#             raise e
            
#     except Exception as e:
#         print(f"Failed to load existing vectorstore: {e}")
#         print("Creating new vectorstore from documents...")
        
#         # Try to load documents
#         try:
#             documents = load_documents()
#             print(f"✓ Loaded {len(documents)} documents")
            
#             if not documents:
#                 raise Exception("No documents found. Please add documents to ./data directory.")
            
#             # Try to create vectorstore
#             try:
#                 vectorstore = create_vectorstore(documents)
#                 print("✓ Successfully created vectorstore")
#                 return vectorstore
#             except Exception as create_err:
#                 print(f"✗ Failed to create vectorstore: {create_err}")
#                 raise create_err
                
#         except Exception as load_err:
#             print(f"✗ Failed to load documents: {load_err}")
#             raise load_err

# def get_vectorstore() -> Chroma:
#     """Get or create vectorstore - wrapper around validate_vectorstore"""
#     global vectorstore
#     if vectorstore is None:
#         vectorstore = validate_vectorstore()
#     return vectorstore





if __name__ == "__main__":
    pass
    # For testing purposes, we can run the vectorstore validation directly
    # try:
        
    # except Exception as e:
    #     print(f"Error during validity check: {e}")