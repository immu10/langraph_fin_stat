import os
from typing import List, Dict, Any
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

# Define the state for our graph
class GraphState(TypedDict):
    question: str
    documents: List[str]
    context: str
    answer: str

# Set local cache directory for sentence transformers
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "./models"

# Initialize components
print("Loading embedding model (this may take a moment on first run)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded successfully!")
llm = Ollama(model="llama2:13b", base_url="http://localhost:11434", temperature=0.0)

# Vector store (will be created/loaded)
vectorstore = None

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

def retrieve_documents(state: GraphState) -> Dict[str, Any]:
    """Retrieve relevant documents for the question"""
    global vectorstore
    print("Checking vectorstore...")
    if vectorstore is None:
        print("Vectorstore is None, trying to load or create...")
        # Load existing vectorstore or create new one
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
            # If no existing vectorstore, create from documents
            documents = load_documents()
            print(f"Loaded {len(documents)} documents")
            if documents:
                try:
                    vectorstore = create_vectorstore(documents)
                    print("Successfully created vectorstore")
                except Exception as e:
                    print(f"Failed to create vectorstore: {e}")
                    return {"context": f"Error creating vectorstore: {e}"}
            else:
                return {"context": "No documents found. Please add documents to ./data directory."}

    if vectorstore:
        print("Performing similarity search...")
        docs = vectorstore.similarity_search(state["question"], k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        print(f"Retrieved {len(docs)} documents")
    else:
        context = "No vector store available."
        print("No vectorstore available")

    return {"context": context}

def generate_answer(state: GraphState) -> Dict[str, Any]:
    """Generate answer using retrieved context"""
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question.
    If you don't know the answer based on the context, say so.

    Context: {context}

    Question: {question}

    Answer:""")

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })

    return {"answer": answer}

def build_rag_graph():
    """Build the RAG graph"""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)

    # Add edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

def main():
    """Main function to run the RAG system"""
    print("LangGraph RAG System")
    print("===================")

    # Build the graph
    rag_graph = build_rag_graph()

    # Example usage
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        # Run the graph
        result = rag_graph.invoke({"question": question})
        print(f"\nAnswer: {result['answer']}")

if __name__ == "__main__":
    main()