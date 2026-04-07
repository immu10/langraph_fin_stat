from typing import Dict, Any
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from config import llm
from langraph_fin_stat.funcs.func import get_vectorstore
from prompt import get_rag_prompt

class GraphState(TypedDict):
    question: str
    documents: list
    context: str
    answer: str

def create_doc_split(state: GraphState) -> Dict[str, Any]:
    """Takes document to make vector store for RAG"""
    

    
    return {"documents": state["documents"]}




def retrieve_documents_for_question(state: GraphState) -> Dict[str, Any]:
    """Retrieve relevant documents for the question"""
    print("Checking vectorstore...")
    vectorstore = get_vectorstore()
    if vectorstore is None:
        raise ValueError("Vectorstore is not accessible. Please ensure it is created successfully.")
    
    print("Performing similarity search...")
    docs = vectorstore.similarity_search(state["question"], k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"Retrieved {len(docs)} documents written in context.txt")
    
    # Save context to file for debugging
    with open("context.txt", "w") as f:
        f.write(context)

    return {"context": context}

def generate_answer(state: GraphState) -> Dict[str, Any]:
    """Generate answer using retrieved context"""
    prompt = get_rag_prompt()
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })

    return {"answer": answer}
