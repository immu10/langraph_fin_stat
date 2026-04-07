from typing import Dict, Any
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from config import llm
from func import get_vectorstore
from prompt import get_rag_prompt

class GraphState(TypedDict):
    question: str
    documents: list
    context: str
    answer: str

def retrieve_documents(state: GraphState) -> Dict[str, Any]:
    """Retrieve relevant documents for the question"""
    print("Checking vectorstore...")
    vectorstore = get_vectorstore()
    
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
