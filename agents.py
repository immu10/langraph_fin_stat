from typing import Dict, Any
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from config import llm
from langraph_fin_stat.funcs.func import get_vectorstore
from prompt import get_rag_prompt, get_relevency_prompt

class GraphState(TypedDict):
    vector_store : Any
    question: str
    context: str
    answer: str
    relevancy: str
    relevancy_check_count: int





def retrieve_documents_for_question(state: GraphState) -> Dict[str, Any]:
    """Retrieve relevant documents for the question"""
    print("Checking vectorstore...")
    vectorstore = state.get("vector_store", None)
    if vectorstore is None:
        raise ValueError("Vectorstore is not accessible. Please ensure it is created successfully.")
    
    print("Performing similarity search...")
    docs = vectorstore.similarity_search(state["question"], k=5)
    context = "\n\n".join([doc.page_content for doc in docs])
    print(f"Retrieved {len(docs)} documents written in context.txt")
    
    # Save context to file for debugging
    with open("context.txt", "w") as f:
        f.write(context)

    return {"context": context}

def answer_relevancy_check(state: GraphState) -> Dict[str, Any]:
    """Check if answer is relevant to question"""
    prompt = get_relevency_prompt()
    chain = prompt | llm | StrOutputParser()
    relevancy = chain.invoke({"answer": state["answer"]})
    
    return {
        "relevancy": relevancy,
        "relevancy_check_count": state["relevancy_check_count"] + 1
    }
def should_retry(state: GraphState) -> str:
    if state["relevancy"] == "yes" or state["relevancy_check_count"] >= 3:
        return "end"
    return "retry"

def generate_answer(state: GraphState) -> Dict[str, Any]:
    """Generate answer using retrieved context"""
    prompt = get_rag_prompt()
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": state["context"],
        "question": state["question"]
    })

    return {"answer": answer}
