from langgraph.graph import START, StateGraph, END
from config import embeddings
from agents import GraphState, answer_relevancy_check, doc_required, query_construction, retrieve_documents_for_question, generate_answer, should_retry
from langchain_community.vectorstores import Chroma
import subprocess
import sys
import os
import warnings

def build_rag_graph():
    """Build the RAG graph"""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("doc_required", doc_required)
    workflow.add_node("query_construction", query_construction)
    workflow.add_node("retrieval", retrieve_documents_for_question)
    workflow.add_node("answer_relevancy_check", answer_relevancy_check)
    workflow.add_node("generate", generate_answer)
   

    # Add edges
    workflow.add_edge(START, "doc_required")
    workflow.add_edge("doc_required", "query_construction")
    workflow.add_edge("query_construction", "retrieval")
    workflow.add_edge("retrieval", "generate")
    workflow.add_edge("generate", "answer_relevancy_check")
    
    workflow.add_conditional_edges("answer_relevancy_check", should_retry, {
    "end": END,
    "retry": "query_construction"  # or whatever node you want to loop back to
})

    return workflow.compile()

def rag_flow(question):
    """Main function to run the RAG system"""
    print("LangGraph RAG System")
    print("===================")

    # Build the graph
    

    # Example usage
    # while True:
    #     question = input("\nEnter your question (or 'quit' to exit): ")
    #     if question.lower() == 'quit':
    #         break

      # needs to change when defining ui
        
      
        
        
        # Run the graph
    print("\nProcessing question through RAG graph...")
    print(f"Question: {question}")
    result = rag_graph.invoke({
        "vector_store": vector_store,
        "question": question,
        "relevancy_check_count": 0  # initialize here
    })
    print(f"\nAnswer: {result['answer']}")
    return result['answer']

# Initialize globals at module level so they are available when imported
vector_store = Chroma(persist_directory="./chroma_db")
rag_graph = build_rag_graph()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    # rag_flow("what make up the expenses of the company?")
    ui_path = os.path.join(os.path.dirname(__file__), "ui.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", ui_path], check=True)