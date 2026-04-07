from langgraph.graph import StateGraph, END
from agents import GraphState, retrieve_documents, generate_answer

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