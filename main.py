from langgraph.graph import START, StateGraph, END
from agents import GraphState, answer_relevancy_check, doc_required, retrieve_documents_for_question, generate_answer, should_retry
from funcs.func import split_summary

def build_rag_graph():
    """Build the RAG graph"""
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("doc_required", doc_required)
    workflow.add_node("retrieval", retrieve_documents_for_question)
    workflow.add_node("answer_relevancy_check", answer_relevancy_check)
    workflow.add_node("generate", generate_answer)
   

    # Add edges
    workflow.add_edge(START, "doc_required")
    workflow.add_edge("doc_required", "retrieval")
    workflow.add_edge("retrieval", "generate")
    workflow.add_edge("generate", "answer_relevancy_check")
    
    workflow.add_conditional_edges("answer_relevancy_check", should_retry, {
    "end": END,
    "retry": "retrieval"  # or whatever node you want to loop back to
})

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

        vector_store = split_summary()  # needs to change when defining ui
        # Run the graph
        result = rag_graph.invoke({
        "vector_store": vector_store,
        "question": question,
        "relevancy_check_count": 0  # initialize here
    })
        print(f"\nAnswer: {result['answer']}")

if __name__ == "__main__":
    main()