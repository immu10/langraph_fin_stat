from typing import Dict, Any
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
from config import llm
from prompt import get_rag_prompt, get_relevency_prompt, get_documents_req_prompt, get_query_construction_prompt

class GraphState(TypedDict):
    vector_store : Any
    documents_required: list
    question: str
    query:str
    context: str
    answer: str
    relevancy: str
    relevancy_check_count: int
    summary: Any

ALLOWED_DOCS = {"balance_sheet", "income_statement", "cash_flow"}


def _normalize_documents_required(result: Any) -> list:
    """Normalize parser output into a clean list of allowed categories."""
    if isinstance(result, dict):
        docs = result.get("documents_required", [])
    elif isinstance(result, list):
        docs = result
    else:
        docs = []

    if isinstance(docs, str):
        docs = [docs]

    normalized = [d for d in docs if isinstance(d, str) and d in ALLOWED_DOCS]
    return normalized or ["income_statement", "balance_sheet", "cash_flow"]


def doc_required(state: GraphState) -> dict:
    """Determine which documents are required"""
    
    response_schemas = [
        ResponseSchema(
            name="documents_required",
            description="List of required documents"
        )
    ]

    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    prompt = get_documents_req_prompt()

    chain = prompt | llm | output_parser
    result = chain.invoke({"question": state["question"], "format_instructions": format_instructions})
    docs_required = _normalize_documents_required(result)
    return {"documents_required": docs_required}

def query_construction(state: GraphState) -> dict:
    """Construct query for vectorstore retrieval"""
    # For simplicity, we will just use the original question as the query
    # In a real implementation, you might want to do some processing here
    prompt = get_query_construction_prompt()
    chain = prompt | llm | StrOutputParser()
    query = chain.invoke({"question": state["question"], "documents_required": state["documents_required"]})
    print(f"Constructed query: {query}")
    return {"query": query}  




def retrieve_documents_for_question(state: GraphState) -> Dict[str, Any]:
    """Retrieve relevant documents for the question"""
    print("Checking vectorstore...")
    vectorstore = state.get("vector_store", None)
    if vectorstore is None:
        raise ValueError("Vectorstore is not accessible. Please ensure it is created successfully.")
    
    print("Performing similarity search...")
    docs = vectorstore.similarity_search(state["query"], k=5)
    context = "\n\n".join([doc for doc in docs])
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

def answer_relevancy_check(state: GraphState) -> Dict[str, Any]:
    """Check if answer is relevant to question"""
    prompt = get_relevency_prompt()
    chain = prompt | llm | StrOutputParser()
    relevancy = chain.invoke({"answer": state["answer"], "question": state["question"]})
    # print(relevancy["explanation"])
    
    return {
        "relevancy": relevancy.strip().lower(),
        "relevancy_check_count": state["relevancy_check_count"] + 1
    }

def should_retry(state: GraphState) -> str:
    if state["relevancy"] == "yes" or state["relevancy_check_count"] >= 3:
        return "end"
    return "retry"

if __name__ == "__main__":
    from langchain_community.vectorstores import Chroma
    vector_store = Chroma(persist_directory="./chroma_db") 
    query = "profit"
    docs = vector_store.get()

    for i, text in enumerate(docs["documents"]):
        print(f"\n--- Document {i+1} ---")
        print(text)
    print(f"Retrieved {len(docs)} documents:")
    print("IDs:", docs["ids"])
    print("Number of documents:", len(docs["ids"]))