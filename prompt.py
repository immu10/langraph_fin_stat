from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt() -> ChatPromptTemplate:
    """Get the RAG prompt template"""
    return ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question.
    If you don't know the answer based on the context, say so.

    Context: {context}

    Question: {question}

    Answer:""")
def get_relevency_prompt() -> ChatPromptTemplate:
    """Get the relevancy check prompt template"""
    return ChatPromptTemplate.from_template("""
    You are a helpful assistant. Check if the answer is relevant to the question.

    Question: {question}

    Answer: {answer}

    Is the answer relevant to the question? Answer "Yes" or "No" only.
    """)
def get_documents_req_prompt() -> ChatPromptTemplate:
    """Get the document requirement prompt template"""
    return ChatPromptTemplate.from_template("""
    You are a helpful assistant. Determine which documents of the three  are required to answer the question.
                                            
    Output format: {format_instructions}
    Question: {question}

    List the documents required from the following options: ["balance_sheet", "income_statement", "cash_flow"] only in a json format (curly brackets always) nothing else.
    """)
def get_query_construction_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template("""
You are a query generator for a vector database.

Your task is to construct a search query using the given question and required documents.

STRICT RULES:
- Output ONLY the final query
- Do NOT include explanations
- Do NOT include prefixes like "Sure", "Here is", "Query =", etc.
- Do NOT include any extra text, markdown, or formatting
- Output must be a single line
- Do NOT wrap the output in quotes
- Do NOT use natural language sentences
- Use only keywords with spaces in between each suitable for vector search
  balance_sheets are assets and liabilities, income_statements are profit and loss tables that have income expenses and taxes, cash_flow describes the cash inflows and outflows from operating, investing and financing activities.                                          

Question: {question}

Required Documents categories: {documents_required}

Return only the query.
""")


if __name__ == "__main__":
    from langchain_community.llms import Ollama
    # Initialize LLM
    llm = Ollama(
        model="llama2:13b",
        base_url="http://localhost:11434",
        temperature=0.0
    )

    # Sample inputs
    question = "What is the company's net income?"
    context = "The income statement shows a net income of $5 million."
    answer = "The company's net income is $5 million."

    # ------------------ RAG ------------------
    rag_chain = get_rag_prompt() | llm
    print("=== RAG OUTPUT ===")
    print(rag_chain.invoke({
        "context": context,
        "question": question
    }))
    print("\n")

    # ------------------ RELEVANCY ------------------
    relevancy_chain = get_relevency_prompt() | llm
    print("=== RELEVANCY OUTPUT ===")
    print(relevancy_chain.invoke({
        "question": question,
        "answer": answer
    }))
    print("\n")

    # ------------------ DOCUMENTS REQUIRED ------------------
    doc_chain = get_documents_req_prompt() | llm
    print("=== DOCUMENTS REQUIRED OUTPUT ===")
    print(doc_chain.invoke({
        "question": question
    }))