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

    Is the answer relevant to the question? Answer "Yes" or "No" and explain why.
    """)
def get_documents_req_prompt() -> ChatPromptTemplate:
    """Get the document requirement prompt template"""
    return ChatPromptTemplate.from_template("""
    You are a helpful assistant. Determine which documents are required to answer the question.

    Question: {question}

    List the required documents from the following options: [Document A, Document B, Document C]
    """)