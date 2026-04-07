from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt() -> ChatPromptTemplate:
    """Get the RAG prompt template"""
    return ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the following context to answer the question.
    If you don't know the answer based on the context, say so.

    Context: {context}

    Question: {question}

    Answer:""")
