from langchain_core.prompts import ChatPromptTemplate

def get_rag_prompt() -> ChatPromptTemplate:
    """Get the RAG prompt template"""
    return ChatPromptTemplate.from_template("""
You are FinLens, a financial document Q&A assistant.

Use ONLY the provided context to answer the question.
Rules:
1) Do not invent values, dates, entities, or calculations not present in context.
2) If context is insufficient, state that clearly and say what is missing.
3) Keep the answer concise but complete.
4) If useful, present key figures as short bullet points.
5) If multiple periods or entities appear, specify which one your answer refers to.

Context:
{context}

Question:
{question}

Return:
- A direct answer grounded in context.
- If uncertain, begin with: "I cannot determine this from the provided context."
""")

def get_relevency_prompt() -> ChatPromptTemplate:
    """Get the relevancy check prompt template"""
    return ChatPromptTemplate.from_template("""
You are a strict grader for answer relevance.

Determine if the answer directly addresses the user's question.
Return "No" if the answer is off-topic, generic, vague, or does not answer the asked metric/fact.
Return "Yes" only if the answer is materially responsive.

Question:
{question}

Answer:
{answer}

Output exactly one token:
Yes
or
No
""")

def get_documents_req_prompt() -> ChatPromptTemplate:
    """Get the document requirement prompt template"""
    return ChatPromptTemplate.from_template("""
You are a financial statement router.

Given a question, select which statement categories are required.
Allowed categories only:
- "balance_sheet"
- "income_statement"
- "cash_flow"

Selection guidance:
- balance_sheet: assets, liabilities, equity, debt, cash balance, working capital, solvency.
- income_statement: revenue, expenses, margins, EBITDA, earnings, profit.
- cash_flow: operating/investing/financing flows, capex, free cash flow, dividends, borrowings.
- Include multiple categories if the question spans them.
- If uncertain, choose the smallest sufficient set.

Question:
{question}

Output format instructions:
{format_instructions}

Return valid JSON only and no extra text.
""")

def get_query_construction_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_template("""
You are a retrieval query generator for a financial vector database.

Construct one compact keyword query using:
- user question
- required document categories

STRICT RULES:
- Output one line only.
- Output query text only (no labels, no explanation, no quotes).
- Use short noun phrases and financial terms.
- Include synonyms/near terms to improve retrieval recall.
- Include category-specific terms based on required documents.
- Avoid conversational wording and punctuation-heavy text.

Category hint terms:
- balance_sheet -> assets liabilities equity current assets current liabilities debt cash receivables inventory
- income_statement -> revenue sales cogs gross profit operating expenses ebit ebitda net income margin tax
- cash_flow -> operating cash flow investing cash flow financing cash flow capex free cash flow dividends borrowings

Question: {question}

Required document categories: {documents_required}

Return the final query only.
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