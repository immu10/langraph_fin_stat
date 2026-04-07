import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import embeddings,llm
from funcs.func2 import vector_store_init






def split_summary(uploaded_file):
    """Create detailed summary of each financial statement section."""
    
    # Financial statement context for better prompts
    STATEMENT_CONTEXTS = {
        "balance_sheet": {
            "focus": "assets, liabilities, equity, liquidity ratios, and overall financial position",
            "key_metrics": "current ratio, debt-to-equity, working capital"
        },
        "cash_flow": {
            "focus": "operating activities, investing activities, financing activities, and cash generation ability",
            "key_metrics": "free cash flow, cash conversion, burn rate"
        },
        "income_statement": {
            "focus": "revenue, expenses, profitability, margins, and operational efficiency",
            "key_metrics": "gross margin, operating margin, net income"
        }
    }
    
    split = vector_store_init(uploaded_file)
    summaries = {}
    
    for key, doc in split.items():
        # Normalize key name (handle variations)
        normalized_key = key.lower().replace(" ", "_")
        
        # Get context if available, otherwise use generic
        context = STATEMENT_CONTEXTS.get(normalized_key, {
            "focus": "key financial metrics and trends",
            "key_metrics": "notable figures and changes"
        })
        
        # Construct detailed prompt
        prompt = f"""You are a financial analyst reviewing a {key.replace('_', ' ').title()}.

Analyze the following financial data and provide a concise summary that:
1. Highlights the most significant figures and trends (focus on: {context['focus']})
2. Identifies any notable changes or red flags
3. Mentions key metrics: {context.get('key_metrics', 'important ratios and indicators')}

Keep your summary to 3-4 sentences, prioritizing actionable insights.

Financial Data:
{doc}

Summary:"""
        
        try:
            summary = llm.invoke(prompt)
            summaries[key] = summary.strip()
            print(f"\n{'='*60}")
            print(f"{key.upper()} SUMMARY:")
            print(f"{'='*60}")
            print(f"{summary}\n")
        except Exception as e:
            print(f"Error summarizing {key}: {str(e)}")
            summaries[key] = f"Error generating summary: {str(e)}"
    
    return summaries
