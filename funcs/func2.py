def vector_store_init():
    vector_store = None # Placeholder for vector store initialization logic
    splits = {
        "Balance Sheet": "", #Summary of Balance Sheet, 
        "Income Statement": "", #Summary of Income Statement
        "Cash Flow Statement": "" #Summary of Cash Flow Statement
    } # Placeholder for returning the 3 splits need it in a dict for the ui summary part
    return vector_store, splits