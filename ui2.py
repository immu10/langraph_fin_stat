import streamlit as st

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Chat UI", layout="wide")

if "chat_input_counter" not in st.session_state:
    st.session_state.chat_input_counter = 0

chat_input_key = f"chat_input_{st.session_state.chat_input_counter}"
user_message = st.text_input("chat_receiver", key=chat_input_key, label_visibility="collapsed")

# -------------------------------
# Initialize session state
# -------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Function: Get bot response
# Replace this with your RAG pipeline
# -------------------------------
def get_bot_response(user_query):
    # Example placeholder
    # Replace with: return main.rag_flow(question=user_query)
    return f"Echo: {user_query}"

# -------------------------------
# Render chat history
# -------------------------------
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------------
# Chat input
# -------------------------------
user_input = st.chat_input("Type your message...")

if user_input:
    # 1. Append user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })

    # 2. Get bot response
    try:
        bot_response = get_bot_response(user_input)
    except Exception as e:
        bot_response = f"Error: {str(e)}"

    # 3. Append bot response
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": bot_response
    })

    # 4. Rerun to update UI
    st.rerun()