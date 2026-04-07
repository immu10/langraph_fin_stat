import streamlit as st
import json
import os
from langchain_community.vectorstores import Chroma
from funcs.func import split_summary
import main  # Import your main module

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinLens",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Inject CSS + floating chat overlay ────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f;
    color: #e8e6e0;
    font-family: 'DM Mono', monospace;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,200,80,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,140,50,0.06) 0%, transparent 55%),
        #0a0a0f;
}

[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { display: none; }
footer { display: none; }
#MainMenu { display: none; }

/* ── Typography ── */
h1, h2, h3, h4 { font-family: 'Syne', sans-serif; }

/* ── Main container ── */
.main-wrap {
    max-width: 1200px;
    margin: 0 auto;
    padding: 48px 32px 120px;
}

/* ── Header ── */
.site-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin-bottom: 56px;
}
.site-logo {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f5c842;
    letter-spacing: -0.04em;
}
.site-tag {
    font-size: 0.72rem;
    color: #555;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding-bottom: 4px;
}

/* ── Upload zone ── */
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 10px;
}

[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(245,200,66,0.25) !important;
    border-radius: 12px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(245,200,66,0.5) !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
}

/* ── Process button ── */
.stButton > button {
    background: #f5c842 !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 12px 32px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    box-shadow: 0 0 24px rgba(245,200,66,0.2) !important;
}
.stButton > button:hover {
    background: #ffd84d !important;
    box-shadow: 0 0 36px rgba(245,200,66,0.35) !important;
    transform: translateY(-1px) !important;
}

/* ── Divider ── */
.section-divider {
    display: flex;
    align-items: center;
    gap: 16px;
    margin: 48px 0 32px;
}
.section-divider-line {
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}
.section-divider-label {
    font-size: 0.7rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #444;
}

/* ── Summary cards ── */
.summary-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 28px 24px;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.summary-card:hover {
    border-color: rgba(245,200,66,0.2);
    transform: translateY(-2px);
}
.summary-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 16px 16px 0 0;
}
.card-bs::before  { background: linear-gradient(90deg, #f5c842, transparent); }
.card-cf::before  { background: linear-gradient(90deg, #f07030, transparent); }
.card-pnl::before { background: linear-gradient(90deg, #50c8a0, transparent); }

.card-icon {
    font-size: 1.4rem;
    margin-bottom: 14px;
    display: block;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.card-bs  .card-title { color: #f5c842; }
.card-cf  .card-title { color: #f07030; }
.card-pnl .card-title { color: #50c8a0; }

.card-body {
    font-size: 0.82rem;
    line-height: 1.7;
    color: #aaa;
}
.card-empty {
    font-size: 0.78rem;
    color: #3a3a3a;
    font-style: italic;
}

/* ── Spinner override ── */
[data-testid="stSpinner"] { color: #f5c842 !important; }

/* ── Chat fab ── */
#chat-fab {
    position: fixed;
    bottom: 32px;
    right: 32px;
    width: 56px;
    height: 56px;
    background: #f5c842;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 4px 24px rgba(245,200,66,0.4);
    z-index: 999;
    transition: all 0.2s;
    border: none;
    font-size: 1.4rem;
}
#chat-fab:hover {
    transform: scale(1.08);
    box-shadow: 0 6px 32px rgba(245,200,66,0.55);
}

/* ── Chat overlay ── */
#chat-overlay {
    position: fixed;
    bottom: 100px;
    right: 32px;
    width: 380px;
    height: 520px;
    background: #111116;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    box-shadow: 0 24px 64px rgba(0,0,0,0.6);
    z-index: 998;
    display: none;
    flex-direction: column;
    overflow: hidden;
}
#chat-overlay.open { display: flex; }

.chat-header {
    padding: 18px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.chat-header-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.9rem;
    font-weight: 700;
    color: #f5c842;
}
.chat-header-sub {
    font-size: 0.68rem;
    color: #444;
    margin-top: 2px;
}
.chat-close {
    background: none;
    border: none;
    color: #555;
    font-size: 1.1rem;
    cursor: pointer;
    transition: color 0.2s;
    line-height: 1;
}
.chat-close:hover { color: #e8e6e0; }

.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    scrollbar-width: thin;
    scrollbar-color: #222 transparent;
}

.msg {
    max-width: 85%;
    padding: 10px 14px;
    border-radius: 12px;
    font-size: 0.8rem;
    line-height: 1.5;
}
.msg-user {
    background: rgba(245,200,66,0.12);
    border: 1px solid rgba(245,200,66,0.2);
    color: #e8e6e0;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
}
.msg-bot {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    color: #aaa;
    align-self: flex-start;
    border-bottom-left-radius: 4px;
}

.chat-input-wrap {
    padding: 12px 16px;
    border-top: 1px solid rgba(255,255,255,0.06);
    display: flex;
    gap: 8px;
}
#chat-input {
    flex: 1;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 8px;
    padding: 10px 14px;
    color: #e8e6e0;
    font-family: 'DM Mono', monospace;
    font-size: 0.78rem;
    outline: none;
    transition: border-color 0.2s;
}
#chat-input:focus { border-color: rgba(245,200,66,0.3); }
#chat-input::placeholder { color: #3a3a3a; }
#chat-send {
    background: #f5c842;
    border: none;
    border-radius: 8px;
    width: 38px;
    height: 38px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 1rem;
    transition: all 0.2s;
    flex-shrink: 0;
}
#chat-send:hover { background: #ffd84d; }

.chat-empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    color: #333;
    font-size: 0.75rem;
    text-align: center;
    padding: 20px;
}
.chat-empty-state span { font-size: 2rem; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #222; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "summaries" not in st.session_state:
    st.session_state.summaries = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processing" not in st.session_state:
    st.session_state.processing = False
if "chat_input_counter" not in st.session_state:
    st.session_state.chat_input_counter = 0

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-wrap">
  <div class="site-header">
    <span class="site-logo">FinLens</span>
    <span class="site-tag">Financial Document Analyzer</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Upload section ─────────────────────────────────────────────────────────────
st.markdown('<div class="upload-label">Upload Financial Document</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="upload",
    type=["pdf"],
    label_visibility="collapsed",
    help="Upload your financial report PDF"
)

col_btn, col_status = st.columns([1, 4])
with col_btn:
    process_clicked = st.button("Analyze →", disabled=uploaded_file is None)

if process_clicked and uploaded_file:
    with st.spinner("Processing document..."):
        summaries = split_summary(uploaded_file)
        # Reload the vector store after processing the document
        main.vector_store = Chroma(persist_directory="./chroma_db")
        st.session_state.summaries = summaries

# ── Summary cards ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-divider">
  <div class="section-divider-line"></div>
  <div class="section-divider-label">Section Summaries</div>
  <div class="section-divider-line"></div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

def render_card(col, css_class, icon, title, key):
    with col:
        summaries = st.session_state.summaries
        if summaries and key in summaries:
            body = f'<div class="card-body">{summaries[key]}</div>'
        else:
            body = '<div class="card-empty">Upload and analyze a document to see the summary.</div>'

        st.markdown(f"""
        <div class="summary-card {css_class}">
            <span class="card-icon">{icon}</span>
            <div class="card-title">{title}</div>
            {body}
        </div>
        """, unsafe_allow_html=True)

render_card(c1, "card-bs",  "🏦", "Balance Sheet",    "balance_sheet")
render_card(c2, "card-cf",  "💸", "Cash Flow",         "cash_flow")
render_card(c3, "card-pnl", "📈", "P&L / Income",      "income_statement")

# ── Chat functionality ─────────────────────────────────────────────────────────
# Hidden text input to receive messages from JavaScript
chat_input_key = f"chat_input_{st.session_state.chat_input_counter}"
user_message = st.text_input("chat_receiver", key=chat_input_key, label_visibility="collapsed")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process new message
if user_message and user_message.strip():
    # Add user message to history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_message
    })
    
    # Get bot response from RAG
    try:
        bot_response = main.rag_flow(question=user_message)
        st.session_state.chat_history.append({
            "role": "bot",
            "content": bot_response
        })
    except Exception as e:
        st.session_state.chat_history.append({
            "role": "bot",
            "content": f"Error: {str(e)}"
        })
    
    # Increment counter to reset input
    st.session_state.chat_input_counter += 1
    st.rerun()

# ── Chat overlay + FAB ────────────────────────────────────────────────────────
def build_messages_html():
    if not st.session_state.chat_history:
        return """
        <div class="chat-empty-state">
            <span>💬</span>
            <div>Ask anything about the<br>uploaded financial document</div>
        </div>
        """
    html = ""
    for msg in st.session_state.chat_history:
        css = "msg-user" if msg["role"] == "user" else "msg-bot"
        content = msg["content"].replace("\n", "<br>")  # Preserve line breaks
        html += f'<div class="msg {css}">{content}</div>'
    return html

messages_html = build_messages_html()

st.markdown(f"""
<!-- Floating action button -->
<button id="chat-fab" onclick="toggleChat()" title="Ask a question">💬</button>

<!-- Chat overlay -->
<div id="chat-overlay">
  <div class="chat-header">
    <div>
      <div class="chat-header-title">Ask FinLens</div>
      <div class="chat-header-sub">RAG-powered document Q&A</div>
    </div>
    <button class="chat-close" onclick="toggleChat()">✕</button>
  </div>

  <div class="chat-messages" id="chat-messages">
    {messages_html}
  </div>

  <div class="chat-input-wrap">
    <input id="chat-input" type="text" placeholder="Ask about the financials..." 
           onkeydown="if(event.key==='Enter') sendMessage()" />
    <button id="chat-send" onclick="sendMessage()">↑</button>
  </div>
</div>

<script>
const stTextInput = window.parent.document.querySelector('input[aria-label="chat_receiver"]');

function toggleChat() {{
    const overlay = document.getElementById('chat-overlay');
    overlay.classList.toggle('open');
    
    // Auto-scroll to bottom when opening
    if (overlay.classList.contains('open')) {{
        setTimeout(() => {{
            const messages = document.getElementById('chat-messages');
            messages.scrollTop = messages.scrollHeight;
        }}, 100);
    }}
}}

function sendMessage() {{
    const input = document.getElementById('chat-input');
    const msg = input.value.trim();
    if (!msg) return;

    // Send message to Streamlit via hidden text input
    if (stTextInput) {{
        stTextInput.value = msg;
        stTextInput.dispatchEvent(new Event('input', {{ bubbles: true }}));
        stTextInput.dispatchEvent(new Event('change', {{ bubbles: true }}));
        
        // Trigger Streamlit's enter key event
        const enterEvent = new KeyboardEvent('keydown', {{
            key: 'Enter',
            code: 'Enter',
            keyCode: 13,
            bubbles: true
        }});
        stTextInput.dispatchEvent(enterEvent);
    }}

    // Clear input
    input.value = '';
}}

// Auto-scroll to bottom on load
window.addEventListener('load', () => {{
    const messages = document.getElementById('chat-messages');
    messages.scrollTop = messages.scrollHeight;
}});
</script>
""", unsafe_allow_html=True)