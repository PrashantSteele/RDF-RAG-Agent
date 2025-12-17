import sys
import os

# Prevent module reloading issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

libs_path = os.path.join(os.path.dirname(__file__), "libs")
if libs_path not in sys.path:
    sys.path.insert(0, libs_path)

import streamlit as st
from query_rag import get_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

# Page Config
st.set_page_config(
    page_title="RGPV RAG Agent", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Get base64 encoded logo for background
import base64
logo_path = os.path.join("image", "RGPV Logo.png")
if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        logo_base64 = base64.b64encode(img_file.read()).decode()
else:
    logo_base64 = ""

# Custom CSS with faded repeating logo background and responsive design
st.markdown(f"""
<style>
    /* Faded repeating logo background */
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('data:image/png;base64,{logo_base64}');
        background-repeat: repeat;
        background-size: 200px 200px;
        background-position: center;
        opacity: 0.04;
        z-index: 0;
        pointer-events: none;
    }}
    
    
    /* Ensure content is above background */
    .block-container {{
        position: relative;
        z-index: 1;
        padding-top: 3rem;
        padding-bottom: 8rem;
        max-width: 900px;
        margin: 0 auto;
    }}
    
    /* Responsive padding */
    @media (max-width: 768px) {{
        .block-container {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}
    }}
    
    /* Chat messages styling with better padding */
    .stChatMessage {{
        background-color: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
    }}
    
    .stChatMessage p {{
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }}
    
    /* Make chat input bigger - 3 lines height with proper padding */
    .stChatInput {{
        padding: 1rem 0;
        margin-bottom: 1rem;
    }}
    
    .stChatInput > div {{
        border-radius: 16px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}
    
    .stChatInput textarea {{
        min-height: 80px !important;
        height: 80px !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
        padding: 16px 20px !important;
        border-radius: 16px !important;
        border: 2px solid #e0e0e0 !important;
        transition: all 0.2s ease !important;
    }}
    
    .stChatInput textarea:focus {{
        border-color: #4285f4 !important;
        box-shadow: 0 0 0 3px rgba(66, 133, 244, 0.1) !important;
        outline: none !important;
    }}
    
    /* Chat input button styling */
    .stChatInput button {{
        padding: 8px 16px !important;
        border-radius: 12px !important;
        margin-left: 8px !important;
    }}
    
    /* Responsive chat input */
    @media (max-width: 768px) {{
        .stChatInput textarea {{
            min-height: 70px !important;
            height: 70px !important;
            font-size: 14px !important;
            padding: 14px 16px !important;
        }}
    }}
    
    /* Header styling */
    h1 {{
        font-size: 2.5rem;
        font-weight: 700;
    }}
    
    @media (max-width: 768px) {{
        h1 {{
            font-size: 1.8rem;
        }}
    }}
    
    /* Logo container responsive */
    .logo-container {{
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }}
    
    @media (max-width: 768px) {{
        .logo-container img {{
            width: 60px !important;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# Header with Logo
col1, col2 = st.columns([1, 8])
with col1:
    logo_path = os.path.join("image", "RGPV Logo.png")
    if os.path.exists(logo_path):
        st.image(logo_path, width=80)
with col2:
    st.title("RGPV RDF Guidelines Bot")
    st.caption("Ask questions about RDF Guidelines 2019")

# Initialize Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Initialize RAG chain once per session
if st.session_state.rag_chain is None:
    with st.spinner("Initializing RAG Agent..."):
        try:
            st.session_state.rag_chain = get_rag_chain()
        except Exception as e:
            st.error(f"Failed to initialize RAG chain: {e}")
            st.stop()

# Display Chat History
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Chat Input with bigger size (3 lines)
if prompt := st.chat_input("Ask a question about Ordinance 11..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.chat_history.append(AIMessage(content=answer))
            except Exception as e:
                st.error(f"An error occurred: {e}")

