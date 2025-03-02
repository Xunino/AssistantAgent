import streamlit as st
import requests
import datetime
import json
import asyncio
import hashlib
from typing import Dict, Any
import bleach

API_URL = st.secrets["API_URL"]

class SessionState:
    def __init__(self):
        self.messages = []
        self.error_count = 0
        self.theme = "light"
        self.last_request_time = None
        self.request_count = 0

def init_session() -> SessionState:
    if "session" not in st.session_state:
        st.session_state.session = SessionState()
    return st.session_state.session

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent XSS"""
    return bleach.clean(text, tags=[], strip=True)

def rate_limit_check() -> bool:
    """Simple rate limiting"""
    now = datetime.datetime.now()
    session = st.session_state.session
    
    if session.last_request_time:
        if (now - session.last_request_time).seconds < 1:
            session.request_count += 1
            if session.request_count > 5:  # Max 5 requests per second
                return False
        else:
            session.request_count = 0
    
    session.last_request_time = now
    return True

# Initialize session and configuration
session = init_session()
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/chatbot',
        'Report a bug': "https://github.com/yourusername/chatbot/issues",
        'About': "# AI Chatbot v1.0.0\nPowered by OpenAI and LangChain"
    }
)

# Enhanced sidebar with more features
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Theme selection
    theme = st.select_slider(
        "🎨 Theme",
        options=["light", "dark"],
        value=session.theme
    )
    if theme != session.theme:
        session.theme = theme
        st.experimental_set_query_params(theme=theme)
    
    # Chat controls section
    st.subheader("💬 Chat Controls")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            session.messages = []
            st.rerun()
    
    with col2:
        if st.button("📥 Export"):
            chat_export = {
                "messages": session.messages,
                "exported_at": datetime.datetime.now().isoformat(),
                "metadata": {
                    "version": "1.0.0",
                    "message_count": len(session.messages)
                }
            }
            export_data = json.dumps(chat_export, indent=2)
            
            # Generate secure filename with timestamp and hash
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            content_hash = hashlib.md5(export_data.encode()).hexdigest()[:8]
            filename = f"chat_history_{timestamp}_{content_hash}.json"
            
            st.download_button(
                "Download JSON",
                data=export_data,
                file_name=filename,
                mime="application/json"
            )
    
    # About section
    st.markdown("---")
    st.markdown("""
    ### About
    
    🤖 **AI Chatbot v1.0.0**
    
    Built with:
    - OpenAI
    - LangChain
    - Streamlit
    - FastAPI
    
    [Documentation](https://github.com/yourusername/chatbot)
    [Report Issues](https://github.com/yourusername/chatbot/issues)
    
    © 2024 Your Company
    """)

# Main chat interface
st.title("💬 AI Chatbot")

# Display chat history with enhanced formatting
for message in session.messages:
    with st.chat_message(message["role"]):
        timestamp = message.get("timestamp", "")
        content = message["content"]
        st.markdown(f"""
        <div class="message-container">
            <div class="timestamp">{timestamp}</div>
            <div class="content">{content}</div>
        </div>
        """, unsafe_allow_html=True)

async def stream_response(response, placeholder):
    """Optimized streaming with backoff"""
    try:
        buffer = ""
        async for chunk in response.content:
            if not chunk:
                continue
            buffer += chunk.decode()
            if len(buffer) >= 100 or not response.content.can_read():
                placeholder.markdown(buffer)
                buffer = ""
            await asyncio.sleep(0.02)
        if buffer:
            placeholder.markdown(buffer)
    except Exception as e:
        st.error(f"Streaming error: {str(e)}")

def chat_with_backend(user_message: str):
    """Enhanced chat interaction with security and performance improvements"""
    if not rate_limit_check():
        st.error("Too many requests. Please wait a moment.")
        return
    
    try:
        # Sanitize input
        clean_message = sanitize_input(user_message)
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        
        # Display user message with timestamp
        with st.chat_message("user"):
            st.write(f"[{timestamp}] {clean_message}")
        
        # Add to history with timestamp
        session.messages.append({
            "role": "user",
            "content": clean_message,
            "timestamp": timestamp
        })
        
        # Get response from backend
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            with st.spinner("🤔 AI is thinking..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={"user_question": clean_message},
                        stream=True,
                        timeout=30  # 30 seconds timeout
                    )
                    
                    if response.status_code == 200:
                        bot_response = ""
                        for chunk in response.iter_content(chunk_size=512):
                            if chunk:
                                bot_response += chunk.decode("utf-8")
                                response_placeholder.write(f"[{timestamp}] {bot_response}")
                        
                        # Add to history with timestamp
                        session.messages.append({
                            "role": "assistant",
                            "content": bot_response,
                            "timestamp": timestamp
                        })
                        
                        # Limit chat history to 50 messages
                        if len(session.messages) > 50:
                            session.messages = session.messages[-50:]
                            st.info("💡 Chat history has been trimmed to last 50 messages")
                    else:
                        error_msg = f"Server Error (HTTP {response.status_code})"
                        st.error(error_msg)
                        session.error_count += 1
                
                except requests.exceptions.Timeout:
                    st.error("⚠️ Request timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Connection failed. Is the server running?")
                    
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
        session.error_count += 1
        
        if session.error_count >= 3:
            st.warning("👉 Tip: If errors persist, try clearing the chat history")

# Secure chat input with validation
try:
    user_input = st.chat_input("Type your message...")
    if user_input:
        if len(user_input.strip()) > 0:
            chat_with_backend(user_input)
        else:
            st.warning("Please enter a valid message")
except Exception as e:
    st.error("An error occurred. Please try again.")

# Add custom CSS for better styling
st.markdown("""
<style>
    .message-container {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
    }
    .timestamp {
        font-size: 0.8em;
        color: #666;
        margin-bottom: 5px;
    }
    .content {
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)
