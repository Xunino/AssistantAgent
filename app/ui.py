import streamlit as st
import requests
import datetime
import json
import asyncio
import hashlib
from typing import Dict, Any
import bleach
import os
from loguru import logger
import time

API_URL = st.secrets["API_URL"]
AUDIO_URL = st.secrets["AUDIO_URL"]


class BaseSessionState:
    """Base session state with common attributes"""

    def __init__(self):
        self.messages = []
        self.error_count = 0
        self.theme = "light"
        self.last_request_time = None
        self.request_count = 0


class TextSessionState(BaseSessionState):
    """Session state specific to text-based chat"""

    def __init__(self):
        super().__init__()


class AudioSessionState(BaseSessionState):
    """Session state specific to audio-based chat"""

    def __init__(self):
        super().__init__()
        self.recording_enabled = True
        self.active_input = (
            None  # Track which input is active: 'text', 'audio', or None
        )
        self.recording_processed = False  # Flag to track if recording was processed
        self.recording_key = 0  # Key to force re-render the audio input


def init_session_states():
    """Initialize both session states if they don't exist"""
    if "text_session" not in st.session_state:
        st.session_state.text_session = TextSessionState()

    if "audio_session" not in st.session_state:
        st.session_state.audio_session = AudioSessionState()


def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent XSS"""
    return bleach.clean(text, tags=[], strip=True)


def rate_limit_check(session) -> bool:
    """Simple rate limiting"""
    now = datetime.datetime.now()

    if session.last_request_time:
        if (now - session.last_request_time).seconds < 1:
            session.request_count += 1
            if session.request_count > 5:  # Max 5 requests per second
                return False
        else:
            session.request_count = 0

    session.last_request_time = now
    return True


def create_sidebar():
    """Create sidebar with settings"""
    # Existing function remains the same
    with st.sidebar:
        st.title("⚙️ Settings")

        # Organize settings into tabs for better navigation
        tab1, tab2, tab3 = st.tabs(["General", "Voice", "About"])

        with tab1:
            # Theme selection with visual indicator
            st.subheader("🎨 Theme")
            theme = st.select_slider(
                "Choose theme",
                options=["light", "dark"],
                value=st.session_state.text_session.theme,
            )

            if theme != st.session_state.text_session.theme:
                st.session_state.text_session.theme = theme
                st.session_state.audio_session.theme = theme
                st.experimental_set_query_params(theme=theme)

        with tab2:
            st.subheader("🎙️ Voice Settings")
            recording_enabled = st.toggle(
                "Enable Voice Input",
                value=st.session_state.audio_session.recording_enabled,
                help="Turn on/off voice recording capability",
            )

            if recording_enabled != st.session_state.audio_session.recording_enabled:
                st.session_state.audio_session.recording_enabled = recording_enabled

            # Show voice settings only when enabled
            if st.session_state.audio_session.recording_enabled:
                max_duration = st.slider(
                    "Recording duration (sec)",
                    min_value=5,
                    max_value=60,
                    value=30,
                    help="Maximum length of voice recording in seconds",
                )
                st.session_state.max_recording_duration = max_duration

        with tab3:
            st.markdown(
                """
            ### About AI Chatbot
            
            Version 1.0.0
            
            Built with:
            - OpenAI
            - LangChain
            - Streamlit
            - FastAPI
            
            [Documentation](https://github.com/yourusername/chatbot)  
            [Report Issues](https://github.com/yourusername/chatbot/issues)
            
            © 2024 Your Company
            """
            )


def display_chat_history(session):
    """Display the chat history"""
    if session.messages:
        with st.container():
            for i, message in enumerate(session.messages):
                # Add subtle alternating background for better message separation
                is_even = i % 2 == 0
                bg_color = "rgba(0,0,0,0.03)" if is_even else "transparent"

                with st.chat_message(message["role"]):
                    timestamp = message.get("timestamp", "")

                    # Handle audio messages from user
                    if message.get("type") == "audio":
                        st.markdown(
                            f"<div class='timestamp'>{timestamp}</div>",
                            unsafe_allow_html=True,
                        )
                        st.audio(message["audio_data"])
                        if "transcription" in message:
                            st.markdown(
                                f"**Transcription:** {message['transcription']}"
                            )
                    # Handle audio responses from assistant
                    elif message.get("type") == "audio_response":
                        # Display content and audio playback for assistant
                        st.markdown(
                            f"""
                            <div class="message-container" style="background-color: {bg_color};">
                                <div class="timestamp">{timestamp}</div>
                                <div class="content">{message.get('content', '')}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        if "audio_data" in message:
                            st.audio(message["audio_data"])
                    else:
                        # Use .get() with default value to safely access content
                        content = message.get("content", "")
                        st.markdown(
                            f"""
                            <div class="message-container" style="background-color: {bg_color};">
                                <div class="timestamp">{timestamp}</div>
                                <div class="content">{content}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
    else:
        # Show welcome message when no chat history
        st.info(
            "👋 Welcome! Start the conversation by typing a message or using voice input below."
        )


def chat_with_backend(user_message: str, session):
    """Enhanced chat interaction with security and performance improvements"""
    # Existing function remains the same
    if not rate_limit_check(session):
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
        session.messages.append(
            {"role": "user", "content": clean_message, "timestamp": timestamp}
        )

        # Get response from backend
        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            with st.spinner("🤔 AI is thinking..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={"user_question": clean_message},
                        stream=True,
                        timeout=180,
                    )

                    if response.status_code == 200:
                        bot_response = ""
                        for chunk in response.iter_content(chunk_size=512):
                            if chunk:
                                bot_response += chunk.decode("utf-8")
                                response_placeholder.write(
                                    f"[{timestamp}] {bot_response}"
                                )

                        # Add to history with timestamp
                        session.messages.append(
                            {
                                "role": "assistant",
                                "content": bot_response,
                                "timestamp": timestamp,
                            }
                        )

                        # Limit chat history to 50 messages
                        if len(session.messages) > 50:
                            session.messages = session.messages[-50:]
                            st.info(
                                "💡 Chat history has been trimmed to last 50 messages"
                            )
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


def reset_audio_recorder(session):
    """Reset the audio recorder state to prepare for a new recording"""
    # Reset the audio input by incrementing the key to force re-rendering
    session.recording_key += 1
    session.recording_processed = True
    session.active_input = None


def process_audio(audio_data, session):
    """Process audio recording and get response"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    # First, display user's recording in chat history
    with st.chat_message("user"):
        st.markdown(f"<div class='timestamp'>{timestamp}</div>", unsafe_allow_html=True)
        st.audio(audio_data)

        # Add to history with timestamp (user's recording)
        user_audio_message = {
            "role": "user",
            "type": "audio",
            "audio_data": audio_data,
            "timestamp": timestamp,
        }
        session.messages.append(user_audio_message)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        st.markdown(f"<div class='timestamp'>{timestamp}</div>", unsafe_allow_html=True)

        with st.status("Processing audio...", expanded=True) as status:
            # Create a temporary file to save the audio locally
            temp_filename = f"temp_audio_{session.recording_key}.wav"
            if not os.path.exists(temp_filename):
                with open(temp_filename, "wb") as temp_file:
                    temp_file.write(audio_data.getbuffer())

            with st.spinner("Sending to server..."):
                response = requests.post(
                    f"{AUDIO_URL}",
                    json={"audio_file_path": temp_filename},
                    timeout=180,
                )

            logger.info(f"Audio processing response status: {response.json()}")

            # Process the response
            if response.status_code == 200:
                # Parse the JSON response
                response_data = response.json()
                status.update(label="Audio processed successfully!", state="complete")

                # Play audio response if available
                if "audio_path" in response_data:
                    try:
                        # Get audio data from the path
                        audio_file_path = response_data["audio_path"]
                        with open(audio_file_path, "rb") as audio_file:
                            audio_bytes = audio_file.read()

                            # First, display the standard audio player as fallback
                            st.audio(audio_bytes, format="audio/mp3")

                            # Then create an auto-playing audio element
                            import base64

                            encoded_audio = base64.b64encode(audio_bytes).decode()

                            audio_html = f"""
                            <audio autoplay>
                                <source src="data:audio/mp3;base64,{encoded_audio}" type="audio/mp3">
                                Your browser does not support the audio element.
                            </audio>
                            <script>
                                // Force play for browsers that might block autoplay
                                document.addEventListener('DOMContentLoaded', function() {{
                                    const audio = document.querySelector('audio');
                                    const playPromise = audio.play();
                                    if (playPromise !== undefined) {{
                                        playPromise.catch(error => {{
                                            console.log('Autoplay prevented. Attempting to play on user interaction.');
                                        }});
                                    }}
                                }});
                            </script>
                            """
                            st.components.v1.html(audio_html, height=0)

                    except Exception as e:
                        st.warning(f"Failed to play audio response: {str(e)}")
                        logger.error(f"Audio playback error: {str(e)}")

                # Add to history with timestamp
                if "assistant_response" in response_data:
                    # Create a chat message for the assistant response
                    assistant_message = ""
                    for chunk in response_data["assistant_response"].split(" "):
                        if chunk:
                            time.sleep(0.2)
                            assistant_message += chunk + " "
                            response_placeholder.write(assistant_message)

                    # Add to history with timestamp and include audio path
                    audio_message = {
                        "role": "assistant",
                        "content": assistant_message,
                        "timestamp": timestamp,
                    }

                    # Save audio path if available
                    if "audio_path" in response_data:
                        with open(response_data["audio_path"], "rb") as audio_file:
                            audio_bytes = audio_file.read()
                        audio_message["audio_path"] = response_data["audio_path"]
                        audio_message["audio_data"] = audio_bytes
                        audio_message["type"] = "audio_response"

                    session.messages.append(audio_message)
                else:
                    st.warning("Received response but missing assistant_response field")
                    logger.warning(f"Missing assistant_response in: {response_data}")
            else:
                error_msg = f"Error processing audio: HTTP {response.status_code}"
                st.error(error_msg)
                status.update(label=error_msg, state="error")

            try:
                os.remove(temp_filename)
            except Exception as e:
                st.warning(f"Failed to clean up temp file: {str(e)}")

            # Reset the audio recorder automatically after response
            reset_audio_recorder(session)

            # Add a small notification to show the recorder was reset
            st.success("Ready for your next voice message!", icon="🎙️")


def text_chatbot_ui():
    """Text-based chatbot interface"""
    # Existing function remains the same
    session = st.session_state.text_session

    # Chat history section
    st.subheader("Text Chat")

    # Chat controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", key="text_clear_chat"):
            session.messages = []
            st.rerun()
    with col2:
        if st.button("📥 Export Chat", key="text_export_chat"):
            chat_export = {
                "messages": session.messages,
                "exported_at": datetime.datetime.now().isoformat(),
                "metadata": {
                    "version": "1.0.0",
                    "message_count": len(session.messages),
                },
            }
            export_data = json.dumps(chat_export, indent=2)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            content_hash = hashlib.md5(export_data.encode()).hexdigest()[:8]
            filename = f"chat_history_{timestamp}_{content_hash}.json"

            st.download_button(
                "Download JSON",
                data=export_data,
                file_name=filename,
                mime="application/json",
            )

    # Display chat history
    display_chat_history(session)

    # Text input
    user_input = st.chat_input("Type your message...")
    if user_input:
        if len(user_input.strip()) > 0:
            chat_with_backend(user_input, session)
        else:
            st.warning("Please enter a valid message")


def audio_chatbot_ui():
    """Audio-based chatbot interface"""
    session = st.session_state.audio_session

    # Audio chat section
    st.subheader("Voice Chat")

    # Audio controls
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", key="audio_clear_chat"):
            session.messages = []
            # Reset audio recording state when clearing chat
            reset_audio_recorder(session)
            st.rerun()
    with col2:
        if st.button("📥 Export Chat", key="audio_export_chat"):
            chat_export = {
                "messages": session.messages,
                "exported_at": datetime.datetime.now().isoformat(),
                "metadata": {
                    "version": "1.0.0",
                    "message_count": len(session.messages),
                },
            }
            export_data = json.dumps(chat_export, indent=2)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            content_hash = hashlib.md5(export_data.encode()).hexdigest()[:8]
            filename = f"voice_chat_history_{timestamp}_{content_hash}.json"

            st.download_button(
                "Download JSON",
                data=export_data,
                file_name=filename,
                mime="application/json",
            )

    # Display chat history
    display_chat_history(session)

    # Audio recording component
    if session.recording_enabled:
        st.markdown("### 🎙️ Voice Input")

        # Create columns for a better layout
        col1, col2 = st.columns([3, 1])

        with col1:
            # If recording was just processed, show success message and reset flag
            if session.recording_processed:
                st.success("Recording processed! You can record a new message now.")
                session.recording_processed = False

            if session.active_input == "text":
                st.info("Please complete your text message before using voice input.")
                audio_disabled = True
            else:
                audio_disabled = False

            # Use a unique key based on recording_key to force re-render
            audio_key = f"audio_input_{session.recording_key}"
            audio_data = st.audio_input(
                "Record your message", key=audio_key, disabled=audio_disabled
            )

        with col2:
            # Display info about recording settings
            st.caption(f"Max duration: {st.session_state.max_recording_duration}s")

            # Add a manual reset button
            if st.button("Reset Recorder", key=f"reset_{session.recording_key}"):
                reset_audio_recorder(session)
                st.rerun()

        if audio_data is not None:
            # Set active input to audio
            session.active_input = "audio"

            # Process the recorded audio
            process_audio(audio_data, session)

    # Allow text input in voice tab too for flexibility
    if session.active_input == "audio":
        st.info("Please complete your voice recording before typing a message.")
        user_input = st.chat_input(
            "Type your message...", key="voice_tab_text_input", disabled=True
        )
    else:
        user_input = st.chat_input("Type your message...", key="voice_tab_text_input")

    if user_input:
        # Set active input to text
        session.active_input = "text"

        if len(user_input.strip()) > 0:
            chat_with_backend(user_input, session)
            # After chat processing is complete, reset active input
            session.active_input = None
        else:
            st.warning("Please enter a valid message")


def main():
    # Initialize session states
    init_session_states()

    # Initialize recording key if not exists
    if "max_recording_duration" not in st.session_state:
        st.session_state.max_recording_duration = 30

    # Page config
    st.set_page_config(
        page_title="AI Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={
            "Get Help": "https://github.com/yourusername/chatbot",
            "Report a bug": "https://github.com/yourusername/chatbot/issues",
            "About": "# AI Chatbot v1.0.0\nPowered by OpenAI and LangChain",
        },
    )

    # Create sidebar
    create_sidebar()

    # Main title
    st.title("💬 AI Chatbot")

    # Create tabs for text and voice interfaces
    text_tab, voice_tab = st.tabs(["💬 Text Chat", "🎙️ Voice Chat"])

    # Render each interface in its respective tab
    with text_tab:
        text_chatbot_ui()

    with voice_tab:
        audio_chatbot_ui()

    # Add custom CSS for better styling
    st.markdown(
        """
    <style>
        /* Message styling */
        .message-container {
            padding: 15px;
            margin: 8px 0;
            border-radius: 8px;
            transition: all 0.2s ease;
        }
        .message-container:hover {
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .timestamp {
            font-size: 0.75em;
            color: #888;
            margin-bottom: 6px;
            font-weight: 500;
        }
        .content {
            line-height: 1.6;
        }
        
        /* Input area styling */
        .stAudioInput {
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .stAudioInput:hover {
            border-color: #aaa;
        }
        
        /* Better button styling */
        .stButton button {
            border-radius: 20px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 8px 8px 0 0;
            gap: 8px;
            padding: 10px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e6f2ff;
            border-bottom: 2px solid #1f77b4;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
