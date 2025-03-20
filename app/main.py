import asyncio
import logging
from fastapi import (
    FastAPI,
    Request,
    HTTPException,
    Depends,
    File,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langchain_core.messages import HumanMessage
import tempfile
import os
import base64

from app.services.graph import AssistantGraph
from app.services.graph.audio_graph import AssistantAudioGraph
from app.utils.llm import openai_llm
from app.schemas.chat import ChatRequest, ErrorResponse, AudioRequest
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress unnecessary logs
logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("openai").setLevel(logging.CRITICAL)
logging.getLogger("primp").setLevel(logging.CRITICAL)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(title="Chat API")

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize LangChain assistant graph
graph = AssistantGraph(llm=openai_llm)
print(graph._visualize())

print("-" * 50)

# Initialize LangChain audio graph
audio_graph = AssistantAudioGraph(llm=openai_llm)
print(audio_graph._visualize())


async def generate_response(user_question: str):
    """Async generator that streams AI responses"""
    try:
        inputs = {
            "user_question": user_question,
            "messages": [HumanMessage(content=user_question)],
        }

        async for response in graph.astream_with_checkpointer(inputs, thread_id="test"):
            yield response
            await asyncio.sleep(0.02)
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        yield f"Error: {str(e)}"


@app.post("/chat", response_model=None)
@limiter.limit(f"{settings.api_rate_limit}/minute")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """HTTP Streaming endpoint for AI chat"""
    try:
        logger.info(f"Received chat request: {chat_request.user_question}")
        return StreamingResponse(
            generate_response(chat_request.user_question), media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/audio", response_model=None)
@limiter.limit(f"{settings.api_rate_limit}/minute")
def audio_endpoint(request: Request, item: AudioRequest):
    """Endpoint to handle audio input from UI"""
    logger.info(f"Received audio file: {item.audio_file_path}")

    # Response assistant audio
    inputs = {"temp_audio_path": item.audio_file_path}
    response = audio_graph.aai_caller(inputs, thread_id="test")

    # Change to string
    response["audio_path"] = str(response["audio_path"])
    # Clean up the temporary file
    # os.unlink(temp_file_path)

    # Return the path to the processed audio file
    return {
        "status": "success",
        "assistant_response": response.get("assistant_response", ""),
        "audio_path": response.get("audio_path", ""),
    }


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """WebSocket endpoint to handle audio input from UI"""
    await websocket.accept()
    try:
        # Receive audio data as bytes
        audio_data = await websocket.receive_bytes()
        logger.info("Received audio data via WebSocket")

        # Create a temporary file to save the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        # Process the audio
        try:
            # Call the audio processing service
            response = audio_graph.aai_caller(temp_file_path)

            # Include transcription in response
            result = {
                "status": "success",
                "assistant_response": response.get("assistant_response", ""),
                "audio_path": response.get("audio_path", ""),
            }

            # Clean up temporary file
            # os.unlink(temp_file_path)

            # Send the response back to the client
            await websocket.send_json(result)

        except Exception as e:
            logger.error(f"Audio processing error: {str(e)}")
            await websocket.send_json({"status": "error", "error": str(e)})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        try:
            await websocket.send_json({"status": "error", "error": str(e)})
        except:
            pass


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return ErrorResponse(error=str(exc))
