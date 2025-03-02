import asyncio
import logging
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from langchain_core.messages import HumanMessage

from app.services.graph import AssistantGraph
from app.utils.llm import openai_llm
from app.schemas.chat import ChatRequest, ErrorResponse
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
async def chat_endpoint(
    request: Request,
    chat_request: ChatRequest
):
    """HTTP Streaming endpoint for AI chat"""
    try:
        logger.info(f"Received chat request: {chat_request.user_question}")
        return StreamingResponse(
            generate_response(chat_request.user_question),
            media_type="text/plain"
        )
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return ErrorResponse(error=str(exc))
