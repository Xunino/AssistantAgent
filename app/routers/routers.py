import base64
from fastapi import APIRouter, UploadFile, File
from typing import Optional

from services import ThinkingFunctions

thinking_router = APIRouter()


thinking_functions = ThinkingFunctions()


@thinking_router.post("/chat_completion")
async def chat_with_brain(
    user_question: Optional[str] = None, uploaded_image: UploadFile = File(...)
):

    content = await uploaded_image.read()

    # Encode the content to Base64
    encoded_content = base64.b64encode(content).decode("utf-8")

    # Call the chat_completion_with_image method
    response = thinking_functions.chat_completion_with_image(
        user_question, encoded_content
    )
    return response
