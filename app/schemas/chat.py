from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    user_question: str = Field(..., min_length=1, max_length=4096)

class ErrorResponse(BaseModel):
    error: str
