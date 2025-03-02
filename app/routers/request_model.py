from pydantic import BaseModel
from typing import Optional, Iterator
from fastapi import UploadFile, File


class ChatRequestModel(BaseModel):
    user_question: Optional[str] = None
    uploaded_image: UploadFile = File(None)
