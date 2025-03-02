from fastapi import APIRouter, Depends
from routers import thinking_router


api_router = APIRouter(prefix="/api/v1")


api_router.include_router(thinking_router, tags=["Thinking"])
