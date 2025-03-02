from app.settings import OpenAISettings
from langchain_openai import ChatOpenAI

openai_llm = ChatOpenAI(
    api_key=OpenAISettings.api_key,
    model=OpenAISettings.model_name,
    temperature=0.0,
)
