from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    api_rate_limit: int = 60  # requests per minute
    max_tokens: int = 4096
    allowed_origins: list[str] = ["*"]

    class Config:
        env_file = ".env"

settings = Settings()
