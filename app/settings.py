import os
from dataclasses import dataclass
from dotenv import dotenv_values
from pathlib import Path

HOME = str(Path(__file__).parent)


config = dotenv_values("/Users/ndlinh.ai/Documents/002_Implementation/Chatbot/app/.env")


@dataclass
class OpenAISettings:
    api_key = config.get("OPENAI_API_KEY", "sk-xxx")
    endpoint = config.get("OPENAI_ENDPOINT", "")
    model_name = config.get("OPENAI_MODEL_NAME", "gpt-4o-mini")
    speech_model_name = config.get("OPENAI_SPEECH_MODEL_NAME", "tts-1")
    max_token = config.get("OPENAI_MAX_TOKEN", 100)


@dataclass
class YoloSettings:
    face_weight = config.get("FACE_WEIGHT", HOME + "/models/face/yolov11n-face.pt")
    embedding_weight = config.get(
        "EMBEDDING_WEIGHT", HOME + "/models/embedding/yolo11m-cls.pt"
    )


@dataclass
class MilvusSettings:
    alias = config.get("MILVUS_ALIAS", "default")
    host = config.get("MILVUS_HOST", "localhost")
    port = config.get("MILVUS_PORT", 19530)
    collection_config = config.get(
        "MILVUS_COLLECTION_CONFIG", "collection_configs.yaml"
    )


@dataclass
class PostgresSettings:
    user = config.get("POSTGRES_USER", "postgresql")
    password = config.get("POSTGRES_PASSWORD", "postgresql@123")
    host = config.get("POSTGRES_HOST", "localhost")
    port = config.get("POSTGRES_PORT", 5432)
    database = config.get("POSTGRES_DATABASE", "postgres")
    uri = f"postgresql://{user}:{password}@{host}:{port}/{database}"


class Settings:
    openai = OpenAISettings()
    yolo = YoloSettings()
    milvus = MilvusSettings()
