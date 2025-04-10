from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "llama2"
    CHUNK_SIZE: int = 250
    CHUNK_OVERLAP: int = 50
    VECTOR_DB_PATH: str = "data/vector_store"
    DOCS_PATH: str = "data/stripe_docs"
    
    class Config:
        env_file = ".env"

settings = Settings()

