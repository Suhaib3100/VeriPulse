"""Backend configuration."""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    debug: bool = True
    ws_port: int = 8000

settings = Settings()
