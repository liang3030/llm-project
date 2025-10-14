"""Configuration management for the LLM project with HuggingFace support."""
import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    app_name: str = Field(default="My LLM Project", alias="APP_NAME")
    debug: bool = Field(default=False, alias="DEBUG")
    version: str = Field(default="0.1.0", alias="VERSION")
    
    # LLM Configuration - Default to HuggingFace
    model_provider: str = Field(default="huggingface", alias="MODEL_PROVIDER")
    model_name: str = Field(default="microsoft/DialoGPT-medium", alias="MODEL_NAME")
    max_tokens: int = Field(default=512, alias="MAX_TOKENS")  # HF models often have lower limits
    temperature: float = Field(default=0.7, alias="TEMPERATURE")
    
    # HuggingFace Configuration
    huggingface_api_token: Optional[str] = Field(default=None, alias="HUGGINGFACE_API_TOKEN")
    huggingface_inference_api_url: str = Field(
        default="https://api-inference.huggingface.co/models/", 
        alias="HUGGINGFACE_INFERENCE_API_URL"
    )
    huggingface_task: str = Field(default="text-generation", alias="HUGGINGFACE_TASK")
    
    # Fallback configurations for other providers
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore",  # Allow extra fields to be ignored
    }


# Global settings instance
settings = Settings()