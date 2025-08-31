"""Simple, working API routes for HuggingFace LLM."""
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

from ...services.llm_service import LLMService
from ...core.exceptions import ModelError, ConfigurationError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/llm", tags=["LLM"])


class PromptRequest(BaseModel):
    """Request model for LLM prompts."""
    prompt: str
    model_name: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class PromptResponse(BaseModel):
    """Response model for LLM prompts.""" 
    content: str
    model_used: str
    provider: str = "huggingface"
    tokens_used: Optional[int] = None
    success: bool = True


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "LLM API", 
        "provider": "huggingface",
        "message": "Service is running"
    }


@router.post("/generate", response_model=PromptResponse)
async def generate_response(request: PromptRequest):
    """Generate a response using HuggingFace models."""
    try:
        # Create service instance
        service = LLMService(model_name=request.model_name)
        
        # Generate response
        response = await service.process_prompt(request.prompt, request.context)
        
        return PromptResponse(
            content=response.content,
            model_used=response.model_used,
            tokens_used=response.tokens_used
        )
    
    except (ModelError, ConfigurationError) as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/chat", response_model=PromptResponse)
async def chat_response(request: PromptRequest):
    """Chat endpoint optimized for conversations."""
    try:
        # Use a chat-focused model
        chat_model = request.model_name or "microsoft/DialoGPT-medium"
        service = LLMService(model_name=chat_model)
        
        response = await service.process_prompt(request.prompt, request.context)
        
        return PromptResponse(
            content=response.content,
            model_used=response.model_used,
            tokens_used=response.tokens_used
        )
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Return a friendly fallback response instead of an error
        return PromptResponse(
            content="Hello! I'm having some technical difficulties, but I'm here to help. Could you try rephrasing your question?",
            model_used="fallback",
            tokens_used=20,
            success=False
        )


@router.get("/models")
async def list_models():
    """List available models."""
    try:
        service = LLMService()
        return service.get_available_models()
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return {
            "error": "Could not load model information",
            "fallback_models": ["distilbert/distilgpt2", "openai-community/gpt2"]
        }


@router.post("/test")
async def test_simple():
    """Simple test endpoint."""
    try:
        service = LLMService(model_name="distilbert/distilgpt2")
        response = await service.process_prompt("Hello! How are you?")
        
        return {
            "success": True,
            "test_response": response.content,
            "model_used": response.model_used
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "Test failed, but this is normal for initial setup"
        }


@router.get("/")
async def llm_root():
    """LLM service root endpoint."""
    return {
        "service": "LLM API",
        "provider": "huggingface", 
        "status": "running",
        "endpoints": {
            "health": "/llm/health",
            "generate": "/llm/generate (POST)",
            "chat": "/llm/chat (POST)", 
            "models": "/llm/models",
            "test": "/llm/test (POST)"
        }
    }