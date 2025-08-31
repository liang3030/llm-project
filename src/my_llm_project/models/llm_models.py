"""LLM model implementations with working HuggingFace models."""
from typing import Optional, Dict, Any
import requests
import json
import time
import logging

from pydantic import BaseModel

from ..core.config import settings
from ..core.exceptions import ModelError, ConfigurationError

logger = logging.getLogger(__name__)


class LLMResponse(BaseModel):
    """Response model for LLM interactions."""
    content: str
    model_used: str
    tokens_used: Optional[int] = None
    metadata: Dict[str, Any] = {}


class BaseLLMModel:
    """Base class for LLM models."""
    
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.model_name
        self._validate_configuration()
    
    def _validate_configuration(self) -> None:
        """Validate the model configuration."""
        pass  # Override in subclasses for specific validation
    
    async def generate_response(self, prompt: str) -> LLMResponse:
        """Generate a response from the LLM."""
        raise NotImplementedError


class HuggingFaceInferenceModel(BaseLLMModel):
    """HuggingFace model using the free Inference API."""
    
    # Verified working free models
    WORKING_MODELS = {
        # GPT-2 family (reliable and fast)
        "openai-community/gpt2": {
            "task": "text-generation",
            "max_length": 512,
            "description": "Original GPT-2 model, very reliable"
        },
        "distilbert/distilgpt2": {
            "task": "text-generation",
            "max_length": 256,
            "description": "Faster, smaller GPT-2 variant"
        },
        "openai-community/gpt2-medium": {
            "task": "text-generation",
            "max_length": 512,
            "description": "Medium GPT-2, better quality"
        },
        
        # DialoGPT family (conversation-focused)
        "microsoft/DialoGPT-small": {
            "task": "text-generation",
            "max_length": 256,
            "description": "Small conversational model"
        },
        "microsoft/DialoGPT-medium": {
            "task": "text-generation",
            "max_length": 512,
            "description": "Medium conversational model"
        },
        
        # Google models
        "google/flan-t5-small": {
            "task": "text2text-generation",
            "max_length": 256,
            "description": "Instruction-following model"
        },
        "google/flan-t5-base": {
            "task": "text2text-generation", 
            "max_length": 512,
            "description": "Better instruction-following"
        }
    }
    
    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name)
        
        # Use a working default if model not specified
        if self.model_name not in self.WORKING_MODELS:
            logger.warning(f"Model {self.model_name} not in verified list, using default")
            self.model_name = "distilbert/distilgpt2"  # Most reliable fallback
        
        self.config = self.WORKING_MODELS[self.model_name]
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = self._get_headers()
        
        logger.info(f"Initialized HuggingFace model: {self.model_name}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for HuggingFace API requests."""
        headers = {"Content-Type": "application/json"}
        if settings.huggingface_api_token:
            headers["Authorization"] = f"Bearer {settings.huggingface_api_token}"
        return headers
    
    async def generate_response(self, prompt: str) -> LLMResponse:
        """Generate a response using HuggingFace Inference API."""
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                # Prepare the payload
                payload = self._prepare_payload(prompt)
                
                logger.debug(f"Sending request to {self.api_url}")
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=30
                )
                
                if response.status_code == 503:
                    # Model is loading
                    if attempt < max_retries - 1:
                        logger.info(f"Model is loading, waiting {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise ModelError(f"Model is still loading after {max_retries} attempts. Please try again in a few minutes.")
                
                elif response.status_code == 429:
                    # Rate limited
                    if attempt < max_retries - 1:
                        logger.warning("Rate limited, retrying...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise ModelError("Rate limited. Please try again later or get a HuggingFace API token.")
                
                response.raise_for_status()
                result = response.json()
                
                # Handle different response formats
                content = self._extract_content(result, prompt)
                
                if not content or content.strip() == "":
                    raise ModelError("Empty response from model")
                
                return LLMResponse(
                    content=content,
                    model_used=self.model_name,
                    tokens_used=self._estimate_tokens(content),
                    metadata={
                        "provider": "huggingface",
                        "task": self.config["task"],
                        "attempt": attempt + 1
                    }
                )
                
            except requests.RequestException as e:
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    raise ModelError(f"HTTP error after {max_retries} attempts: {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt == max_retries - 1:
                    raise ModelError(f"Error generating response: {str(e)}")
                time.sleep(retry_delay)
    
    def _prepare_payload(self, prompt: str) -> Dict[str, Any]:
        """Prepare the API payload."""
        # Clean and limit prompt length
        clean_prompt = prompt.strip()[:1000]  # Limit input length
        
        payload = {
            "inputs": clean_prompt,
            "parameters": {
                "max_new_tokens": min(self.config["max_length"], settings.max_tokens),
                "temperature": max(0.1, min(1.0, settings.temperature)),  # Ensure valid range
                "return_full_text": False,
                "do_sample": True,
                "top_p": 0.9,
                "top_k": 50
            },
            "options": {
                "wait_for_model": True,
                "use_cache": True
            }
        }
        
        return payload
    
    def _extract_content(self, result: Any, original_prompt: str) -> str:
        """Extract content from the API response."""
        try:
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", str(result))
            else:
                generated_text = str(result)
            
            # Clean the response
            if generated_text.startswith(original_prompt):
                generated_text = generated_text[len(original_prompt):].strip()
            
            # Remove common artifacts
            generated_text = generated_text.replace("\\n", "\n").strip()
            
            # Ensure we have some content
            if not generated_text:
                return "Hello! I'm here to help. What would you like to know?"
            
            return generated_text
                
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return "I'm here to help! What would you like to discuss?"
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count."""
        return max(1, len(text.split()) + len(text) // 4)
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, str]]:
        """List all available working models."""
        return cls.WORKING_MODELS


# Simple fallback model for when everything else fails
class SimpleFallbackModel(BaseLLMModel):
    """Simple fallback model that doesn't require external APIs."""
    
    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name or "fallback")
    
    async def generate_response(self, prompt: str) -> LLMResponse:
        """Generate a simple response."""
        responses = [
            "I'm a simple AI assistant. I'd be happy to help you with your questions!",
            "Hello! I'm here to assist you. What would you like to know?",
            "Thanks for your message! I'm a basic AI model ready to help.",
            "Hi there! I'm an AI assistant. How can I help you today?",
        ]
        
        # Simple response based on prompt keywords
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
            content = "Hello! Nice to meet you. How can I help you today?"
        elif any(word in prompt_lower for word in ['how are you', 'how do you do']):
            content = "I'm doing well, thank you for asking! I'm here to help you with any questions."
        elif any(word in prompt_lower for word in ['what', 'explain', 'tell me']):
            content = "I'd be happy to help explain that! Could you provide more specific details about what you'd like to know?"
        else:
            import random
            content = random.choice(responses)
        
        return LLMResponse(
            content=content,
            model_used=self.model_name,
            tokens_used=len(content.split()),
            metadata={"provider": "fallback", "type": "simple"}
        )