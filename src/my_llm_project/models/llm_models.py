"""Updated LLM models with verified working HuggingFace Inference API models."""
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
        pass
    
    async def generate_response(self, prompt: str) -> LLMResponse:
        """Generate a response from the LLM."""
        raise NotImplementedError


class HuggingFaceInferenceModel(BaseLLMModel):
    """HuggingFace model using the free Inference API with verified working models."""
    
    # VERIFIED WORKING MODELS on HuggingFace Inference API (tested 2025)
    WORKING_MODELS = {
        # GPT-2 family (most reliable)
        "openai-community/gpt2": {
            "task": "text-generation",
            "max_length": 512,
            "description": "Original GPT-2, very reliable for text generation"
        },
        "distilbert/distilgpt2": {
            "task": "text-generation", 
            "max_length": 256,
            "description": "Faster GPT-2 variant, good for quick responses"
        },
        "openai-community/gpt2-medium": {
            "task": "text-generation",
            "max_length": 512,
            "description": "Medium GPT-2, better quality text generation"
        },
        
        # Microsoft DialoGPT (conversation focused)
        "microsoft/DialoGPT-small": {
            "task": "text-generation",
            "max_length": 256,
            "description": "Small conversational model, fast responses"
        },
        "microsoft/DialoGPT-medium": {
            "task": "text-generation",
            "max_length": 400,
            "description": "Medium conversational model, good quality"
        },
        
        # Alternative summarization models
        "facebook/bart-large-cnn": {
            "task": "summarization",
            "max_length": 400,
            "description": "BART model fine-tuned for CNN summarization"
        },
        "sshleifer/distilbart-cnn-12-6": {
            "task": "summarization", 
            "max_length": 300,
            "description": "Distilled BART for faster summarization"
        },
        
        # Hugging Face BlenderBot (conversation)
        "facebook/blenderbot-400M-distill": {
            "task": "text-generation",
            "max_length": 300,
            "description": "Facebook's conversational AI model"
        }
    }
    
    def __init__(self, model_name: Optional[str] = None):
        # Use fallback if specified model not available
        if model_name and model_name not in self.WORKING_MODELS:
            logger.warning(f"Model {model_name} not verified, using reliable fallback")
            model_name = "distilbert/distilgpt2"  # Most reliable
        
        super().__init__(model_name)
        
        # Ensure we're using a working model
        if self.model_name not in self.WORKING_MODELS:
            logger.warning(f"Using fallback model instead of {self.model_name}")
            self.model_name = "distilbert/distilgpt2"
        
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
                payload = self._prepare_payload(prompt)
                
                logger.debug(f"Attempt {attempt + 1}: Calling {self.api_url}")
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    data=json.dumps(payload),
                    timeout=30
                )
                
                logger.debug(f"Response status: {response.status_code}")
                
                # Handle specific response codes
                if response.status_code == 503:
                    if attempt < max_retries - 1:
                        logger.info(f"Model loading, waiting {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return self._create_fallback_response(prompt, "Model is loading, please try again in a few minutes.")
                
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        logger.warning("Rate limited, retrying...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        return self._create_fallback_response(prompt, "Service temporarily busy, please try again later.")
                
                elif response.status_code == 404:
                    logger.error(f"Model {self.model_name} not found")
                    return self._create_fallback_response(prompt, f"The model {self.model_name} is not available.")
                
                response.raise_for_status()
                result = response.json()
                logger.debug(f"API response: {result}")
                
                # Extract content from response
                content = self._extract_content(result, prompt)
                
                return LLMResponse(
                    content=content,
                    model_used=self.model_name,
                    tokens_used=self._estimate_tokens(content),
                    metadata={
                        "provider": "huggingface",
                        "task": self.config["task"],
                        "attempt": attempt + 1,
                        "status_code": response.status_code
                    }
                )
                
            except requests.RequestException as e:
                logger.error(f"Request failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return self._create_fallback_response(prompt, f"Network error: {str(e)}")
                time.sleep(retry_delay)
                retry_delay *= 2
            
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    return self._create_fallback_response(prompt, f"Processing error: {str(e)}")
                time.sleep(1)
    
    def _prepare_payload(self, prompt: str) -> Dict[str, Any]:
        """Prepare the API payload based on model type."""
        # Clean and limit prompt
        clean_prompt = prompt.strip()[:1500]  # Limit to avoid token issues
        
        if self.config["task"] == "summarization":
            # For summarization models like BART
            return {
                "inputs": clean_prompt,
                "parameters": {
                    "max_length": min(self.config["max_length"], settings.max_tokens),
                    "min_length": 50,
                    "do_sample": True,
                    "temperature": settings.temperature
                },
                "options": {"wait_for_model": True}
            }
        else:
            # For text generation models
            return {
                "inputs": clean_prompt,
                "parameters": {
                    "max_new_tokens": min(self.config["max_length"], settings.max_tokens),
                    "temperature": max(0.1, min(1.0, settings.temperature)),
                    "return_full_text": False,
                    "do_sample": True,
                    "top_p": 0.9
                },
                "options": {"wait_for_model": True}
            }
    
    def _extract_content(self, result: Any, original_prompt: str) -> str:
        """Extract content from API response."""
        try:
            if isinstance(result, dict) and "error" in result:
                logger.error(f"API returned error: {result['error']}")
                return f"I apologize, but I encountered an issue: {result['error']}"
            
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                
                # Handle summarization response
                if isinstance(item, dict) and "summary_text" in item:
                    return item["summary_text"].strip()
                
                # Handle text generation response  
                if isinstance(item, dict) and "generated_text" in item:
                    generated = item["generated_text"].strip()
                    # Remove original prompt if included
                    if generated.startswith(original_prompt):
                        generated = generated[len(original_prompt):].strip()
                    return generated
                
                # Handle direct string response
                if isinstance(item, str):
                    return item.strip()
            
            # Fallback: convert to string
            content = str(result).strip()
            return content if content else "I'm here to help! What would you like to know?"
            
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return "I'm having trouble processing that. Could you try rephrasing your question?"
    
    def _create_fallback_response(self, prompt: str, message: str) -> LLMResponse:
        """Create a fallback response when API fails."""
        # Simple fallback based on prompt keywords
        if any(word in prompt.lower() for word in ['summarize', 'summary', 'main points']):
            content = f"{message} For summarization, I can help identify key points when the service is available."
        elif any(word in prompt.lower() for word in ['hello', 'hi', 'hey']):
            content = f"Hello! {message}"
        else:
            content = f"{message} I'm still here to help once the service is restored."
        
        return LLMResponse(
            content=content,
            model_used=f"{self.model_name} (fallback)",
            tokens_used=len(content.split()),
            metadata={"provider": "fallback", "reason": message}
        )
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return max(1, len(text.split()) + len(text) // 5)
    
    @classmethod
    def get_best_model_for_task(cls, task: str) -> str:
        """Get the best model for a specific task."""
        task_models = {
            "summarization": "facebook/bart-large-cnn",
            "conversation": "microsoft/DialoGPT-medium", 
            "chat": "microsoft/DialoGPT-medium",
            "general": "distilbert/distilgpt2",
            "fast": "distilbert/distilgpt2"
        }
        
        model = task_models.get(task, "distilbert/distilgpt2")
        
        # Ensure the model exists in our working list
        if model not in cls.WORKING_MODELS:
            return "distilbert/distilgpt2"
        
        return model
    
    @classmethod
    def list_available_models(cls) -> Dict[str, Dict[str, Any]]:
        """List all available working models."""
        return cls.WORKING_MODELS


class SimpleFallbackModel(BaseLLMModel):
    """Simple fallback model for when HuggingFace API fails."""
    
    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name or "simple-fallback")
    
    async def generate_response(self, prompt: str) -> LLMResponse:
        """Generate a simple contextual response."""
        prompt_lower = prompt.lower()
        
        # Summarization requests
        if any(word in prompt_lower for word in ['summarize', 'summary', 'main points', 'key points']):
            content = "I can help with summarization when the AI service is available. In the meantime, I recommend looking for the main topics, key arguments, and conclusions in your document."
        
        # Greetings
        elif any(word in prompt_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            content = "Hello! I'm a helpful AI assistant. I can help with text analysis and summarization once the main service is restored."
        
        # Questions
        elif any(word in prompt_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            content = "That's a great question! I'd love to help you find the answer once the AI service is back online. Please try again in a moment."
        
        # Default
        else:
            content = "I'm here to help! The main AI service is temporarily unavailable, but I'm standing by to assist you as soon as it's restored."
        
        return LLMResponse(
            content=content,
            model_used=self.model_name,
            tokens_used=len(content.split()),
            metadata={"provider": "simple-fallback", "type": "contextual"}
        )