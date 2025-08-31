"""LLM service layer with robust error handling."""
from typing import Optional, Dict, Any
import logging

from ..models.llm_models import (
    LLMResponse, BaseLLMModel, HuggingFaceInferenceModel, SimpleFallbackModel
)
from ..core.config import settings
from ..core.exceptions import ModelError, ConfigurationError

logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing LLM interactions with robust fallbacks."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        use_fallback: bool = True
    ):
        self.model_name = model_name or self._get_safe_default_model()
        self.use_fallback = use_fallback
        self.model = self._initialize_model_with_fallback()
        
        logger.info(f"Initialized LLMService with model: {self.model_name}")
    
    def _get_safe_default_model(self) -> str:
        """Get a safe default model that's guaranteed to work."""
        # These models are verified to work with HuggingFace Inference API
        safe_models = [
            "distilbert/distilgpt2",      # Most reliable, fast
            "openai-community/gpt2",      # Very reliable
            "microsoft/DialoGPT-small",   # Good for chat
        ]
        
        # Use configured model if it's in our safe list
        if settings.model_name in HuggingFaceInferenceModel.WORKING_MODELS:
            return settings.model_name
        
        # Otherwise use the most reliable default
        return safe_models[0]
    
    def _initialize_model_with_fallback(self) -> BaseLLMModel:
        """Initialize model with automatic fallback to simpler models."""
        fallback_order = [
            self.model_name,
            "distilbert/distilgpt2",
            "openai-community/gpt2", 
            "microsoft/DialoGPT-small"
        ]
        
        for model_name in fallback_order:
            try:
                logger.info(f"Trying to initialize model: {model_name}")
                model = HuggingFaceInferenceModel(model_name)
                self.model_name = model_name  # Update to working model
                logger.info(f"Successfully initialized: {model_name}")
                return model
                
            except Exception as e:
                logger.warning(f"Failed to initialize {model_name}: {e}")
                continue
        
        # If all HuggingFace models fail, use simple fallback
        if self.use_fallback:
            logger.warning("All HuggingFace models failed, using simple fallback")
            return SimpleFallbackModel()
        else:
            raise ConfigurationError("Failed to initialize any HuggingFace model")
    
    async def process_prompt(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None,
        enhance_prompt: bool = True
    ) -> LLMResponse:
        """Process a prompt with error handling and fallback."""
        try:
            logger.info(f"Processing prompt with {self.model_name}")
            
            # Enhance prompt for better responses
            if enhance_prompt:
                processed_prompt = self._enhance_prompt(prompt, context)
            else:
                processed_prompt = self._basic_preprocess(prompt, context)
            
            response = await self.model.generate_response(processed_prompt)
            
            # Post-process response
            response.content = self._postprocess_response(response.content)
            
            logger.info(f"Successfully generated response")
            return response
            
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}")
            
            # Try fallback response if enabled
            if self.use_fallback and not isinstance(self.model, SimpleFallbackModel):
                try:
                    logger.info("Attempting fallback response...")
                    fallback_model = SimpleFallbackModel()
                    return await fallback_model.generate_response(prompt)
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            raise ModelError(f"Failed to generate response: {str(e)}")
    
    def _enhance_prompt(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Enhance prompts for better model performance."""
        enhanced_prompt = prompt.strip()
        
        # Add context if provided
        if context:
            context_lines = []
            for key, value in context.items():
                if value:  # Only add non-empty values
                    context_lines.append(f"{key}: {value}")
            
            if context_lines:
                context_str = "\n".join(context_lines)
                enhanced_prompt = f"Context:\n{context_str}\n\nUser: {enhanced_prompt}"
        
        # Model-specific enhancements
        if "DialoGPT" in self.model_name:
            # DialoGPT works better with conversational markers
            if not enhanced_prompt.endswith(('?', '.', '!', ':')):
                enhanced_prompt += ""  # DialoGPT handles this well
        
        elif "flan-t5" in self.model_name.lower():
            # FLAN-T5 works better with clear instructions
            if not any(enhanced_prompt.lower().startswith(word) for word in ['answer', 'explain', 'describe', 'what', 'how', 'why']):
                enhanced_prompt = f"Please answer: {enhanced_prompt}"
        
        elif "gpt2" in self.model_name.lower():
            # GPT-2 works well with natural continuation
            enhanced_prompt = enhanced_prompt  # Keep as is
        
        return enhanced_prompt
    
    def _basic_preprocess(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Basic prompt preprocessing."""
        if context:
            context_str = " | ".join([f"{k}: {v}" for k, v in context.items() if v])
            return f"[{context_str}] {prompt}"
        return prompt.strip()
    
    def _postprocess_response(self, response: str) -> str:
        """Clean up the response."""
        if not response:
            return "I'm here to help! What would you like to know?"
        
        # Basic cleaning
        response = response.strip()
        
        # Remove excessive whitespace
        while "\n\n\n" in response:
            response = response.replace("\n\n\n", "\n\n")
        
        # Limit length for readability
        if len(response) > 1000:
            sentences = response.split('. ')
            truncated = '. '.join(sentences[:3])
            if truncated and not truncated.endswith('.'):
                truncated += '.'
            response = truncated
        
        return response or "I'm here to help! What would you like to discuss?"
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        try:
            old_model = self.model_name
            self.model_name = model_name
            self.model = self._initialize_model_with_fallback()
            logger.info(f"Switched from {old_model} to {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to switch to {model_name}: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information.""" 
        return {
            "model_name": self.model_name,
            "provider": "huggingface",
            "available": self.model_name in HuggingFaceInferenceModel.WORKING_MODELS,
            "config": HuggingFaceInferenceModel.WORKING_MODELS.get(
                self.model_name,
                {"description": "Unknown model"}
            )
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get all available models and recommendations."""
        return {
            "working_models": HuggingFaceInferenceModel.WORKING_MODELS,
            "recommendations": {
                "fastest": "distilbert/distilgpt2",
                "best_chat": "microsoft/DialoGPT-medium", 
                "most_reliable": "openai-community/gpt2",
                "instruction_following": "google/flan-t5-base",
                "small_fast": "microsoft/DialoGPT-small"
            },
            "current_model": self.model_name
        }