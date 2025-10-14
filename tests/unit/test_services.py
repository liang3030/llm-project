"""Unit tests for HuggingFace LLM service."""
import pytest
from unittest.mock import Mock, patch, AsyncMock

from src.my_llm_project.services.llm_service import LLMService
from src.my_llm_project.models.llm_models import LLMResponse


class TestLLMServiceHuggingFace:
    """Test cases for LLMService with HuggingFace focus."""
    
    @patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
    def test_init_default_inference(self, mock_hf_model):
        """Test default initialization with inference model."""
        service = LLMService()
        
        assert service.model_type == "inference"
        assert service.task_type == "conversation"
        mock_hf_model.assert_called_once()
    
    @patch('src.my_llm_project.services.llm_service.HuggingFacePipelineModel')
    def test_init_pipeline_model(self, mock_pipeline_model):
        """Test initialization with pipeline model."""
        service = LLMService(model_type="pipeline")
        
        assert service.model_type == "pipeline"
        mock_pipeline_model.assert_called_once()
    
    @patch('src.my_llm_project.services.llm_service.ChatHuggingFaceModel')
    def test_init_chat_model(self, mock_chat_model):
        """Test initialization with chat model."""
        service = LLMService(model_type="chat")
        
        assert service.model_type == "chat"
        mock_chat_model.assert_called_once()
    
    @patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
    def test_init_custom_model_name(self, mock_hf_model):
        """Test initialization with custom model name."""
        service = LLMService(model_name="google/flan-t5-base", task_type="instruction")
        
        mock_hf_model.assert_called_once_with("google/flan-t5-base")
        assert service.model_name == "google/flan-t5-base"
        assert service.task_type == "instruction"
    
    @patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
    @pytest.mark.asyncio
    async def test_process_prompt_success(self, mock_hf_model):
        """Test successful prompt processing."""
        # Mock the model response
        mock_response = LLMResponse(
            content="This is a test response from HuggingFace",
            model_used="microsoft/DialoGPT-medium",
            tokens_used=25,
            metadata={"provider": "huggingface"}
        )
        
        mock_model_instance = AsyncMock()
        mock_model_instance.generate_response.return_value = mock_response
        mock_hf_model.return_value = mock_model_instance
        
        service = LLMService()
        result = await service.process_prompt("Hello, how are you?")
        
        assert result.content == "This is a test response from HuggingFace"
        assert result.model_used == "microsoft/DialoGPT-medium"
        assert result.tokens_used == 25
        assert result.metadata["provider"] == "huggingface"
    
    @patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
    @pytest.mark.asyncio
    async def test_process_prompt_with_context(self, mock_hf_model):
        """Test prompt processing with context."""
        mock_response = LLMResponse(
            content="Response with context",
            model_used="microsoft/DialoGPT-medium",
            tokens_used=20
        )
        
        mock_model_instance = AsyncMock()
        mock_model_instance.generate_response.return_value = mock_response
        mock_hf_model.return_value = mock_model_instance
        
        service = LLMService()
        context = {"user_name": "Alice", "topic": "weather"}
        
        result = await service.process_prompt("What's the weather like?", context)
        
        # Check that the model was called with enhanced prompt
        called_prompt = mock_model_instance.generate_response.call_args[0][0]
        assert "user_name: Alice" in called_prompt
        assert "topic: weather" in called_prompt
        assert "What's the weather like?" in called_prompt
    
    @patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
    def test_enhance_prompt_for_dialogpt(self, mock_hf_model):
        """Test prompt enhancement for DialoGPT models."""
        service = LLMService(model_name="microsoft/DialoGPT-medium")
        
        prompt = "How are you"  # No punctuation
        enhanced = service._enhance_prompt_for_huggingface(prompt)
        
        assert enhanced.endswith(":")  # Should add colon for DialoGPT
    
    @patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')  
    def test_enhance_prompt_for_flan_t5(self, mock_hf_model):
        """Test prompt enhancement for FLAN-T5 models."""
        service = LLMService(model_name="google/flan-t5-base")
        
        prompt = "What is Python?"
        enhanced = service._enhance_prompt_for_huggingface(prompt)
        
        assert enhanced.startswith("Answer the following question:")
    
    @patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
    def test_enhance_prompt_for_bloom(self, mock_hf_model):
        """Test prompt enhancement for BLOOM models."""
        service = LLMService(model_name="bigscience/bloom-560m")
        
        prompt = "Tell me about AI"
        enhanced = service._enhance_prompt_for_huggingface(prompt)
        
        assert "Human:" in enhanced
        assert "Assistant:" in enhanced
    
    def test_postprocess_response_cleaning(self):
        """Test response post-processing and cleaning."""
        service = LLMService()
        
        # Test repetitive text removal
        messy_response = "Hello there!\nHello there!\nHow can I help you?\nHow can I help you?"
        cleaned = service._postprocess_response(messy_response)
        
        lines = cleaned.split('\n')
        assert len(lines) == 2  # Should remove duplicate lines
        assert "Hello there!" in lines
        assert "How can I help you?" in lines
    
    def test_postprocess_response_truncation(self):
        """Test response truncation for very long responses."""
        service = LLMService()
        
        # Create a very long response
        long_response = ". ".join([f"This is sentence number {i}" for i in range(20)])
        cleaned = service._postprocess_response(long_response)
        
        # Should be truncated to first 5 sentences
        sentences = cleaned.split('. ')
        assert len(sentences) <= 6  # 5 sentences + potentially incomplete last one
    
    @patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
    def test_switch_model(self, mock_hf_model):
        """Test switching between models."""
        service = LLMService(model_name="microsoft/DialoGPT-medium")
        initial_model_name = service.model_name
        
        # Switch to a different model
        service.switch_model(model_name="google/flan-t5-base", task_type="instruction")
        
        assert service.model_name == "google/flan-t5-base"
        assert service.task_type == "instruction"
        assert mock_hf_model.call_count == 2  # Initial + switch
    
    def test_get_available_models(self):
        """Test getting available models information."""
        service = LLMService()
        models_info = service.get_available_models()
        
        assert "recommended_models" in models_info
        assert "model_types" in models_info
        assert "task_recommendations" in models_info
        
        # Check structure
        assert "inference" in models_info["model_types"]
        assert "conversation" in models_info["task_recommendations"]
    
    def test_get_model_info(self):
        """Test getting current model information."""
        service = LLMService(
            model_name="microsoft/DialoGPT-medium",
            model_type="inference",
            task_type="conversation"
        )
        
        info = service.get_model_info()
        
        assert info["model_name"] == "microsoft/DialoGPT-medium"
        assert info["model_type"] == "inference"
        assert info["task_type"] == "conversation"
        assert info["provider"] == "huggingface"
        assert "config" in info