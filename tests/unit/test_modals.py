"""Unit tests for HuggingFace LLM models."""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import json

from src.my_llm_project.models.llm_models import (
    HuggingFaceInferenceModel, LLMResponse, get_recommended_huggingface_model
)
from src.my_llm_project.core.exceptions import ModelError, ConfigurationError


class TestHuggingFaceInferenceModel:
    """Test cases for HuggingFaceInferenceModel."""
    
    @patch('src.my_llm_project.models.llm_models.settings')
    @patch('src.my_llm_project.models.llm_models.requests')
    def test_init_success(self, mock_requests, mock_settings):
        """Test successful initialization."""
        mock_settings.huggingface_api_token = "test-token"
        mock_settings.huggingface_inference_api_url = "https://api-inference.huggingface.co/models/"
        
        # Mock the validation request
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.get.return_value = mock_response
        
        model = HuggingFaceInferenceModel("microsoft/DialoGPT-medium")
        
        assert model.model_name == "microsoft/DialoGPT-medium"
        assert model.api_url == "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        assert "Authorization" in model.headers
    
    @patch('src.my_llm_project.models.llm_models.settings')
    @patch('src.my_llm_project.models.llm_models.requests')
    def test_init_without_token(self, mock_requests, mock_settings):
        """Test initialization without API token."""
        mock_settings.huggingface_api_token = None
        mock_settings.huggingface_inference_api_url = "https://api-inference.huggingface.co/models/"
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_requests.get.return_value = mock_response
        
        model = HuggingFaceInferenceModel("distilgpt2")
        
        assert "Authorization" not in model.headers
        assert "Content-Type" in model.headers
    
    @patch('src.my_llm_project.models.llm_models.settings')
    @patch('src.my_llm_project.models.llm_models.requests')
    def test_model_not_found(self, mock_requests, mock_settings):
        """Test handling of model not found."""
        mock_settings.huggingface_api_token = "test-token"
        mock_settings.huggingface_inference_api_url = "https://api-inference.huggingface.co/models/"
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_requests.get.return_value = mock_response
        
        with pytest.raises(ConfigurationError, match="Model.*not found"):
            HuggingFaceInferenceModel("nonexistent/model")
    
    @patch('src.my_llm_project.models.llm_models.settings')
    @patch('src.my_llm_project.models.llm_models.requests')
    @pytest.mark.asyncio
    async def test_generate_response_success(self, mock_requests, mock_settings):
        """Test successful response generation."""
        mock_settings.huggingface_api_token = "test-token"
        mock_settings.huggingface_inference_api_url = "https://api-inference.huggingface.co/models/"
        mock_settings.max_tokens = 100
        mock_settings.temperature = 0.7
        
        # Mock validation request
        mock_validation = Mock()
        mock_validation.status_code = 200
        
        # Mock generation request
        mock_generation = Mock()
        mock_generation.status_code = 200
        mock_generation.raise_for_status = Mock()
        mock_generation.json.return_value = [{"generated_text": "Hello! I'm doing well, thank you for asking."}]
        
        mock_requests.get.return_value = mock_validation
        mock_requests.post.return_value = mock_generation
        
        model = HuggingFaceInferenceModel("microsoft/DialoGPT-medium")
        result = await model.generate_response("How are you?")
        
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello! I'm doing well, thank you for asking."
        assert result.model_used == "microsoft/DialoGPT-medium"
        assert result.metadata["provider"] == "huggingface"
    
    @patch('src.my_llm_project.models.llm_models.settings')
    @patch('src.my_llm_project.models.llm_models.requests')
    @pytest.mark.asyncio
    async def test_generate_response_with_prompt_removal(self, mock_requests, mock_settings):
        """Test response generation with prompt removal."""
        mock_settings.huggingface_api_token = "test-token"
        mock_settings.huggingface_inference_api_url = "https://api-inference.huggingface.co/models/"
        mock_settings.max_tokens = 100
        mock_settings.temperature = 0.7
        
        original_prompt = "How are you?"
        
        # Mock validation request
        mock_validation = Mock()
        mock_validation.status_code = 200
        
        # Mock generation request - response includes the original prompt
        mock_generation = Mock()
        mock_generation.status_code = 200
        mock_generation.raise_for_status = Mock()
        mock_generation.json.return_value = [{
            "generated_text": f"{original_prompt} I'm doing well, thank you!"
        }]
        
        mock_requests.get.return_value = mock_validation
        mock_requests.post.return_value = mock_generation
        
        model = HuggingFaceInferenceModel("microsoft/DialoGPT-medium")
        result = await model.generate_response(original_prompt)
        
        # The original prompt should be removed from the response
        assert result.content == "I'm doing well, thank you!"
        assert original_prompt not in result.content
    
    @patch('src.my_llm_project.models.llm_models.settings')
    @patch('src.my_llm_project.models.llm_models.requests')
    @patch('src.my_llm_project.models.llm_models.time')
    @pytest.mark.asyncio
    async def test_generate_response_model_loading(self, mock_time, mock_requests, mock_settings):
        """Test handling of model loading (503 error)."""
        mock_settings.huggingface_api_token = "test-token"
        mock_settings.huggingface_inference_api_url = "https://api-inference.huggingface.co/models/"
        mock_settings.max_tokens = 100
        mock_settings.temperature = 0.7
        
        # Mock validation request
        mock_validation = Mock()
        mock_validation.status_code = 200
        
        # Mock first request returns 503 (loading), second succeeds
        mock_loading_response = Mock()
        mock_loading_response.status_code = 503
        
        mock_success_response = Mock()
        mock_success_response.status_code = 200
        mock_success_response.raise_for_status = Mock()
        mock_success_response.json.return_value = [{"generated_text": "Response after loading"}]
        
        mock_requests.get.return_value = mock_validation
        mock_requests.post.side_effect = [mock_loading_response, mock_success_response]
        
        model = HuggingFaceInferenceModel("microsoft/DialoGPT-medium")
        result = await model.generate_response("Test prompt")
        
        assert result.content == "Response after loading"
        mock_time.sleep.assert_called_once_with(1)  # Should have slept before retry
    
    def test_list_available_models(self):
        """Test listing available models."""
        models = HuggingFaceInferenceModel.list_available_models()
        
        assert isinstance(models, dict)
        assert "microsoft/DialoGPT-medium" in models
        assert "google/flan-t5-base" in models
        assert "distilgpt2" in models
        
        # Check model configuration structure
        dialog_config = models["microsoft/DialoGPT-medium"]
        assert "task" in dialog_config
        assert "max_length" in dialog_config
        assert "description" in dialog_config


class TestModelRecommendations:
    """Test cases for model recommendation system."""
    
    def test_get_recommended_model_conversation(self):
        """Test getting recommended model for conversation."""
        model = get_recommended_huggingface_model("conversation")
        assert model == "microsoft/DialoGPT-medium"
    
    def test_get_recommended_model_instruction(self):
        """Test getting recommended model for instructions.""" 
        model = get_recommended_huggingface_model("instruction")
        assert model == "google/flan-t5-base"
    
    def test_get_recommended_model_unknown(self):
        """Test getting recommended model for unknown task."""
        model = get_recommended_huggingface_model("unknown_task")
        assert model == "microsoft/DialoGPT-medium"  # Should fallback to default