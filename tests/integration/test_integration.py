"""Integration tests for HuggingFace API endpoints."""
import pytest
from httpx import AsyncClient
from unittest.mock import patch, Mock

from src.my_llm_project.api import create_app


@pytest.fixture
def app():
    """Create test app."""
    return create_app()


@pytest.mark.asyncio
async def test_health_check(app):
    """Test health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/llm/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["provider"] == "huggingface"


@pytest.mark.asyncio
async def test_list_models_endpoint(app):
    """Test the models listing endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/llm/models")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "recommended_models" in data
        assert "model_types" in data
        assert "task_recommendations" in data


@pytest.mark.asyncio
async def test_model_recommendations_endpoint(app):
    """Test the model recommendations endpoint.""" 
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/llm/models/recommendations?task=conversation")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "primary" in data
        assert "description" in data
        assert data["primary"] == "microsoft/DialoGPT-medium"


@pytest.mark.asyncio
@patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
async def test_generate_response_endpoint(mock_hf_model, app):
    """Test the generate response endpoint."""
    from src.my_llm_project.models.llm_models import LLMResponse
    
    # Mock the model response
    mock_response = LLMResponse(
        content="Hello! I'm doing well, thank you for asking. How are you?",
        model_used="microsoft/DialoGPT-medium",
        tokens_used=30,
        metadata={"provider": "huggingface", "task": "text-generation"}
    )
    
    mock_model_instance = Mock()
    mock_model_instance.generate_response.return_value = mock_response
    mock_hf_model.return_value = mock_model_instance
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/llm/generate",
            json={
                "prompt": "How are you today?",
                "model_name": "microsoft/DialoGPT-medium",
                "model_type": "inference"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["content"] == "Hello! I'm doing well, thank you for asking. How are you?"
        assert data["model_used"] == "microsoft/DialoGPT-medium"
        assert data["provider"] == "huggingface"
        assert data["tokens_used"] == 30


@pytest.mark.asyncio
@patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
async def test_chat_endpoint(mock_hf_model, app):
    """Test the specialized chat endpoint."""
    from src.my_llm_project.models.llm_models import LLMResponse
    
    mock_response = LLMResponse(
        content="Nice to meet you! I'm an AI assistant. What would you like to talk about?",
        model_used="microsoft/DialoGPT-medium",
        tokens_used=25,
        metadata={"provider": "huggingface"}
    )
    
    mock_model_instance = Mock()
    mock_model_instance.generate_response.return_value = mock_response
    mock_hf_model.return_value = mock_model_instance
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/llm/chat",
            json={
                "prompt": "Hello! Nice to meet you.",
                "context": {"user_name": "Alice"}
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_type"] == "chat-optimized"
        assert "Nice to meet you" in data["content"]


@pytest.mark.asyncio
@patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
async def test_instruction_endpoint(mock_hf_model, app):
    """Test the instruction-following endpoint."""
    from src.my_llm_project.models.llm_models import LLMResponse
    
    mock_response = LLMResponse(
        content="Python is a high-level, interpreted programming language known for its simplicity and readability.",
        model_used="google/flan-t5-base",
        tokens_used=20,
        metadata={"provider": "huggingface", "task": "text2text-generation"}
    )
    
    mock_model_instance = Mock()
    mock_model_instance.generate_response.return_value = mock_response
    mock_hf_model.return_value = mock_model_instance
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/llm/instruction",
            json={
                "prompt": "What is Python?",
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_type"] == "instruction-following"
        assert "Python is a" in data["content"]


@pytest.mark.asyncio
async def test_switch_model_endpoint(app):
    """Test the model switching endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/llm/models/switch",
            json={
                "model_name": "google/flan-t5-base",
                "model_type": "inference",
                "task_type": "instruction"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert "current_model" in data
        assert data["current_model"]["model_name"] == "google/flan-t5-base"


@pytest.mark.asyncio
@patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
async def test_test_model_endpoint_success(mock_hf_model, app):
    """Test the model testing endpoint with success."""
    from src.my_llm_project.models.llm_models import LLMResponse
    
    mock_response = LLMResponse(
        content="Hello! I'm doing great, thanks for asking.",
        model_used="microsoft/DialoGPT-small",
        tokens_used=15,
        metadata={"provider": "huggingface"}
    )
    
    mock_model_instance = Mock()
    mock_model_instance.generate_response.return_value = mock_response
    mock_hf_model.return_value = mock_model_instance
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/llm/test?model_name=microsoft/DialoGPT-small&model_type=inference"
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] is True
        assert data["model_tested"] == "microsoft/DialoGPT-small"
        assert "Hello!" in data["response_preview"]


@pytest.mark.asyncio
@patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel')
async def test_test_model_endpoint_failure(mock_hf_model, app):
    """Test the model testing endpoint with failure."""
    # Mock model initialization failure
    mock_hf_model.side_effect = Exception("Model not available")
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post(
            "/llm/test?model_name=nonexistent/model&model_type=inference"
        )
        
        assert response.status_code == 200  # Endpoint doesn't fail, just returns error info
        data = response.json()
        
        assert data["success"] is False
        assert "error" in data
        assert "suggestion" in data


@pytest.mark.asyncio
async def test_generate_with_context(app):
    """Test generation with context data."""
    with patch('src.my_llm_project.services.llm_service.HuggingFaceInferenceModel') as mock_hf:
        from src.my_llm_project.models.llm_models import LLMResponse
        
        mock_response = LLMResponse(
            content="Based on the context provided, I can help you with your Python project.",
            model_used="microsoft/DialoGPT-medium",
            tokens_used=35
        )
        
        mock_model_instance = Mock()
        mock_model_instance.generate_response.return_value = mock_response
        mock_hf.return_value = mock_model_instance
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/llm/generate",
                json={
                    "prompt": "Can you help me with my project?",
                    "context": {
                        "project_type": "Python web application",
                        "framework": "FastAPI",
                        "issue": "API endpoint design"
                    },
                    "enhance_prompt": True
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            
            # Verify the model was called with enhanced prompt containing context
            called_prompt = mock_model_instance.generate_response.call_args[0][0]
            assert "project_type: Python web application" in called_prompt
            assert "framework: FastAPI" in called_prompt
            assert "issue: API endpoint design" in called_prompt


@pytest.mark.asyncio
async def test_error_handling(app):
    """Test error handling in API endpoints."""
    with patch('src.my_llm_project.services.llm_service.LLMService') as mock_service:
        # Mock service to raise an exception
        mock_service.side_effect = Exception("Service unavailable")
        
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/llm/generate",
                json={"prompt": "This will fail"}
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "Internal server error" in data["detail"]


@pytest.mark.asyncio
async def test_recommendations_unknown_task(app):
    """Test model recommendations for unknown task."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/llm/models/recommendations?task=unknown_task")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "error" in data
        assert "available_tasks" in data
        assert "conversation" in data["available_tasks"]