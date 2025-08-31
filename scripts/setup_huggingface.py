#!/usr/bin/env python3
"""
Setup script for HuggingFace integration.
This script helps verify the setup and test different models.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from my_llm_project.services.llm_service import LLMService
from my_llm_project.models.llm_models import HuggingFaceInferenceModel
from my_llm_project.core.config import settings


async def test_model(model_name: str, model_type: str = "inference") -> dict:
    """Test a specific model."""
    try:
        print(f"Testing {model_name} ({model_type})...")
        
        service = LLMService(model_type=model_type, model_name=model_name)
        response = await service.process_prompt("Hello! How are you today?")
        
        return {
            "model": model_name,
            "type": model_type,
            "success": True,
            "response": response.content[:100] + "..." if len(response.content) > 100 else response.content,
            "tokens": response.tokens_used
        }
    except Exception as e:
        return {
            "model": model_name,
            "type": model_type,
            "success": False,
            "error": str(e)
        }


async def main():
    """Main setup and testing function."""
    print("🤗 HuggingFace LLM Project Setup")
    print("=" * 40)
    
    # Check configuration
    print(f"Current model provider: {settings.model_provider}")
    print(f"Default model: {settings.model_name}")
    print(f"API token configured: {'Yes' if settings.huggingface_api_token else 'No'}")
    print()
    
    # List available models
    print("📋 Available Models:")
    models = HuggingFaceInferenceModel.list_available_models()
    for model_name, config in models.items():
        print(f"  • {model_name}: {config['description']}")
    print()
    
    # Test different models
    test_models = [
        ("microsoft/DialoGPT-small", "inference"),  # Fast and reliable
        ("microsoft/DialoGPT-medium", "inference"), # Balanced
        ("google/flan-t5-base", "inference"),       # Good for instructions
        ("distilgpt2", "inference"),                # Very fast
    ]
    
    print("🧪 Testing Models:")
    results = []
    
    for model_name, model_type in test_models:
        result = await test_model(model_name, model_type)
        results.append(result)
        
        if result["success"]:
            print(f"  ✅ {model_name}: {result['response']}")
        else:
            print(f"  ❌ {model_name}: {result['error']}")
    
    print()
    
    # Show recommendations
    print("💡 Recommendations:")
    working_models = [r for r in results if r["success"]]
    
    if working_models:
        print(f"  • For chat/conversation: Use 'microsoft/DialoGPT-medium'")
        print(f"  • For instructions: Use 'google/flan-t5-base'") 
        print(f"  • For speed: Use 'microsoft/DialoGPT-small'")
        print(f"  • For general use: Use 'distilgpt2'")
    else:
        print("  ⚠️  No models working. Check your internet connection.")
        print("  💡 Consider getting a HuggingFace API token for better reliability.")
    
    print()
    print("🚀 Setup Complete!")
    
    if not settings.huggingface_api_token:
        print()
        print("💡 Optional: Get a free HuggingFace API token for better performance:")
        print("   1. Go to https://huggingface.co/settings/tokens")
        print("   2. Create a new token")
        print("   3. Add HUGGINGFACE_API_TOKEN=your_token to your .env file")


if __name__ == "__main__":
    asyncio.run(main())