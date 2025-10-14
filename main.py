"""Main application entry point."""
import uvicorn
import logging
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from my_llm_project.api import create_app
    from my_llm_project.core.config import settings
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, getattr(settings, 'log_level', 'INFO'), logging.INFO),
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    app = create_app()
    
    
    if __name__ == "__main__":
        print(f"Starting {settings.app_name}")
        print(f"Model: {settings.model_name}")
        print(f"Visit: http://localhost:{settings.port}/docs")
        
        uvicorn.run(
            app,
            host=settings.host,
            port=settings.port,
            reload=settings.debug
        )
        
except Exception as e:
    print(f"Error starting app: {e}")
    print("Trying basic startup...")
    
    # Fallback startup
    uvicorn.run(
        "my_llm_project.api:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=True
    )