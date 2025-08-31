"""FastAPI application factory."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.llm_routes import router as llm_router
from .routes.pdf_routes import router as pdf_router
from ..core.config import settings


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        debug=settings.debug,
        docs_url="/docs",
        redoc_url="/redoc",
        description="LLM API with PDF summarization capabilities using HuggingFace models"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure this properly for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(llm_router)
    app.include_router(pdf_router)
    
    # Add root endpoint
    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.version,
            "provider": settings.model_provider,
            "model": settings.model_name,
            "features": ["LLM Chat", "PDF Summarization"],
            "docs": "/docs",
            "endpoints": {
                "llm": "/llm/",
                "pdf": "/pdf/"
            }
        }
    
    return app