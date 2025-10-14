"""FastAPI application factory."""
from contextlib import asynccontextmanager
from pathlib import Path
from chromadb import logger
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.pdf_chunks_routes import get_pdf_service, router as pdf_chunks_router
from ..core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    Replaces deprecated @app.on_event("startup") and @app.on_event("shutdown")
    """
    # Startup
    logger.info("Starting application services...")
    
    # Create chroma_db directory
    chroma_db_path = Path("./chroma_db")
    chroma_db_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"ChromaDB directory: {chroma_db_path.absolute()}")
    
    # Initialize PDF chunk service (triggers singleton creation)
    try:
        pdf_service = get_pdf_service()
        logger.info("✓ PDF Chunk Service initialized successfully")
        logger.info(f"  Current chunks in database: {pdf_service.get_stats()['total_chunks']}")
    except Exception as e:
        logger.error(f"✗ Failed to initialize PDF Chunk Service: {e}")
        logger.warning("PDF search endpoints may not work until service is initialized")
    
    logger.info("Application startup complete")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down application...")
    # Add any cleanup logic here if needed

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
    app.include_router(pdf_chunks_router)
    
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