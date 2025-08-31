"""API routes for PDF processing and summarization."""
import logging
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...services.summarization_service import PDFSummarizationService
from ...core.exceptions import ModelError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pdf", tags=["PDF Processing"])


class SummaryResponse(BaseModel):
    """Response model for PDF summarization."""
    summary: str
    key_points: List[str]
    document_stats: dict
    filename: str
    summary_type: str
    success: bool
    error: Optional[str] = None


class PreviewResponse(BaseModel):
    """Response model for PDF preview."""
    preview_text: str
    stats: dict
    success: bool
    error: Optional[str] = None


@router.post("/upload-and-summarize", response_model=SummaryResponse)
async def upload_and_summarize_pdf(
    file: UploadFile = File(..., description="PDF file to summarize"),
    summary_type: str = Form("comprehensive", description="Type: brief, comprehensive, bullet_points"),
    model_name: str = Form("google/flan-t5-base", description="HuggingFace model to use")
):
    """
    Upload a PDF file and get a summary of its content.
    """
    # Validate file type
    if not file.content_type or "pdf" not in file.content_type.lower():
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400, 
                detail="Only PDF files are supported"
            )
    
    # Validate file size (10MB limit)
    file_size = 0
    content = await file.read()
    file_size = len(content)
    
    if file_size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=413,
            detail="File too large. Maximum size is 10MB."
        )
    
    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file received"
        )
    
    try:
        logger.info(f"Processing PDF: {file.filename} ({file_size} bytes)")
        
        # Initialize summarization service
        summarization_service = PDFSummarizationService(
            summarization_model=model_name,
            chunk_size=800
        )
        
        # Process the PDF
        result = await summarization_service.summarize_pdf(
            pdf_content=content,
            filename=file.filename,
            summary_type=summary_type
        )
        
        return SummaryResponse(**result)
        
    except ModelError as e:
        logger.error(f"Model error processing {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/preview", response_model=PreviewResponse)
async def preview_pdf(
    file: UploadFile = File(..., description="PDF file to preview")
):
    """
    Upload a PDF file and get a quick preview of its content.
    """
    # Validate file
    if not file.content_type or "pdf" not in file.content_type.lower():
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file received")
        
        logger.info(f"Generating preview for: {file.filename}")
        
        # Initialize service
        summarization_service = PDFSummarizationService()
        
        # Get preview
        result = await summarization_service.get_quick_preview(content)
        
        return PreviewResponse(**result)
        
    except Exception as e:
        logger.error(f"Preview failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Preview failed: {str(e)}")


@router.get("/supported-models")
async def get_supported_models():
    """Get list of supported models for summarization."""
    return {
        "models": {
            "google/flan-t5-base": {
                "description": "Good for instruction-following and summarization",
                "recommended": True,
                "speed": "medium",
                "quality": "high"
            },
            "google/flan-t5-small": {
                "description": "Faster, smaller model",
                "recommended": False,
                "speed": "fast", 
                "quality": "medium"
            },
            "microsoft/DialoGPT-medium": {
                "description": "Conversational model, good for natural summaries",
                "recommended": True,
                "speed": "medium",
                "quality": "good"
            },
            "distilbert/distilgpt2": {
                "description": "Fast general-purpose model",
                "recommended": False,
                "speed": "very fast",
                "quality": "medium"
            }
        },
        "summary_types": {
            "brief": "2-3 sentence summary",
            "comprehensive": "Detailed summary with main points",
            "bullet_points": "Key points as bullet list"
        }
    }


@router.get("/health")
async def pdf_health_check():
    """Health check for PDF processing service."""
    try:
        # Quick test of PDF processor
        from ...utils.pdf_processor import PDFProcessor
        processor = PDFProcessor()
        
        return {
            "status": "healthy",
            "service": "PDF Processing",
            "pdf_libraries_available": True,
            "max_file_size_mb": 10
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "service": "PDF Processing",
            "error": str(e),
            "pdf_libraries_available": False
        }


@router.post("/test")
async def test_pdf_processing():
    """Test endpoint for PDF processing without file upload."""
    try:
        # Create a simple test
        from ...services.summarization_service import PDFSummarizationService
        
        service = PDFSummarizationService()
        
        # Test with sample text
        test_text = """
        This is a test document about artificial intelligence. 
        AI has become increasingly important in modern technology. 
        Machine learning algorithms can process vast amounts of data.
        Natural language processing enables computers to understand human language.
        The future of AI looks very promising with many applications.
        """
        
        # Simulate PDF processing result
        return {
            "status": "success",
            "message": "PDF processing components are working",
            "test_summary": "This document discusses artificial intelligence, machine learning, and natural language processing technologies.",
            "service_ready": True
        }
        
    except Exception as e:
        logger.error(f"PDF processing test failed: {e}")
        return {
            "status": "error",
            "message": f"PDF processing test failed: {str(e)}",
            "service_ready": False
        }


@router.get("/examples")
async def get_usage_examples():
    """Get usage examples for the PDF API."""
    return {
        "upload_and_summarize": {
            "method": "POST",
            "url": "/pdf/upload-and-summarize",
            "description": "Upload PDF and get summary",
            "form_data": {
                "file": "PDF file (required)",
                "summary_type": "brief|comprehensive|bullet_points (optional, default: comprehensive)",
                "model_name": "HuggingFace model name (optional, default: google/flan-t5-base)"
            },
            "curl_example": """curl -X POST "http://localhost:8000/pdf/upload-and-summarize" \\
  -F "file=@document.pdf" \\
  -F "summary_type=comprehensive" \\
  -F "model_name=google/flan-t5-base\""""
        },
        "preview": {
            "method": "POST", 
            "url": "/pdf/preview",
            "description": "Get quick preview of PDF content",
            "curl_example": """curl -X POST "http://localhost:8000/pdf/preview" \\
  -F "file=@document.pdf\""""
        }
    }