

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.my_llm_project.services.pdf_chunker import PDFVectorService


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/pdf-search", tags=["PDF Processing and search"])


# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'pdf'}
MAX_CONTENT_LENGTH = 20 * 1024 * 1024 # 20MB max file size

# Singleton service instance
_pdf_service: Optional['PDFVectorService'] = None


def get_pdf_service():
    """
    Get or create PDF service instance (singleton pattern).
    This ensures only ONE instance is created and reused.
    """
    global _pdf_service
    if _pdf_service is None:
        # Import here to avoid circular imports
        from ...services.pdf_chunker import PDFVectorService
        logger.info("Initializing PDF service singleton...")
        _pdf_service = PDFVectorService()
        logger.info(f"PDF service initialized. Database has {_pdf_service.get_stats()['total_chunks']} chunks")
    return _pdf_service

class ProcessResponse(BaseModel):
		"""Response model for PDF processing."""
		success: bool = True
		message: str
		filename: str
		details: Dict[str, Any]

class SearchRequest(BaseModel):
		"""Request model for search."""
		query: str = Field(..., description="Search query string")
		n_results :int=Field(default=5, ge=1, le=50, description="Number of results to return")
		source_filter: Optional[str] = Field(None, description="Filter by specific PDF filename")

class SearchResult(BaseModel):
    """Individual search result."""
    text: str
    metadata: Dict[str, Any]
    similarity_score: float
    distance: float


class SearchResponse(BaseModel):
    """Response model for search."""
    success: bool = True
    query: str
    num_results: int
    results: List[SearchResult]

@router.post('/process-pdf', response_model=ProcessResponse)
async def process_pdf(
		file: UploadFile = File(..., description="PDF file to process"),
	  chunking_method: str = Form(default="sentences", description="Chunking method: sentences, paragraphs, or sliding_window"),
    max_chunk_size: Optional[int] = Form(default=None, description="Maximum chunk size"),
    overlap: Optional[int] = Form(default=None, description="Overlap size"),
):
		"""
		Process PDF and store in vector database

		Expected from data:
		- file: PDF file (required)
		- chunking_method: 'sentences', 'paragraphs' (optional, default: sentences)
		- max_chunk_size: Maximum size of chunks (optional, default: 500)
		- overlap: Overlap size (optional, default: 50)
		"""
		pdf_service = get_pdf_service()
		try:
			# check if file in request
			if not file.filename.lower().endswith('.pdf'):
				raise HTTPException(status_code=400, detail="Only PDF files are allowed")
			
			# Validate file size (20MB limit)
			contents = await file.read()
			if len(contents) > 20 * 1024 * 1024:
				raise HTTPException(status_code=400, detail="File size exceeds 20MB limit")
			
			# Build kwargs based on chunking method
			kwargs = {}
			if chunking_method == 'sentences':
				kwargs['max_chunk_size'] = max_chunk_size or 500
				kwargs['overlap'] = overlap or 50
			elif chunking_method == 'paragraphs':
				kwargs['max_chunk_size'] = max_chunk_size or 1000
				kwargs['overlap'] = overlap or 100	
			else:
				raise HTTPException(status_code=400, detail="Invalid chunking_method")	

			# Save file temporarily
			with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
				tmp_file.write(contents)
				tmp_path = tmp_file.name
			
			try:
				# Process PDF using service
				logger.info(f"Processing PDF: {file.filename} with method: {chunking_method}")
				result = pdf_service.process_pdf(tmp_path, file.filename, chunking_method, **kwargs)

				return ProcessResponse(
					message = "PDF processed successfully",
					filename=file.filename,
					details = result
				)
			finally:
				# Clearn up temporary file
				if os.path.exists(tmp_path):
					os.remove(tmp_path)
		
		except ValueError as ve:
			logger.error(f"Validation error: {str(ve)}")
			raise HTTPException(status_code=400, detail=str(ve))
		except Exception as e:
			logger.error(f"Error processing PDF: {str(e)}")
			raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
		

@router.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for similar content using English or German queries.
    
    - **query**: Search query string
    - **n_results**: Number of results to return (1-50)
    - **source_filter**: Optional filter by PDF filename
    """
    try:
        # Get singleton service instance
        pdf_service = get_pdf_service()
        
        # Search using service with named parameters
        logger.info(f"Searching for: '{request.query}' with n_results={request.n_results}")
        results = pdf_service.search_similar(
            query=request.query,
            n_results=request.n_results,
            source_filter=request.source_filter
        )

        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                formatted_results.append(
                    SearchResult(  # ✅ CORRECT! Use SearchResult, not SearchRequest
                        text=results['documents'][0][i],
                        metadata=results['metadatas'][0][i],
                        similarity_score=1 - results['distances'][0][i],
                        distance=results['distances'][0][i]
                    )
                )
        
        return SearchResponse(
            query=request.query,
            num_results=len(formatted_results),
            results=formatted_results
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")