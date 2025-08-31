"""Fixed PDF summarization service using working HuggingFace models."""
import logging
from typing import List, Dict, Any, Optional
from langchain.schema import Document

from .llm_service import LLMService
from ..utils.pdf_processor import PDFProcessor
from ..models.llm_models import LLMResponse, HuggingFaceInferenceModel
from ..core.exceptions import ModelError

logger = logging.getLogger(__name__)


class PDFSummarizationService:
    """Service for PDF text extraction and summarization using working models."""
    
    def __init__(
        self, 
        summarization_model: Optional[str] = None,
        chunk_size: int = 800,
        max_summary_length: int = 500
    ):
        self.pdf_processor = PDFProcessor(chunk_size=chunk_size, chunk_overlap=100)
        
        # Use working model for summarization
        if not summarization_model or summarization_model not in HuggingFaceInferenceModel.WORKING_MODELS:
            # Try summarization-specific model first, fallback to general model
            if "facebook/bart-large-cnn" in HuggingFaceInferenceModel.WORKING_MODELS:
                summarization_model = "facebook/bart-large-cnn"
            else:
                summarization_model = "microsoft/DialoGPT-medium"  # Good for conversational summaries
            logger.info(f"Using fallback model for summarization: {summarization_model}")
        
        self.summarization_model = summarization_model
        self.llm_service = LLMService(model_name=summarization_model)
        self.max_summary_length = max_summary_length
        
        logger.info(f"Initialized summarization service with model: {summarization_model}")
    
    async def summarize_pdf(
        self, 
        pdf_content: bytes, 
        filename: str = "document.pdf",
        summary_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Summarize a PDF document using working HuggingFace models.
        """
        try:
            logger.info(f"Starting summarization of {filename} with {self.summarization_model}")
            
            # Step 1: Extract text from PDF
            extracted_text = await self.pdf_processor.extract_text_from_pdf(pdf_content)
            
            if not extracted_text.strip():
                raise ModelError("No text could be extracted from the PDF")
            
            # Step 2: Get document statistics
            stats = self.pdf_processor.get_text_stats(extracted_text)
            logger.info(f"Document stats: {stats['word_count']} words, {stats['character_count']} chars")
            
            # Step 3: Generate summary based on document length and model capability
            try:
                if stats['word_count'] <= 200:
                    summary = await self._summarize_short_document(extracted_text, summary_type)
                elif stats['word_count'] <= 800:
                    summary = await self._summarize_medium_document(extracted_text, summary_type)
                else:
                    summary = await self._summarize_long_document(extracted_text, summary_type)
            except Exception as e:
                logger.error(f"Summarization failed, creating manual summary: {e}")
                summary = self._create_manual_summary(extracted_text, summary_type)
            
            # Step 4: Extract key points
            key_points = self._extract_key_points(extracted_text)
            
            return {
                "summary": summary,
                "key_points": key_points,
                "document_stats": stats,
                "filename": filename,
                "summary_type": summary_type,
                "model_used": self.summarization_model,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"PDF summarization failed for {filename}: {e}")
            # Return a basic analysis instead of failing completely
            try:
                text = await self.pdf_processor.extract_text_from_pdf(pdf_content)
                manual_summary = self._create_manual_summary(text, summary_type) if text else "Could not extract text from PDF"
                stats = self.pdf_processor.get_text_stats(text) if text else {}
                
                return {
                    "summary": manual_summary,
                    "key_points": self._extract_key_points(text) if text else [],
                    "document_stats": stats,
                    "filename": filename,
                    "summary_type": summary_type,
                    "model_used": "manual-fallback",
                    "success": False,
                    "error": str(e)
                }
            except Exception as final_error:
                return {
                    "summary": f"Unable to process PDF: {str(final_error)}",
                    "key_points": [],
                    "document_stats": {},
                    "filename": filename,
                    "summary_type": summary_type,
                    "success": False,
                    "error": str(e)
                }
    
    async def _summarize_short_document(self, text: str, summary_type: str) -> str:
        """Summarize a short document (< 200 words)."""
        try:
            if "bart" in self.summarization_model.lower():
                # BART is trained for summarization
                prompt = text[:1000]  # BART can handle longer input
            else:
                # Use conversational approach for other models
                prompt = self._create_conversational_summary_prompt(text, summary_type)
            
            response = await self.llm_service.process_prompt(prompt, enhance_prompt=False)
            return self._clean_summary(response.content)
        except Exception as e:
            logger.error(f"Short doc summarization failed: {e}")
            return self._create_manual_summary(text, summary_type)
    
    async def _summarize_medium_document(self, text: str, summary_type: str) -> str:
        """Summarize a medium document (200-800 words)."""
        try:
            if "bart" in self.summarization_model.lower():
                # For BART, use the first 1000 characters
                prompt = text[:1000]
            else:
                # For other models, create a conversational prompt
                prompt = self._create_conversational_summary_prompt(text[:800], summary_type)
            
            response = await self.llm_service.process_prompt(prompt, enhance_prompt=False)
            return self._clean_summary(response.content)
        except Exception as e:
            logger.error(f"Medium doc summarization failed: {e}")
            return self._create_manual_summary(text, summary_type)
    
    async def _summarize_long_document(self, text: str, summary_type: str) -> str:
        """Summarize a long document using chunking strategy."""
        try:
            # Split into manageable chunks
            chunks = self.pdf_processor.chunk_text(text)
            logger.info(f"Split long document into {len(chunks)} chunks")
            
            if not chunks:
                return self._create_manual_summary(text, summary_type)
            
            # For long documents, take key sections and summarize
            key_chunks = chunks[:3]  # First 3 chunks usually contain main content
            
            chunk_summaries = []
            for i, chunk in enumerate(key_chunks):
                try:
                    chunk_text = chunk.page_content[:600]  # Limit chunk size
                    
                    if "bart" in self.summarization_model.lower():
                        prompt = chunk_text
                    else:
                        prompt = f"Summarize this text section: {chunk_text}"
                    
                    response = await self.llm_service.process_prompt(prompt, enhance_prompt=False)
                    chunk_summary = self._clean_summary(response.content)
                    
                    if chunk_summary and len(chunk_summary) > 20:
                        chunk_summaries.append(chunk_summary)
                        
                except Exception as e:
                    logger.warning(f"Failed to summarize chunk {i}: {e}")
                    continue
            
            if chunk_summaries:
                # Combine summaries
                combined = " ".join(chunk_summaries)
                return self._finalize_summary(combined, summary_type)
            else:
                return self._create_manual_summary(text, summary_type)
                
        except Exception as e:
            logger.error(f"Long document summarization failed: {e}")
            return self._create_manual_summary(text, summary_type)
    
    def _create_conversational_summary_prompt(self, text: str, summary_type: str) -> str:
        """Create a conversational prompt for non-BART models."""
        if summary_type == "brief":
            return f"Please give me a brief summary of this text: {text[:500]}"
        elif summary_type == "bullet_points":
            return f"List the main points from this text: {text[:500]}"
        else:
            return f"What are the key ideas in this text? {text[:600]}"
    
    def _create_manual_summary(self, text: str, summary_type: str) -> str:
        """Create a manual summary when AI models fail."""
        if not text or len(text.strip()) < 50:
            return "The document appears to be very short or empty."
        
        # Extract sentences
        sentences = []
        for sent in text.replace('\n', ' ').split('.'):
            sent = sent.strip()
            if len(sent) > 20:  # Filter short fragments
                sentences.append(sent)
        
        if not sentences:
            return "Unable to extract meaningful content from the document."
        
        # Take first few sentences and last sentence for basic summary
        if summary_type == "brief":
            if len(sentences) >= 2:
                return f"{sentences[0]}. {sentences[-1]}."
            else:
                return sentences[0] + "." if sentences else "Document summary not available."
        
        elif summary_type == "bullet_points":
            points = []
            for i, sent in enumerate(sentences[:5]):  # Max 5 points
                points.append(f"• {sent}")
            return "\n".join(points)
        
        else:  # comprehensive
            summary_sentences = sentences[:4]  # First 4 sentences
            return ". ".join(summary_sentences) + "."
    
    def _finalize_summary(self, combined_summary: str, summary_type: str) -> str:
        """Finalize the combined summary."""
        if not combined_summary:
            return "Summary not available."
        
        # Clean up the combined summary
        summary = self._clean_summary(combined_summary)
        
        # Ensure it's not too long
        if len(summary) > self.max_summary_length:
            sentences = summary.split('.')
            truncated = []
            length = 0
            for sent in sentences:
                if length + len(sent) < self.max_summary_length:
                    truncated.append(sent.strip())
                    length += len(sent)
                else:
                    break
            summary = '. '.join(truncated)
            if summary and not summary.endswith('.'):
                summary += '.'
        
        return summary
    
    def _clean_summary(self, summary: str) -> str:
        """Clean and format the generated summary."""
        if not summary:
            return "No summary could be generated."
        
        # Basic cleaning
        summary = summary.strip()
        
        # Remove common prefixes
        prefixes = [
            "Summary:", "Brief summary:", "Main points:", "Key points:",
            "The text discusses", "This document", "The document", 
            "Here is a summary", "Here are the main points"
        ]
        
        for prefix in prefixes:
            if summary.lower().startswith(prefix.lower()):
                summary = summary[len(prefix):].strip()
                if summary.startswith(":"):
                    summary = summary[1:].strip()
        
        # Ensure reasonable length
        if len(summary) > self.max_summary_length:
            summary = summary[:self.max_summary_length-3] + "..."
        
        return summary or "Unable to generate a meaningful summary."
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key sentences/points from the original text."""
        try:
            sentences = []
            
            # Split by sentences more carefully
            import re
            sentence_endings = re.split(r'[.!?]+', text)
            
            for sentence in sentence_endings:
                sentence = sentence.strip()
                if 30 < len(sentence) < 200:  # Filter by length
                    sentences.append(sentence)
            
            # Sort by length (longer sentences often have more info)
            sentences.sort(key=len, reverse=True)
            
            # Take top sentences
            key_points = []
            for sentence in sentences[:5]:
                if sentence not in key_points:  # Avoid duplicates
                    key_points.append(sentence.strip())
            
            return key_points[:3]  # Return max 3 key points
            
        except Exception as e:
            logger.error(f"Key point extraction failed: {e}")
            return ["Key points could not be extracted from this document."]
    
    async def get_quick_preview(self, pdf_content: bytes) -> Dict[str, Any]:
        """Get a quick preview of PDF content without full summarization."""
        try:
            text = await self.pdf_processor.extract_text_from_pdf(pdf_content)
            stats = self.pdf_processor.get_text_stats(text)
            
            # Get first few sentences as preview
            preview_text = text[:600] + "..." if len(text) > 600 else text
            
            return {
                "preview_text": preview_text,
                "stats": stats,
                "success": True
            }
        except Exception as e:
            return {
                "preview_text": "Could not extract text preview",
                "stats": {},
                "success": False,
                "error": str(e)
            }