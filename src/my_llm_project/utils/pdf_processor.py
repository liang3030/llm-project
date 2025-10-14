"""PDF processing utilities for text extraction and chunking."""
import logging
from typing import List, Optional
from pathlib import Path
import tempfile
from io import BytesIO
import re
import pdfplumber

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False

try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ..core.exceptions import ModelError

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handle PDF text extraction and processing."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Check PDF library availability
        if not (HAS_PYPDF2 or HAS_PYPDF):
            raise ModelError("No PDF processing library available. Install PyPDF2 or pypdf.")
    
    async def extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF bytes."""
        try:
            # Try pypdf first (more modern)
            return self.extract_with_pdfplumber(pdf_content)
            
                
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise ModelError(f"Failed to extract text from PDF: {str(e)}")
    
    def _extract_with_pypdf(self, pdf_content: bytes) -> str:
        """Extract text using pypdf library."""
        text_parts = []
        
        try:
            pdf_stream = BytesIO(pdf_content)
            reader = PdfReader(pdf_stream)
            
            logger.info(f"Processing PDF with {len(reader.pages)} pages")
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"[Page {page_num}]\n{page_text.strip()}\n")
                        logger.debug(f"Extracted {len(page_text)} characters from page {page_num}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
            
            full_text = "\n".join(text_parts)
            logger.info(f"Total extracted text length: {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"pypdf extraction failed: {e}")
            raise ModelError(f"PDF reading error: {str(e)}")
    
    def _extract_with_pypdf2(self, pdf_content: bytes) -> str:
        """Extract text using PyPDF2 library."""
        text_parts = []
        
        try:
            pdf_stream = BytesIO(pdf_content)
            reader = PyPDF2.PdfReader(pdf_stream)
            
            logger.info(f"Processing PDF with {len(reader.pages)} pages")
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_parts.append(f"[Page {page_num}]\n{page_text.strip()}\n")
                        logger.debug(f"Extracted {len(page_text)} characters from page {page_num}")
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num}: {e}")
                    continue
            
            full_text = "\n".join(text_parts)
            logger.info(f"Total extracted text length: {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise ModelError(f"PDF reading error: {str(e)}")
    
    def extract_with_pdfplumber(self, pdf_content:bytes) -> str:
        """
        Extract text usisng pdfplumber                               
        """
        try:
            text_parts = []
            with pdfplumber.open(BytesIO(pdf_content)) as pdf:
                logger.info(f"Processing PDF with {len(pdf.pages)} pages using pdfplumber") 

                for page_num, page in enumerate(pdf.pages, 1):
                    try:
                        text = page.extract_text()
                        if text and text.strip():
                            cleaned_text = self._basic_clean_text(text)
                            text_parts.append(f"[Page {page_num}]\n{cleaned_text}\n")
                            logger.debug(f"Extracted {len(cleaned_text)} characters from page {page_num}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}")
                        continue
            return "\n".join(text_parts)
        except ImportError:
            logger.error("pdfplumber not installed. Install with: pip install pdfplumber")
            return ""
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return ""
                             
                    

    def chunk_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """Split text into chunks for processing."""
        if not text or not text.strip():
            return []
        
        try:
            # Clean the text
            cleaned_text = self._clean_text(text)
            
            # Split into chunks
            chunks = self._split_text_smart(cleaned_text)
            
            # Convert to Document objects
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    **(metadata or {})
                }
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
            
            logger.info(f"Created {len(documents)} text chunks")
            return documents
            
        except Exception as e:
            logger.error(f"Text chunking failed: {e}")
            raise ModelError(f"Failed to chunk text: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        lines = []
        for line in text.split('\n'):
            cleaned_line = line.strip()
            if cleaned_line:
                lines.append(cleaned_line)
        
        # Join with single newlines
        cleaned = '\n'.join(lines)
        
        # Remove excessive spaces
        while '  ' in cleaned:
            cleaned = cleaned.replace('  ', ' ')
        
        return cleaned
    
    def _split_text_smart(self, text: str) -> List[str]:
        """Smart text splitting that preserves context."""
        # If text is short enough, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        # Use LangChain's text splitter
        chunks = self.text_splitter.split_text(text)
        
        # Post-process chunks to ensure they're meaningful
        processed_chunks = []
        for chunk in chunks:
            if len(chunk.strip()) > 50:  # Skip very short chunks
                processed_chunks.append(chunk.strip())
        
        return processed_chunks
    
    def extract_key_sentences(self, text: str, num_sentences: int = 5) -> List[str]:
        """Extract key sentences from text for summarization."""
        sentences = []
        
        # Split by common sentence endings
        import re
        sentence_endings = re.split(r'[.!?]+', text)
        
        for sentence in sentence_endings:
            sentence = sentence.strip()
            if len(sentence) > 20:  # Filter out very short fragments
                sentences.append(sentence)
        
        # Sort by length (longer sentences often contain more information)
        sentences.sort(key=len, reverse=True)
        
        return sentences[:num_sentences]
    
    def get_text_stats(self, text: str) -> dict:
        """Get basic statistics about the extracted text."""
        words = text.split()
        sentences = text.split('.')
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_words_per_sentence": len(words) / max(len(sentences), 1),
            "estimated_reading_time_minutes": len(words) / 200  # ~200 WPM average
        }
    
    def _basic_clean_text(self, text:str) -> str:
        """
        Basic text cleaning that works for most cases
        """
        if not text:
            return ""
        
				# Remove excessive witespace
        text = re.sub(r'\s+', ' ', text)
        
				# Remove excessive newlines but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'\n(?!\n)', ' ', text)

				# Fix punctuation spacing
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)
        return text.strip()