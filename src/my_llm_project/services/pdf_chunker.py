import pdfplumber
import re
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PDFVectorService:
    def __init__(self, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        """Initialize the PDF vector service"""
        logger.info("Initializing PDF Vector Service...")
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name="pdf_chunks",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("PDF Vector Service initialized successfully")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using pdfplumber"""
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
        return text.strip()
    
    def chunk_by_sentences(self, text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
          Chunk text by sentences with overlap
          Good for maintaining semantic coherence
    
          Args:
          text: Text to chunk
          max_chunk_size: Maximum size of each chunk in characters
          overlap: Number of characters to overlap between chunks
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    
        chunks = []
        current_chunk = ""
    
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
        
            # Try to add sentence to current chunk
            if not current_chunk:
                # First sentence in chunk
                current_chunk = sentence
                i += 1
            elif len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                # Sentence fits in current chunk
                current_chunk += " " + sentence
                i += 1
            else:
                # Current chunk is full, save it
                chunks.append(current_chunk.strip())
            
                # Create overlap for next chunk
                overlap_text = ""
                overlap_size = 0
            
                # Go backwards from current position to collect overlap
                j = i - 1
                while j >= 0 and overlap_size < overlap:
                    sentence_to_add = sentences[j]
                    if overlap_size + len(sentence_to_add) <= overlap:
                        overlap_text = sentence_to_add + (" " + overlap_text if overlap_text else "")
                        overlap_size += len(sentence_to_add) + 1
                        j -= 1
                    else:
                        break
            
                # Start new chunk with overlap
                current_chunk = overlap_text
    
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
    
        return chunks
    def chunk_by_paragraphs(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Chunk text by paragraphs with overlap"""
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                words = current_chunk.split()
                overlap_words = words[-overlap//5:] if len(words) >= overlap//5 else words
                current_chunk = ' '.join(overlap_words) + "\n\n" + paragraph
            else:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def chunk_by_sliding_window(self, text: str, window_size: int = 800, step_size: int = 400) -> List[str]:
        """Sliding window chunking"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), step_size):
            chunk_words = words[i:i + window_size]
            if len(chunk_words) < 50:
                break
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
        
        return chunks
    
    def create_embeddings(self, chunks: List[str]) -> List[List[float]]:
        """Create embeddings for chunks using sentence transformer"""
        embeddings = self.model.encode(chunks, convert_to_tensor=False)
        return embeddings.tolist()
    
    def store_in_vectordb(self, chunks: List[str], embeddings: List[List[float]], 
                         pdf_filename: str, chunking_method: str):
        """Store chunks and embeddings in ChromaDB"""
        ids = [f"{pdf_filename}_{chunking_method}_{i}" for i in range(len(chunks))]
        
        metadatas = [
            {
                "source": pdf_filename,
                "chunk_id": i,
                "chunking_method": chunking_method,
                "text_length": len(chunk)
            } 
            for i, chunk in enumerate(chunks)
        ]
        
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Stored {len(chunks)} chunks from {pdf_filename}")
    
    def process_pdf(self, pdf_path: str, pdf_filename: str, chunking_method: str = "sentences", **kwargs):
        """
        Main processing method
        
        Args:
            pdf_path: Path to PDF file
            pdf_filename: Name of the PDF file
            chunking_method: 'sentences', 'paragraphs', or 'sliding_window'
            **kwargs: Additional parameters for chunking methods
            
        Returns:
            Dictionary with processing results
        """
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        text = self.clean_text(text)
        
        if not text:
            raise ValueError("No text could be extracted from PDF")
        
        # Choose chunking method
        if chunking_method == "sentences":
            chunks = self.chunk_by_sentences(text, **kwargs)
        elif chunking_method == "paragraphs":
            chunks = self.chunk_by_paragraphs(text, **kwargs)
        elif chunking_method == "sliding_window":
            chunks = self.chunk_by_sliding_window(text, **kwargs)
        else:
            raise ValueError("Invalid chunking_method. Must be 'sentences', 'paragraphs', or 'sliding_window'")
        
        if not chunks:
            raise ValueError("No chunks were created from the text")
        
        # Create embeddings
        embeddings = self.create_embeddings(chunks)
        
        # Store in vector database
        self.store_in_vectordb(chunks, embeddings, pdf_filename, chunking_method)
        
        return {
            "num_chunks": len(chunks),
            "total_characters": len(text),
            "chunking_method": chunking_method,
            "avg_chunk_size": sum(len(c) for c in chunks) // len(chunks)
        }
    
    def search_similar(self, query: str, n_results: int = 5, source_filter: Optional[str] = None) -> Dict:
        """
        Search for similar chunks based on query
        
        Args:
            query: Search query string
            n_results: Number of results to return
            source_filter: Optional filter by PDF filename
            
        Returns:
            Dictionary with search results
        """
        # Create embedding for query
        query_embedding = self.model.encode([query], convert_to_tensor=False)[0]
        
        # Build where clause for filtering
        where = {"source": source_filter} if source_filter else None
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
            include=['documents', 'metadatas', 'distances']
        )
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about stored documents"""
        collection_count = self.collection.count()
        return {
            "total_chunks": collection_count
        }
    
    def delete_by_source(self, pdf_filename: str) -> int:
        """
        Delete all chunks from a specific PDF file
        
        Args:
            pdf_filename: Name of the PDF file
            
        Returns:
            Number of chunks deleted
        """
        # Get all IDs for this source
        results = self.collection.get(
            where={"source": pdf_filename},
            include=[]
        )
        
        if results['ids']:
            self.collection.delete(ids=results['ids'])
            logger.info(f"Deleted {len(results['ids'])} chunks from {pdf_filename}")
            return len(results['ids'])
        
        return 0
