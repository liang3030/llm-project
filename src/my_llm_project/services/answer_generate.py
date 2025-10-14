import logging
from typing import Dict, List, Optional

from my_llm_project.services.pdf_chunker import PDFVectorService


logger = logging.getLogger(__name__)


class AnswerGenerationService:
	"""Service for generating answers based on retrieved context"""

	def __init__(self, vector_service: PDFVectorService, llm_client=None):
		"""
			Initialize Answer Generation Service

			Args:
				vector_service: Instance of PDFVectorService for retrieving context
				llm_client: LLM client with generate_answer(prompt) method
		"""

		self.vector_service = vector_service
		self.llm_client = llm_client
		logger.info("Answer Generation Service initialized")

	def generate_answer(
			self,
			question: str,
			n_context_chunks: int = 5,
			source_filter: Optional[str] = None,
			include_sources: bool = True,
			temperature: float = 0.3
	) -> Dict:
		"""
		Generate answer to a question using RAG approach
		Args:
			question: User's question
			n_context_chunks: Number of relevant chunks to retrieve
			source_filter: Optional filter to search in specific PDF
			include_sources: Whether to include source information in response
			temperature: LLM temperature (0.0 - 1.0, lower is more focused)

		Returns:
			Dictinoary containing:
				- answer: Generated answer text
				- sources: List of source information (if include_sources = True)
				- context_chunks: Retrieved context chunks
				- confidence: Confidence score based on relevance
		"""
		logger.info(f"Generating answer for question: {question[:50]}")

		# Step1: Retrieve relevant context
		search_results = self._retrieve_context(
			question,
			n_context_chunks,
			source_filter
		)
		if not search_results['chunks']:
			return self._no_context_response()
		
		# Step 2: Build prompt with context
		prompt = self._build_prompt(question, search_results['chunks'])

		# Step 3: Generate answer using LLM
		if self.llm_client:
			answer = self._generate_with_llm(prompt, temperature)
		else:
			answer = self._generate_fallback_answer(search_results['chunks'])

		# Step 4: Calculate confidence score
		confidence = self._calculate_confidence(search_results['distances'])

		# Step 5: Prepare response
		response = {
			'answer': answer,
			'confidence': confidence,
			'num_chunks_used': len(search_results['chunks'])
		}

		if include_sources:
			response['sources'] = self._format_sources(
                search_results['metadatas'],
                search_results['distances']
            )
			response['context_chunks'] = search_results['chunks']
		
		logger.info(f"Answer generated successfully (confidence: {confidence: .2f})")
		return response
	
	def generate_answer_with_history(
			self,
			question: str,
			conversation_history: List[Dict],
			n_context_chunks: int = 5,
			source_filter: Optional[str] = None
	) -> Dict:
		"""
		Generate answer considering conversation history

		Args:
			question: Current question
			conversation_history: List of {"role": "user"/"assistant", "content": "..."}
			n_context_chunks: Number of chunks to retrieve
			source_filter: Optional PDF filename filter

		Returns:
			Same as generate_answer()
		"""

		# Enhance question with conversation context
		enhanced_question = self._enhance_with_history(question, conversation_history)
        
    # Generate answer using enhanced question
		return self.generate_answer(
            enhanced_question,
            n_context_chunks,
            source_filter
        )
	
	def _retrieve_context(
			self,
			question: str,
			n_chunks: int,
			source_filter: Optional[str]
	) -> Dict:
		"""Retrieve relevatn context chunks from vector database"""
		try:
			search_results = self.vector_service.search_similar(
				query = question,
				n_results = n_chunks,
				source_filter= source_filter
			)
			return {
				'chunks': search_results['documents'][0] if search_results['documents'] else [],
        'metadatas': search_results['metadatas'][0] if search_results['metadatas'] else [],
        'distances': search_results['distances'][0] if search_results['distances'] else []
			}
		except Exception as e:
			logger.error(f"Error retrieving context: {e}")
			return {'chunks':[], 'metadatas': [], 'distances': []}
		
	def _build_prompt(self, question: str, chunks: List[str]) -> str:
		"""Build prompt for LLM with context"""

		# Format context with chunk numbers
		context_parts = []
		for i, chunk in enumerate(chunks, 1):
			context_parts.append(f"[Context {i}]\n{chunk}")

		context = "\n\n".join(context_parts)
		prompt = f"""
			You are a helpful assistant thta answers questions base on provided context.
			CONTEXT:
			{context}

			QUESTION:
			{question}

			INSTRUCTION:
				- Answer in English, even if the context is in German or another language
				- Base your answer ONLY on the information provided in the context
				- If the context doesn't contain enough information to answer the question, say so clearly
				- Be concise and accurate
				- Reference which context sections you used (e.g., "According to Context 1...")
				- If you're uncertain, express that appropriately

			ANSWER: 
		"""
		return prompt
	
	def _generate_with_llm(
			self,
			prompt: str,
			temperature: float
	) -> str:
		try:
			# Check if llm_client has generate_answer method.
			if hasattr(self.llm_client, 'generate_answer'):
				return self.llm_client.generate_answer(prompt)
			
			# log warning message
			else:
				logger.warning("LLM client method not recognized, using fallback")
				return self._generate_fallback_answer([prompt])
			
		except Exception as e:
			logger.error(f"Error generating answer with LLM: {e}")
			return f"Error generating answer: {str(e)}"
	
	def _generate_fallback_answer(self, chunks: List[str]) -> str:
		if not chunks:
			return "No relevant information found in documents"
		
		answer = "Based on the uploaded documents, here is the relevant information:\n\n"

		# Include top 3 chunks
		for i, chunk in enumerate(chunks[:3], 1):
			answer += f"**Excerpt {i}: **\n{chunk}\n\n"

		if len(chunks) > 3:
			answer += f"(+ {len(chunks) - 3} more relevant sections found)"

		return answer
	
	# def _no_context_response(self) -> Dict:
