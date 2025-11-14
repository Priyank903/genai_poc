"""
Improved RAG system with Sarvam AI integration and web search fallback
"""

import os
import re
import json
import logging
import pickle
from typing import List, Dict, Optional, Tuple
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from dotenv import load_dotenv

try:
    from sarvamai import SarvamAI
except ImportError:  # pragma: no cover - handled gracefully at runtime
    SarvamAI = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"

# Basic stopword list for keyword extraction
STOPWORDS = {
    "the", "and", "that", "have", "with", "this", "about", "your", "from",
    "what", "would", "could", "should", "there", "which", "their", "been",
    "when", "where", "into", "over", "under", "than", "then", "also",
    "just", "into", "onto", "they", "them", "been", "will", "here", "know",
    "need", "more", "less", "very", "like", "have", "much", "some", "such",
    "does", "doesn't", "don't", "can't", "aren't", "isn't", "i'm", "i'll",
    "you're", "we're", "he's", "she's", "it's", "for", "have", "washer"
}

# Phrases indicating the LLM could not use the provided context
INSUFFICIENT_CONTEXT_PHRASES = [
    "does not contain relevant information",
    "context does not contain",
    "references do not cover",
    "no relevant information",
    "insufficient context"
]

class ImprovedRAGSystem:
    """Enhanced RAG system with Sarvam AI and web search fallback"""
    
    def __init__(self, index_dir: str = None, sarvam_api_key: str = None):
        """Initialize RAG system"""
        if index_dir is None:
            index_dir = self._get_index_dir()

        # Load environment variables (once per initialization)
        load_dotenv()

        self.index_dir = index_dir
        self.sarvam_api_key = sarvam_api_key or os.getenv("SARVAM_API_KEY")
        self.sarvam_model = os.getenv("SARVAM_MODEL", "ai21labs/jamba-1.5-large")
        self.sarvam_client = None

        # Load FAISS index and metadata
        self._load_index()

        # Initialize sentence transformer for query encoding
        self.model = SentenceTransformer(MODEL_NAME)

        # Initialize Sarvam AI client when possible
        if self.sarvam_api_key and SarvamAI:
            try:
                self.sarvam_client = SarvamAI(api_subscription_key=self.sarvam_api_key)
                logger.info("Sarvam AI client initialized successfully")
            except Exception as exc:  # pragma: no cover - network issue
                logger.error(f"Failed to initialize Sarvam AI client: {exc}")
                self.sarvam_client = None
        else:
            if not SarvamAI:
                logger.warning("sarvamai package not installed. Install with `pip install sarvamai`")
            if not self.sarvam_api_key:
                logger.warning("Sarvam API key not provided; RAG will fall back to web search or direct chunks")

        logger.info("RAG system initialized successfully")
    
    def _get_index_dir(self) -> str:
        """Get the embeddings index directory"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        index_dir = os.path.join(base_dir, "embeddings", "nephro_faiss")
        
        # Fallback to relative path
        if not os.path.exists(index_dir):
            index_dir = "../embeddings/nephro_faiss"
        
        return index_dir
    
    def _load_index(self):
        """Load FAISS index and metadata"""
        index_path = os.path.join(self.index_dir, "nephro_index.faiss")
        metadata_path = os.path.join(self.index_dir, "nephro_metadata.pkl")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. "
                "Please run improved_embeddings.py first."
            )
        
        self.index = faiss.read_index(index_path)
        
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded FAISS index with {len(self.metadata)} chunks")
    
    def _encode_query(self, query: str) -> np.ndarray:
        """Encode query using the same model as embeddings"""
        embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(embedding)  # Normalize for cosine similarity
        return embedding
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3, min_similarity: float = 0.3) -> List[Dict]:
        """
        Retrieve top-k most relevant chunks for the query
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant chunks with metadata and scores
        """
        logger.info(f"Retrieving chunks for query: '{query[:50]}...'")
        
        # Encode query
        query_embedding = self._encode_query(query)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score >= min_similarity:  # Filter by minimum similarity
                chunk = self.metadata[idx]
                result = {
                    "id": chunk.get("chunk_id", idx),
                    "page": chunk.get("page", 0),
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", "nephrology-reference"),
                    "similarity_score": float(score),
                    "paragraph_index": chunk.get("paragraph_index", 0)
                }
                results.append(result)
        
        logger.info(f"Retrieved {len(results)} relevant chunks (similarity >= {min_similarity})")
        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from query"""
        if not text:
            return []
        words = re.findall(r"[a-zA-Z]+", text.lower())
        return [w for w in words if len(w) > 3 and w not in STOPWORDS]

    def _is_chunk_relevant(self, keywords: List[str], chunk: Dict) -> bool:
        """Check if a chunk contains keywords or has high similarity"""
        if not keywords:
            return True
        chunk_text = chunk["text"].lower()
        if chunk.get("similarity_score", 0) >= 0.6:
            return True
        return any(keyword in chunk_text for keyword in keywords)

    def _filter_relevant_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Filter chunks to those likely relevant to the query"""
        keywords = self._extract_keywords(query)
        filtered = [chunk for chunk in chunks if self._is_chunk_relevant(keywords, chunk)]
        logger.info(
            "Filtered chunks based on relevance: %s of %s kept",
            len(filtered),
            len(chunks)
        )
        return filtered
    
    def format_context_for_llm(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks as context for LLM"""
        if not chunks:
            return ""
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            citation = f"[Source: {chunk['source']}, Page {chunk['page']}]"
            context_parts.append(f"Context {i}: {chunk['text']}\n{citation}")
        
        return "\n\n".join(context_parts)

    def _log_retrieved_chunks(self, label: str, chunks: List[Dict]) -> None:
        """Log retrieved chunk information for debugging"""
        if not chunks:
            logger.info(f"{label}: no chunks after filtering")
            return
        for idx, chunk in enumerate(chunks, 1):
            snippet = chunk['text'][:200].replace("\n", " ")
            logger.info(
                "%s [%d]: page=%s score=%.3f snippet=%s",
                label,
                idx,
                chunk.get('page'),
                chunk.get('similarity_score', 0.0),
                snippet
            )

    def _build_chunk_summary(self, chunks: List[Dict]) -> str:
        """Build a textual summary directly from retrieved chunks"""
        parts = ["Key excerpts from nephrology references:"]
        for idx, chunk in enumerate(chunks, 1):
            parts.append(
                f"\nReference {idx}: (Source: {chunk['source']}, Page {chunk['page']}, Similarity: {chunk.get('similarity_score', 0.0):.3f})\n"
                f"{chunk['text']}"
            )
        return "\n".join(parts)

    def _summarize_chunks_for_query(self, query: str, chunks: List[Dict], max_points: int = 5) -> str:
        """Generate a concise, citation-ready summary from relevant chunks"""
        if not chunks:
            return "No supporting medical context available."

        keywords = self._extract_keywords(query)
        summary_lines = []

        for chunk in chunks:
            sentences = re.split(r'(?<=[.!?])\s+', chunk['text'])
            for sentence in sentences:
                normalized = sentence.strip()
                if not normalized:
                    continue
                sentence_lower = normalized.lower()
                if keywords and not any(keyword in sentence_lower for keyword in keywords):
                    continue
                line = (
                    f"- {normalized} "
                    f"(Source: {chunk['source']}, Page {chunk['page']}, Similarity: {chunk.get('similarity_score', 0.0):.3f})"
                )
                summary_lines.append(line)
                if len(summary_lines) >= max_points:
                    break
            if len(summary_lines) >= max_points:
                break

        if not summary_lines:
            # Fallback to first sentences if no keyword-aligned content found
            for chunk in chunks:
                sentences = re.split(r'(?<=[.!?])\s+', chunk['text'])
                for sentence in sentences[:2]:
                    normalized = sentence.strip()
                    if normalized:
                        line = (
                            f"- {normalized} "
                            f"(Source: {chunk['source']}, Page {chunk['page']}, Similarity: {chunk.get('similarity_score', 0.0):.3f})"
                        )
                        summary_lines.append(line)
                        if len(summary_lines) >= max_points:
                            break
                if len(summary_lines) >= max_points:
                    break

        header = "Based on the retrieved nephrology references:\n"
        return header + "\n".join(summary_lines)
    
    def query_sarvam_ai(self, query: str, context: str) -> Dict:
        """
        Query Sarvam AI with RAG context
        
        Args:
            query: User question
            context: Retrieved context from RAG
            
        Returns:
            Response from Sarvam AI
        """
        if not self.sarvam_client:
            logger.error("Sarvam AI client not available (missing key or dependency)")
            return {"error": "Sarvam AI client not configured"}

        if not context.strip():
            logger.warning("No valid context provided for Sarvam AI request")
            return {"error": "No context available for Sarvam AI"}

        # Construct prompt with context
        system_prompt = """You are a clinical nephrology AI assistant. Answer questions based ONLY on the provided medical context from nephrology textbooks. 

Instructions:
1. STRICTLY use ONLY the information provided in the context - do not use your internal knowledge
2. If the provided context contains ANY relevant information that helps answer the question, provide a comprehensive answer based on that context
3. ONLY respond with "The provided nephrology references do not contain sufficient information to answer this question" if the context is completely unrelated to the question or contains no useful information whatsoever
4. Always cite the source and page when using information from the context
5. Be thorough and detailed when the context supports the answer
6. Focus on the most relevant parts of the context that directly answer the question
7. IMPORTANT: If the context contains the exact information being asked about, or directly relevant medical information, USE IT to answer - do not claim it's insufficient

CRITICAL: Be generous in determining if context is helpful. If the context mentions the topic, anatomy, or medical concepts in the question, it IS sufficient to provide an answer. Only claim insufficient information if the context is completely unrelated to the medical question."""

        user_prompt = f"""Context from nephrology references:
{context}

Question: {query}

Please answer based on the provided context. If the context doesn't contain relevant information for this question, please state that clearly."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            # Primary call with explicit model
            try:
                response = self.sarvam_client.chat.completions(
                    messages=messages,
                    model=self.sarvam_model,
                    temperature=0.1,
                )
            except TypeError:
                # Some client versions set model server-side
                response = self.sarvam_client.chat.completions(
                    messages=messages,
                    temperature=0.1,
                )

            # Extract content from various possible response structures
            answer_text = None
            if isinstance(response, dict):
                choices = response.get("choices", [])
                if choices:
                    first_choice = choices[0]
                    if isinstance(first_choice, dict):
                        message = first_choice.get("message", {})
                        answer_text = message.get("content") or first_choice.get("text")
            elif hasattr(response, "choices"):
                choices = getattr(response, "choices")
                if choices:
                    first_choice = choices[0]
                    if hasattr(first_choice, "message") and first_choice.message:
                        answer_text = getattr(first_choice.message, "content", None)
                    elif hasattr(first_choice, "text"):
                        answer_text = first_choice.text
            elif hasattr(response, "text"):
                answer_text = response.text
            elif isinstance(response, str):
                answer_text = response

            if not answer_text:
                logger.error(f"Sarvam AI returned no content. Raw response: {response}")
                return {"error": "Sarvam AI response contained no content"}

            return {
                "answer": answer_text,
                "model": "sarvam-ai",
                "context_used": True,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as exc:
            logger.error(f"Error calling Sarvam AI: {exc}")
            return {"error": f"Failed to call Sarvam AI: {exc}"}
    
    def web_search_with_llm(self, query: str) -> Dict:
        """
        Perform web search and use Sarvam AI to process the results
        
        Args:
            query: User question
            
        Returns:
            LLM-processed web search results
        """
        try:
            from web_search import WebSearchTool
            
            search_tool = WebSearchTool()
            # Create focused medical search query
            medical_query = self._create_medical_search_query(query)
            
            results = search_tool.search(medical_query, max_results=5)
            
            if results:
                # Format web search results as context
                web_context = "\n\n".join([
                    f"Source {i+1}: {r.get('title', 'Unknown')}\nURL: {r.get('url', '')}\nContent: {r.get('snippet', r.get('content', ''))}"
                    for i, r in enumerate(results[:5])
                ])
                
                # Use Sarvam AI to process web search results
                web_system_prompt = """You are a clinical nephrology AI assistant. Answer questions based ONLY on the provided web search results about medical topics.

Instructions:
1. STRICTLY use ONLY the information provided in the web search results - do not use your internal knowledge
2. Synthesize information from multiple sources when relevant
3. Always cite the sources by their titles and URLs when using information
4. Be thorough and detailed when the web results support the answer
5. If the web results don't contain sufficient information, state that clearly
6. Focus on the most relevant and credible sources

CRITICAL: Never use information not present in the provided web search results. Always ground your answer in the specific sources provided."""

                web_user_prompt = f"""Web search results for medical question:
{web_context}

Question: {query}

Please answer based on the provided web search results. Cite sources appropriately and provide a comprehensive answer if the results contain relevant information."""

                web_messages = [
                    {"role": "system", "content": web_system_prompt},
                    {"role": "user", "content": web_user_prompt}
                ]

                if self.sarvam_client:
                    try:
                        web_llm_response = self.sarvam_client.chat.completions(
                            messages=web_messages,
                            max_tokens=1000,
                            temperature=0.1
                        )
                        
                        return {
                            "answer": web_llm_response.choices[0].message.content,
                            "web_results": results,
                            "results_count": len(results),
                            "timestamp": datetime.now().isoformat(),
                            "processed_by_llm": True
                        }
                    except Exception as e:
                        logger.error(f"Sarvam AI processing of web results failed: {e}")
                        # Fallback to basic web search formatting
                        return self.web_search_fallback(query)
                else:
                    # Fallback to basic web search formatting
                    return self.web_search_fallback(query)
            else:
                return {"error": "No web search results found"}
                
        except Exception as e:
            logger.error(f"Web search with LLM failed: {str(e)}")
            return {"error": f"Web search with LLM failed: {str(e)}"}

    def web_search_fallback(self, query: str) -> Dict:
        """
        Perform web search when RAG context is insufficient
        
        Args:
            query: User question
            
        Returns:
            Web search results
        """
        try:
            from web_search import WebSearchTool
            
            search_tool = WebSearchTool()
            # Create focused medical search query
            medical_query = self._create_medical_search_query(query)
            
            results = search_tool.search(medical_query, max_results=3)
            
            if results:
                # Format web search results
                web_context = "\n\n".join([
                    f"Source: {r.get('title', 'Unknown')}\nURL: {r.get('url', '')}\nContent: {r.get('snippet', r.get('content', ''))}"
                    for r in results[:3]
                ])
                
                return {
                    "answer": f"Based on web search results:\n\n{web_context}",
                    "source": "web_search",
                    "results_count": len(results),
                    "timestamp": datetime.now().isoformat(),
                    "disclaimer": "Information from web search. Please verify with healthcare professionals."
                }
            else:
                return {"error": "No web search results found"}
                
        except Exception as e:
            logger.error(f"Web search failed: {str(e)}")
            return {"error": f"Web search failed: {str(e)}"}
    
    def _create_medical_search_query(self, query: str) -> str:
        """
        Create a focused medical search query from user input
        
        Args:
            query: Original user query (may include patient context)
            
        Returns:
            Optimized search query for medical information
        """
        import re
        
        # Extract the actual question from patient context if present
        if "Question:" in query:
            actual_question = query.split("Question:")[-1].strip()
        else:
            actual_question = query
        
        # Always clean the question of patient-specific context for web search
        cleaned_question = re.sub(r'Patient context:.*?Question:', '', actual_question, flags=re.DOTALL)
        cleaned_question = re.sub(r'Current medications:.*?Question:', '', cleaned_question, flags=re.DOTALL)
        cleaned_question = cleaned_question.strip()
        
        # Use the cleaned question for search query creation
        question_to_use = cleaned_question if cleaned_question else actual_question
        
        # Common medical symptom keywords to prioritize
        symptom_keywords = [
            'swelling', 'edema', 'pain', 'fever', 'nausea', 'vomiting', 
            'headache', 'dizziness', 'fatigue', 'shortness of breath',
            'chest pain', 'leg swelling', 'ankle swelling', 'bloating',
            'kidney', 'renal', 'nephrology', 'dialysis', 'transplant'
        ]
        
        # Extract key medical terms from the cleaned question
        question_lower = question_to_use.lower()
        found_symptoms = [keyword for keyword in symptom_keywords if keyword in question_lower]
        
        if found_symptoms:
            # Create search query focused on the symptoms
            primary_symptom = found_symptoms[0]
            if 'swelling' in question_lower or 'edema' in question_lower:
                if 'leg' in question_lower or 'ankle' in question_lower:
                    # This pattern works well - returns Mayo Clinic results
                    search_query = "edema leg swelling kidney disease"
                else:
                    search_query = "edema swelling causes kidney disease medical"
            elif 'pain' in question_lower:
                search_query = f"{primary_symptom} causes medical treatment kidney disease"
            else:
                search_query = f"{primary_symptom} medical causes treatment kidney disease"
        else:
            # Fallback: use the cleaned question with medical context
            # Special handling for drug/medication queries
            if 'topiramate' in question_lower or 'medication' in question_lower or 'drug' in question_lower:
                # Extract drug name if present
                if 'topiramate' in question_lower:
                    search_query = "topiramate uses medical indications treatment"
                else:
                    search_query = f"{question_to_use} medication uses medical treatment"
            else:
                search_query = f"medical advice {question_to_use} symptoms causes treatment"
        
        logger.info(f"Created medical search query: '{search_query}' from original: '{question_to_use[:50]}...'")
        return search_query
    
    def answer_query(self, query: str, use_web_fallback: bool = True) -> Dict:
        """
        Complete RAG pipeline: retrieve -> generate answer -> fallback if needed
        
        Args:
            query: User question
            use_web_fallback: Whether to use web search if RAG context is insufficient
            
        Returns:
            Complete response with answer and metadata
        """
        logger.info(f"Processing query: '{query}'")
        
        # Step 1: Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=5, min_similarity=0.1)
        filtered_chunks = self._filter_relevant_chunks(query, relevant_chunks)
        self._log_retrieved_chunks("Filtered medical references", filtered_chunks)

        response = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "chunks_found": len(filtered_chunks),
            "chunks_retrieved": len(relevant_chunks),
            "relevant_chunks": filtered_chunks
        }

        if filtered_chunks:
            # Step 2: Format context and query Sarvam AI
            context = self.format_context_for_llm(filtered_chunks)
            sarvam_response = self.query_sarvam_ai(query, context)
            
            if "error" not in sarvam_response:
                # Check if Sarvam AI indicates insufficient context
                answer_text = sarvam_response["answer"].lower()
                insufficient_indicators = [
                    "do not contain sufficient information",
                    "does not contain relevant information", 
                    "insufficient information",
                    "not enough information",
                    "cannot answer based on the provided context"
                ]
                
                context_insufficient = any(indicator in answer_text for indicator in insufficient_indicators)
                
                if not context_insufficient:
                    # RAG context is sufficient - use Sarvam's answer
                    response.update({
                        "answer": sarvam_response["answer"],
                        "source": "rag_with_sarvam",
                        "relevant_chunks": filtered_chunks,
                        "context_used": context
                    })
                    logger.info("Successfully answered using RAG + Sarvam AI")
                    return response
                else:
                    # Context insufficient - trigger web search fallback
                    logger.info("Sarvam AI indicated insufficient RAG context, triggering web search")
                    if use_web_fallback:
                        web_response = self.web_search_with_llm(query)
                        if "error" not in web_response:
                            # Include both RAG chunks and web search results
                            response.update({
                                "answer": web_response["answer"],
                                "source": "web_search_with_rag_context",
                                "relevant_chunks": filtered_chunks,
                                "web_search_results": web_response.get("web_results", []),
                                "rag_context_insufficient": True,
                                "rag_answer": sarvam_response["answer"]
                            })
                            logger.info("Successfully answered using web search after RAG context insufficient")
                            return response
            else:
                logger.warning(f"Sarvam AI error: {sarvam_response['error']}")
            
            # Step 2.5: If Sarvam AI fails (error), provide chunk excerpts
            if sarvam_response.get("error"):
                basic_answer = self._build_chunk_summary(filtered_chunks)
                response.update({
                    "answer": basic_answer,
                    "source": "rag_chunks_only",
                    "relevant_chunks": filtered_chunks,
                    "context_used": context,
                    "api_error": sarvam_response["error"]
                })
                logger.info("Provided answer using RAG chunks only (Sarvam AI unavailable)")
                return response

        # Step 3: Fallback to web search if RAG failed or no relevant chunks
        if use_web_fallback:
            logger.info("Falling back to web search")
            web_response = self.web_search_with_llm(query)
            
            if "error" not in web_response:
                response.update({
                    "answer": web_response["answer"],
                    "source": "web_search_fallback",
                    "web_search_results": web_response.get("web_results", []),
                    "relevant_chunks": filtered_chunks,  # ALWAYS include RAG chunks
                    "processed_by_llm": True
                })
                logger.info("Successfully answered using web search fallback with LLM processing")
                return response
        
        # Step 4: No answer available
        response.update({
            "answer": "I couldn't find relevant information to answer your question. Please consult with a healthcare professional for medical advice.",
            "source": "no_answer",
            "error": "Insufficient context and web search failed"
        })
        
        logger.warning("Could not provide answer through any method")
        return response


def main():
    """Test the RAG system"""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Initialize RAG system
    rag = ImprovedRAGSystem()
    
    # Test queries
    test_queries = [
        "What is chronic kidney disease?",
        "How does the nephron function?",
        "What are the symptoms of kidney stones?",
        "Latest research on SGLT2 inhibitors"  # This should trigger web search
    ]
    
    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        print('='*50)
        
        response = rag.answer_query(query)
        
        print(f"Source: {response['source']}")
        print(f"Chunks found: {response['chunks_found']}")
        print(f"Answer: {response['answer'][:200]}...")
        
        if 'relevant_chunks' in response:
            print(f"Relevant chunks: {len(response['relevant_chunks'])}")


if __name__ == "__main__":
    main()
