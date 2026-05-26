import os
import logging
import hashlib
import numpy as np
from typing import List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("finance_app.embedding_service")

class EmbeddingService:
    """Generates dense vector representations for semantic search and RAG contexts."""
    def __init__(self):
        self.use_gemini = False
        self.use_local_transformers = False
        
        # Test Gemini embedding model
        gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if gemini_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                self.use_gemini = True
                logger.info("Embedding service configured with Gemini API (text-embedding-004).")
            except Exception as e:
                logger.error(f"Failed to configure Gemini embeddings: {e}")

        # If Gemini is not set, try local sentence-transformers
        if not self.use_gemini:
            try:
                from sentence_transformers import SentenceTransformer
                self.model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                # Load lazily to avoid holding Streamlit startup
                self.transformer_model = None 
                self.use_local_transformers = True
                logger.info(f"Embedding service configured with local SentenceTransformers ({self.model_name}).")
            except ImportError:
                logger.warning("sentence-transformers package is missing. Fallback local hash embeddings will be used.")
            except Exception as e:
                logger.error(f"Failed to initialize sentence-transformers: {e}")

    def get_embedding(self, text: str) -> List[float]:
        """Generate high-dimensional vector for a string."""
        if not text.strip():
            # Return empty/zero vector
            return [0.0] * 384

        # 1. Try Gemini API
        if self.use_gemini:
            try:
                import google.generativeai as genai
                response = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                embedding = response.get("embedding", [])
                if embedding:
                    return embedding
            except Exception as e:
                logger.error(f"Gemini Cloud Embedding Generation failed: {e}. Transitioning...")

        # 2. Try local SentenceTransformers
        if self.use_local_transformers:
            try:
                from sentence_transformers import SentenceTransformer
                if self.transformer_model is None:
                    # Lazy loading
                    logger.info(f"Loading local SentenceTransformer model: {self.model_name}...")
                    self.transformer_model = SentenceTransformer(self.model_name)
                
                vector = self.transformer_model.encode(text)
                return vector.tolist()
            except Exception as e:
                logger.error(f"Local SentenceTransformer Embedding Generation failed: {e}. Utilizing offline hash fallback...")

        # 3. Ultimate robust offline mock hash fallback (TF-IDF mimic)
        return self._generate_hash_embedding(text)

    def _generate_hash_embedding(self, text: str, dimension: int = 384) -> List[float]:
        """
        Deterministic hash-based vector fallback representing words in high-dimensional space.
        Enables structural search similarities without neural network runtime requirements.
        """
        words = text.lower().split()
        vector = np.zeros(dimension)
        
        for word in words:
            # Hash each word into a deterministic index
            h = hashlib.sha256(word.encode('utf-8')).hexdigest()
            # Map index and seed to deterministic coordinates
            seed = int(h[:8], 16)
            np.random.seed(seed)
            # Add normal distribution unit vectors
            word_vec = np.random.normal(0, 0.1, dimension)
            vector += word_vec

        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        return vector.tolist()

# Singleton Instance
embedding_service = EmbeddingService()
