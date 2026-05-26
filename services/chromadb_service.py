import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from threading import Lock
from services.embedding_service import embedding_service
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("finance_app.chromadb_service")

class MemoryVectorDBFallback:
    """Thread-safe In-Memory Vector database fallback doing cosine similarity searches."""
    def __init__(self):
        self.lock = Lock()
        self.collections: Dict[str, List[Dict[str, Any]]] = {}

    def get_or_create_collection(self, name: str):
        with self.lock:
            if name not in self.collections:
                self.collections[name] = []
            return self

    def add(self, ids: List[str], embeddings: List[List[float]], documents: List[str], metadatas: List[Dict[str, Any]], collection_name: str):
        with self.lock:
            if collection_name not in self.collections:
                self.collections[collection_name] = []
            
            for idx, item_id in enumerate(ids):
                # Ensure duplicate prevention
                self.collections[collection_name] = [item for item in self.collections[collection_name] if item["id"] != item_id]
                
                self.collections[collection_name].append({
                    "id": item_id,
                    "embedding": embeddings[idx],
                    "document": documents[idx],
                    "metadata": metadatas[idx]
                })
            logger.info(f"Added {len(ids)} items to memory-vector collection '{collection_name}'.")

    def query(self, query_embeddings: List[List[float]], n_results: int, collection_name: str) -> Dict[str, Any]:
        with self.lock:
            items = self.collections.get(collection_name, [])
            if not items:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            q_emb = np.array(query_embeddings[0])
            results = []
            
            for item in items:
                i_emb = np.array(item["embedding"])
                # Compute Cosine Distance: 1.0 - Cosine Similarity
                dot_product = np.dot(q_emb, i_emb)
                q_norm = np.linalg.norm(q_emb)
                i_norm = np.linalg.norm(i_emb)
                
                if q_norm > 0 and i_norm > 0:
                    similarity = dot_product / (q_norm * i_norm)
                else:
                    similarity = 0.0
                
                distance = 1.0 - similarity
                results.append((distance, item))
            
            # Sort by ascending distance (most similar first)
            results.sort(key=lambda x: x[0])
            top_results = results[:n_results]
            
            return {
                "ids": [[item["id"] for _, item in top_results]],
                "documents": [[item["document"] for _, item in top_results]],
                "metadatas": [[item["metadata"] for _, item in top_results]],
                "distances": [[dist for dist, _ in top_results]]
            }

    def delete(self, ids: List[str], collection_name: str):
        with self.lock:
            if collection_name in self.collections:
                self.collections[collection_name] = [
                    item for item in self.collections[collection_name] if item["id"] not in ids
                ]
                logger.info(f"Deleted IDs {ids} from memory collection '{collection_name}'.")


class ChromaDBService:
    """Service interfacing with ChromaDB vector indices or a thread-safe Memory Vector database."""
    def __init__(self):
        self.is_fallback = False
        self.client = None
        self.fallback_db = MemoryVectorDBFallback()
        self.persist_directory = "./chromadb_data"
        
        # Detect remote settings or default to local persistent database
        self.host = os.getenv("CHROMA_HOST", "").strip()
        self.port = os.getenv("CHROMA_PORT", "").strip()

        try:
            import chromadb
            os.makedirs(self.persist_directory, exist_ok=True)
            
            if self.host and self.port:
                logger.info(f"Connecting to remote ChromaDB server at {self.host}:{self.port}...")
                self.client = chromadb.HttpClient(host=self.host, port=int(self.port))
            else:
                logger.info("Initializing local Persistent ChromaDB Client...")
                self.client = chromadb.PersistentClient(path=self.persist_directory)
                
            self.is_fallback = False
            logger.info("ChromaDB Client initialized successfully.")
        except ImportError:
            logger.warning("chromadb library is not installed. Bypassing with Local Vector Index Fallback.")
            self.is_fallback = True
        except Exception as e:
            logger.error(f"Failed to initialize native ChromaDB client: {e}. Activating memory fallback...")
            self.is_fallback = True

    def get_or_create_collection(self, name: str):
        """Access or instantiate a named vector store collection."""
        if self.is_fallback:
            return self.fallback_db.get_or_create_collection(name)
        try:
            return self.client.get_or_create_collection(name)
        except Exception as e:
            logger.error(f"ChromaDB failed to get/create collection '{name}': {e}. Using fallback...")
            return self.fallback_db.get_or_create_collection(name)

    def add_expense(self, expense_id: str, document_text: str, metadata: Dict[str, Any]):
        """Embed and append an expense record into the vector index."""
        logger.info(f"Indexing expense record {expense_id}...")
        vector = embedding_service.get_embedding(document_text)
        collection_name = "expense_embeddings"

        if self.is_fallback:
            self.fallback_db.add(
                ids=[expense_id],
                embeddings=[vector],
                documents=[document_text],
                metadatas=[metadata],
                collection_name=collection_name
            )
            return

        try:
            collection = self.get_or_create_collection(collection_name)
            # Add to native Chroma
            collection.add(
                ids=[expense_id],
                embeddings=[vector],
                documents=[document_text],
                metadatas=[metadata]
            )
        except Exception as e:
            logger.error(f"Failed to add expense to ChromaDB: {e}. Diverting to fallback store...")
            self.fallback_db.add(
                ids=[expense_id],
                embeddings=[vector],
                documents=[document_text],
                metadatas=[metadata],
                collection_name=collection_name
            )

    def query_expenses(self, query_text: str, user_id: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Retrieve closest vector documents matching query text, filtering by user_id."""
        logger.info(f"Querying vector database semantically: '{query_text}' for User {user_id}")
        query_vector = embedding_service.get_embedding(query_text)
        collection_name = "expense_embeddings"
        
        raw_results = None
        if self.is_fallback:
            raw_results = self.fallback_db.query(
                query_embeddings=[query_vector],
                n_results=n_results,
                collection_name=collection_name
            )
        else:
            try:
                collection = self.get_or_create_collection(collection_name)
                # Query native ChromaDB. We retrieve double the results to manually filter by user_id
                # (to keep it completely robust and error-free on custom metadata setups)
                raw_results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=n_results * 2,
                )
            except Exception as e:
                logger.error(f"Native ChromaDB query failed: {e}. Using fallback...")
                raw_results = self.fallback_db.query(
                    query_embeddings=[query_vector],
                    n_results=n_results,
                    collection_name=collection_name
                )

        # Parse and filter documents matching user_id
        parsed_results = []
        if raw_results and "ids" in raw_results and raw_results["ids"]:
            ids = raw_results["ids"][0]
            docs = raw_results["documents"][0]
            metas = raw_results["metadatas"][0]
            dists = raw_results.get("distances", [[]])[0]
            
            for idx, item_id in enumerate(ids):
                meta = metas[idx]
                # Filter by active User ID
                if meta.get("user_id") == user_id:
                    distance = dists[idx] if idx < len(dists) else 0.0
                    similarity = 1.0 - distance
                    parsed_results.append({
                        "id": item_id,
                        "text": docs[idx],
                        "metadata": meta,
                        "similarity": similarity
                    })
                
                # Truncate at user's desired limit
                if len(parsed_results) >= n_results:
                    break
                    
        return parsed_results

    def delete_expense(self, expense_id: str):
        """Remove a transaction embedding from the vector indexes."""
        collection_name = "expense_embeddings"
        if self.is_fallback:
            self.fallback_db.delete(ids=[expense_id], collection_name=collection_name)
            return
        try:
            collection = self.get_or_create_collection(collection_name)
            collection.delete(ids=[expense_id])
        except Exception as e:
            logger.error(f"Native ChromaDB deletion failed: {e}. Directing delete to fallback...")
            self.fallback_db.delete(ids=[expense_id], collection_name=collection_name)

# Singleton Instance
chromadb_service = ChromaDBService()
