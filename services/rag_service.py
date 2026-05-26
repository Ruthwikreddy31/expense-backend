import logging
from typing import List, Dict, Any
from services.chromadb_service import chromadb_service
from database.mongodb import db_manager
from agents.chat_agent import chat_agent

logger = logging.getLogger("finance_app.rag_service")

class RAGService:
    """Orchestrates Retrieval-Augmented Generation flows for financial contexts."""
    
    def process_chat_query(self, user_id: str, question: str) -> str:
        """
        Query vector stores for semantic transaction history, fetch MongoDB budgets,
        compile the grounding context, and prompt the conversational agent.
        """
        logger.info(f"RAGService: Initializing pipeline for query: '{question}'")
        
        # 1. Semantic search for relevant expenses (retrieve top 10 to ensure wide coverage)
        retrieved_expenses = chromadb_service.query_expenses(
            query_text=question,
            user_id=user_id,
            n_results=10
        )
        logger.info(f"RAGService: Vector search retrieved {len(retrieved_expenses)} matches.")

        # 2. Retrieve active monthly budgets from MongoDB
        from datetime import datetime
        current_month = datetime.utcnow().strftime("%Y-%m")
        budgets = db_manager.get_budgets_by_user(user_id=user_id, month=current_month)
        
        # 3. Answer question using chat agent
        answer = chat_agent.answer_question(
            user_id=user_id,
            question=question,
            retrieved_records=retrieved_expenses,
            active_budgets=budgets
        )
        
        return answer

# Singleton Instance
rag_service = RAGService()
