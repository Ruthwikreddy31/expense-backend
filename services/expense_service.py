import logging
from typing import Dict, Any, List, Optional
from database.mongodb import db_manager
from services.chromadb_service import chromadb_service
from agents.categorization_agent import categorization_agent

logger = logging.getLogger("finance_app.expense_service")

class ExpenseService:
    """Orchestrator managing transaction records across Mongo and ChromaDB stores."""

    def add_expense(self, user_id: str, expense_data: Dict[str, Any]) -> str:
        """
        Add a new expense transaction.
        1. Categorize automatically using categorization_agent if category is "Other" or blank.
        2. Persist in MongoDB database.
        3. Compile descriptive RAG document text.
        4. Embed and index in ChromaDB.
        """
        logger.info(f"ExpenseService: Adding transaction for user {user_id}...")
        
        merchant = expense_data.get("merchant", "Unknown Merchant").strip()
        amount = float(expense_data.get("amount", 0.0))
        notes = expense_data.get("notes", "").strip()
        category = expense_data.get("category", "Other").strip()
        date_str = expense_data.get("date", "")
        payment_mode = expense_data.get("payment_mode", "Cash").strip()
        items = expense_data.get("items", [])
        receipt_path = expense_data.get("receipt_path")

        # 1. Run automatic categorization if category is default "Other"
        if category == "Other" or not category:
            cat_result = categorization_agent.categorize_transaction(
                merchant=merchant,
                amount=amount,
                notes=notes
            )
            category = cat_result.get("category", "Other")
            logger.info(f"ExpenseService: Automated categorization resolved: '{category}' (Reason: {cat_result.get('reasoning')})")

        # Prepare Mongo record
        db_record = {
            "user_id": user_id,
            "amount": amount,
            "category": category,
            "date": date_str,
            "payment_mode": payment_mode,
            "notes": notes,
            "merchant": merchant,
            "items": items,
            "receipt_path": receipt_path
        }

        # 2. Persist in Mongo
        expense_id = db_manager.create_expense(db_record)
        logger.info(f"ExpenseService: Expense recorded in Mongo with ID: {expense_id}")

        # 3. Create descriptive grounding text for ChromaDB vector search
        items_str = ", ".join([f"{i.get('name')} (${i.get('price')})" for i in items]) if items else "No items listed"
        document_text = (
            f"Transaction ID {expense_id} for User {user_id}. "
            f"Paid ${amount:.2f} to '{merchant}' on date {date_str}. "
            f"Category: {category}. Payment mode: {payment_mode}. "
            f"Notes: {notes}. Items purchased: {items_str}."
        )

        # Prepare metadata for vector querying filter efficiency
        vector_metadata = {
            "expense_id": expense_id,
            "user_id": user_id,
            "amount": amount,
            "category": category,
            "merchant": merchant,
            "date": date_str
        }

        # 4. Index in ChromaDB
        try:
            chromadb_service.add_expense(
                expense_id=expense_id,
                document_text=document_text,
                metadata=vector_metadata
            )
            logger.info("ExpenseService: Vector index matching added successfully.")
        except Exception as e:
            logger.error(f"ExpenseService: ChromaDB indexing failed for expense {expense_id}: {e}")

        return expense_id

    def delete_expense(self, user_id: str, expense_id: str) -> bool:
        """Delete an expense record from both Mongo and ChromaDB vector stores."""
        logger.info(f"ExpenseService: Deleting transaction {expense_id} for user {user_id}...")
        
        # Verify transaction ownership in Mongo
        expenses = db_manager.get_expenses_by_user(user_id)
        is_owned = any((e.get("_id") == expense_id or e.get("id") == expense_id) for e in expenses)
        
        if not is_owned:
            logger.warning(f"Delete rejected: User {user_id} does not own transaction {expense_id}.")
            return False

        # Delete from Mongo
        mongo_success = db_manager.delete_expense(expense_id)
        
        # Delete from ChromaDB
        try:
            chromadb_service.delete_expense(expense_id)
            vector_success = True
        except Exception as e:
            logger.error(f"ChromaDB deletion failed: {e}")
            vector_success = False

        return mongo_success and vector_success

# Singleton Instance
expense_service = ExpenseService()
