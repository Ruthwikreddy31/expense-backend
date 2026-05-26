import os
import json
import logging
import uuid
from datetime import datetime
from threading import Lock
from typing import List, Dict, Any, Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("finance_app.database")
logging.basicConfig(level=logging.INFO)

class JSONDatabaseFallback:
    """Thread-safe JSON-based database fallback replicating MongoDB collection concepts."""
    def __init__(self, file_path: str = "./database/local_db.json"):
        self.file_path = file_path
        self.lock = Lock()
        self._initialize_db()

    def _initialize_db(self):
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, "w") as f:
                json.dump({
                    "users": [],
                    "expenses": [],
                    "income": [],
                    "budgets": [],
                    "chat_history": []
                }, f, indent=4)

    def _read(self) -> Dict[str, List[Dict[str, Any]]]:
        with self.lock:
            try:
                with open(self.file_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON database fallback: {e}")
                return {"users": [], "expenses": [], "income": [], "budgets": [], "chat_history": []}

    def _write(self, data: Dict[str, List[Dict[str, Any]]]):
        with self.lock:
            try:
                with open(self.file_path, "w") as f:
                    json.dump(data, f, indent=4, default=str)
            except Exception as e:
                logger.error(f"Error writing to JSON database fallback: {e}")

    # --- Users ---
    def insert_user(self, user_data: Dict[str, Any]) -> str:
        db = self._read()
        user_data["_id"] = str(uuid.uuid4())
        user_data["created_at"] = user_data.get("created_at", datetime.utcnow()).isoformat()
        db["users"].append(user_data)
        self._write(db)
        return user_data["_id"]

    def find_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        db = self._read()
        for u in db["users"]:
            if u["username"].lower() == username.lower():
                return u
        return None

    def find_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        db = self._read()
        for u in db["users"]:
            if u["_id"] == user_id:
                return u
        return None

    # --- Expenses ---
    def insert_expense(self, expense_data: Dict[str, Any]) -> str:
        db = self._read()
        expense_data["_id"] = str(uuid.uuid4())
        expense_data["created_at"] = expense_data.get("created_at", datetime.utcnow()).isoformat()
        db["expenses"].append(expense_data)
        self._write(db)
        return expense_data["_id"]

    def get_expenses(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        db = self._read()
        results = []
        for e in db["expenses"]:
            match = True
            for k, v in query.items():
                if e.get(k) != v:
                    match = False
                    break
            if match:
                results.append(e)
        return results

    def update_expense(self, expense_id: str, update_data: Dict[str, Any]) -> bool:
        db = self._read()
        for idx, e in enumerate(db["expenses"]):
            if e["_id"] == expense_id:
                db["expenses"][idx].update(update_data)
                self._write(db)
                return True
        return False

    def delete_expense(self, expense_id: str) -> bool:
        db = self._read()
        initial_len = len(db["expenses"])
        db["expenses"] = [e for e in db["expenses"] if e["_id"] != expense_id]
        if len(db["expenses"]) < initial_len:
            self._write(db)
            return True
        return False

    # --- Income ---
    def insert_income(self, income_data: Dict[str, Any]) -> str:
        db = self._read()
        income_data["_id"] = str(uuid.uuid4())
        income_data["created_at"] = income_data.get("created_at", datetime.utcnow()).isoformat()
        db["income"].append(income_data)
        self._write(db)
        return income_data["_id"]

    def get_income(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        db = self._read()
        results = []
        for i in db["income"]:
            match = True
            for k, v in query.items():
                if i.get(k) != v:
                    match = False
                    break
            if match:
                results.append(i)
        return results

    def delete_income(self, income_id: str) -> bool:
        db = self._read()
        initial_len = len(db["income"])
        db["income"] = [i for i in db["income"] if i["_id"] != income_id]
        if len(db["income"]) < initial_len:
            self._write(db)
            return True
        return False

    # --- Budgets ---
    def upsert_budget(self, budget_data: Dict[str, Any]) -> str:
        db = self._read()
        user_id = budget_data["user_id"]
        category = budget_data["category"]
        month = budget_data["month"]

        # Check if already exists
        for idx, b in enumerate(db["budgets"]):
            if b["user_id"] == user_id and b["category"] == category and b["month"] == month:
                db["budgets"][idx].update(budget_data)
                self._write(db)
                return b["_id"]

        budget_data["_id"] = str(uuid.uuid4())
        budget_data["created_at"] = budget_data.get("created_at", datetime.utcnow()).isoformat()
        db["budgets"].append(budget_data)
        self._write(db)
        return budget_data["_id"]

    def get_budgets(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        db = self._read()
        results = []
        for b in db["budgets"]:
            match = True
            for k, v in query.items():
                if b.get(k) != v:
                    match = False
                    break
            if match:
                results.append(b)
        return results

    # --- Chat History ---
    def save_chat_history(self, user_id: str, messages: List[Dict[str, Any]]):
        db = self._read()
        updated_iso = datetime.utcnow().isoformat()
        for idx, ch in enumerate(db["chat_history"]):
            if ch["user_id"] == user_id:
                db["chat_history"][idx]["messages"] = messages
                db["chat_history"][idx]["updated_at"] = updated_iso
                self._write(db)
                return

        db["chat_history"].append({
            "_id": str(uuid.uuid4()),
            "user_id": user_id,
            "messages": messages,
            "updated_at": updated_iso
        })
        self._write(db)

    def get_chat_history(self, user_id: str) -> List[Dict[str, Any]]:
        db = self._read()
        for ch in db["chat_history"]:
            if ch["user_id"] == user_id:
                return ch.get("messages", [])
        return []


class DatabaseManager:
    """Manages the database interface, providing MongoDB or falling back to local JSON file db."""
    def __init__(self):
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/expense_tracker")
        self.client = None
        self.db = None
        self.fallback_db = JSONDatabaseFallback()
        self.is_fallback = False
        self._connect()

    def _connect(self):
        try:
            # Short 2-second timeout to avoid hanging UI if Mongo is down
            self.client = MongoClient(self.mongodb_uri, serverSelectionTimeoutMS=2000)
            # Trigger simple query to test connection
            self.client.server_info()
            self.db = self.client.get_database()
            logger.info("Successfully connected to MongoDB.")
            self.is_fallback = False
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            logger.warning(f"MongoDB connection failed. Falling back to local JSON database. Error: {e}")
            self.is_fallback = True

    # --- Users ---
    def create_user(self, user_data: Dict[str, Any]) -> str:
        if self.is_fallback:
            return self.fallback_db.insert_user(user_data)
        try:
            user_data["created_at"] = user_data.get("created_at", datetime.utcnow())
            result = self.db.users.insert_one(user_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB error in create_user: {e}. Retrying with fallback...")
            return self.fallback_db.insert_user(user_data)

    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        if self.is_fallback:
            return self.fallback_db.find_user_by_username(username)
        try:
            user = self.db.users.find_one({"username": {"$regex": f"^{username}$", "$options": "i"}})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            logger.error(f"MongoDB error in get_user_by_username: {e}. Using fallback...")
            return self.fallback_db.find_user_by_username(username)

    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        if self.is_fallback:
            return self.fallback_db.find_user_by_id(user_id)
        try:
            from bson import ObjectId
            user = self.db.users.find_one({"_id": ObjectId(user_id)})
            if user:
                user["_id"] = str(user["_id"])
            return user
        except Exception as e:
            # Catch ObjectId conversion or database error
            logger.warning(f"MongoDB ObjectId query failed or database offline for ID: {user_id}. Using fallback...")
            return self.fallback_db.find_user_by_id(user_id)

    # --- Expenses ---
    def create_expense(self, expense_data: Dict[str, Any]) -> str:
        if self.is_fallback:
            return self.fallback_db.insert_expense(expense_data)
        try:
            expense_data["created_at"] = expense_data.get("created_at", datetime.utcnow())
            result = self.db.expenses.insert_one(expense_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB error in create_expense: {e}. Using fallback...")
            return self.fallback_db.insert_expense(expense_data)

    def get_expenses_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        if self.is_fallback:
            return self.fallback_db.get_expenses({"user_id": user_id})
        try:
            cursor = self.db.expenses.find({"user_id": user_id})
            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)
            return results
        except Exception as e:
            logger.error(f"MongoDB error in get_expenses_by_user: {e}. Using fallback...")
            return self.fallback_db.get_expenses({"user_id": user_id})

    def update_expense(self, expense_id: str, update_data: Dict[str, Any]) -> bool:
        if self.is_fallback:
            return self.fallback_db.update_expense(expense_id, update_data)
        try:
            from bson import ObjectId
            result = self.db.expenses.update_one({"_id": ObjectId(expense_id)}, {"$set": update_data})
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"MongoDB error in update_expense: {e}. Using fallback...")
            return self.fallback_db.update_expense(expense_id, update_data)

    def delete_expense(self, expense_id: str) -> bool:
        if self.is_fallback:
            return self.fallback_db.delete_expense(expense_id)
        try:
            from bson import ObjectId
            result = self.db.expenses.delete_one({"_id": ObjectId(expense_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"MongoDB error in delete_expense: {e}. Using fallback...")
            return self.fallback_db.delete_expense(expense_id)

    # --- Income ---
    def create_income(self, income_data: Dict[str, Any]) -> str:
        if self.is_fallback:
            return self.fallback_db.insert_income(income_data)
        try:
            income_data["created_at"] = income_data.get("created_at", datetime.utcnow())
            result = self.db.income.insert_one(income_data)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"MongoDB error in create_income: {e}. Using fallback...")
            return self.fallback_db.insert_income(income_data)

    def get_income_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        if self.is_fallback:
            return self.fallback_db.get_income({"user_id": user_id})
        try:
            cursor = self.db.income.find({"user_id": user_id})
            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)
            return results
        except Exception as e:
            logger.error(f"MongoDB error in get_income_by_user: {e}. Using fallback...")
            return self.fallback_db.get_income({"user_id": user_id})

    def delete_income(self, income_id: str) -> bool:
        if self.is_fallback:
            return self.fallback_db.delete_income(income_id)
        try:
            from bson import ObjectId
            result = self.db.income.delete_one({"_id": ObjectId(income_id)})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"MongoDB error in delete_income: {e}. Using fallback...")
            return self.fallback_db.delete_income(income_id)

    # --- Budgets ---
    def create_or_update_budget(self, budget_data: Dict[str, Any]) -> str:
        if self.is_fallback:
            return self.fallback_db.upsert_budget(budget_data)
        try:
            user_id = budget_data["user_id"]
            category = budget_data["category"]
            month = budget_data["month"]
            
            result = self.db.budgets.update_one(
                {"user_id": user_id, "category": category, "month": month},
                {"$set": budget_data},
                upsert=True
            )
            if result.upserted_id:
                return str(result.upserted_id)
            
            # Find and return the updated document ID
            existing = self.db.budgets.find_one({"user_id": user_id, "category": category, "month": month})
            return str(existing["_id"]) if existing else ""
        except Exception as e:
            logger.error(f"MongoDB error in upsert budget: {e}. Using fallback...")
            return self.fallback_db.upsert_budget(budget_data)

    def get_budgets_by_user(self, user_id: str, month: Optional[str] = None) -> List[Dict[str, Any]]:
        query = {"user_id": user_id}
        if month:
            query["month"] = month
            
        if self.is_fallback:
            return self.fallback_db.get_budgets(query)
        try:
            cursor = self.db.budgets.find(query)
            results = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                results.append(doc)
            return results
        except Exception as e:
            logger.error(f"MongoDB error in get_budgets_by_user: {e}. Using fallback...")
            return self.fallback_db.get_budgets(query)

    # --- Chat History ---
    def save_chat_history(self, user_id: str, messages: List[Dict[str, Any]]):
        if self.is_fallback:
            self.fallback_db.save_chat_history(user_id, messages)
            return
        try:
            self.db.chat_history.update_one(
                {"user_id": user_id},
                {"$set": {
                    "messages": messages,
                    "updated_at": datetime.utcnow()
                }},
                upsert=True
            )
        except Exception as e:
            logger.error(f"MongoDB error in save_chat_history: {e}. Using fallback...")
            self.fallback_db.save_chat_history(user_id, messages)

    def get_chat_history(self, user_id: str) -> List[Dict[str, Any]]:
        if self.is_fallback:
            return self.fallback_db.get_chat_history(user_id)
        try:
            doc = self.db.chat_history.find_one({"user_id": user_id})
            return doc.get("messages", []) if doc else []
        except Exception as e:
            logger.error(f"MongoDB error in get_chat_history: {e}. Using fallback...")
            return self.fallback_db.get_chat_history(user_id)

# Singleton Instance
db_manager = DatabaseManager()
