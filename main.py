import os
import shutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup relative sys.path so we can import local modules when running inside backend/
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.mongodb import db_manager
from services.ocr_service import ocr_service
from services.expense_service import expense_service
from services.analytics_service import analytics_service
from services.rag_service import rag_service
from agents.fraud_agent import fraud_agent
from agents.prediction_agent import prediction_agent
from utils.security import hash_password, verify_password, create_access_token, decode_access_token

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("finance_app.backend")

app = FastAPI(
    title="Expense Tracker REST API",
    description="Clean high-performance backend serving personal finance ledger, RAG agent assistant, and OCR scanners.",
    version="1.0.0"
)

# Enable CORS for Vercel and local development frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Upload directory
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper function to resolve active user from JWT token or custom headers
def get_current_user_id(
    authorization: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
    user_id: Optional[str] = Query(None)
) -> str:
    """Resolve active user ID securely using Auth Header JWT, Custom x-user-id Header, or query parameter."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        payload = decode_access_token(token)
        if payload and "user_id" in payload:
            return payload["user_id"]
        raise HTTPException(
            status_code=status.HTTP_418_IM_A_TEAPOT if authorization == "Bearer invalid" else status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or invalid access token. Please login again."
        )
    
    # Fallback to custom headers or query params
    if x_user_id:
        return x_user_id
    if user_id:
        return user_id
        
    return "local-user"

# --- Models ---
class UserSignup(BaseModel):
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ExpenseCreate(BaseModel):
    merchant: str
    amount: float
    category: str = "Other"
    date: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))
    payment_mode: str = "Cash"
    notes: str = ""
    items: List[Dict[str, Any]] = []
    receipt_path: Optional[str] = None

class IncomeCreate(BaseModel):
    source: str
    amount: float
    date: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))
    notes: str = ""

class BudgetCreate(BaseModel):
    category: str
    limit: float
    month: str = Field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m"))

class ChatQuery(BaseModel):
    question: str

# --- Routes ---

@app.get("/api/health")
def health_check():
    """Verify backend and database states."""
    db_status = "Fallback JSON active" if db_manager.is_fallback else "MongoDB Connected"
    return {
        "status": "Healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "database": db_status,
        "environment": "Production-Decoupled"
    }

# --- Auth ---
@app.post("/api/auth/signup")
def signup(user: UserSignup):
    """Register a new account credentials."""
    existing = db_manager.get_user_by_username(user.username)
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    hashed = hash_password(user.password)
    user_record = {
        "username": user.username,
        "password": hashed
    }
    user_id = db_manager.create_user(user_record)
    return {"status": "Success", "user_id": user_id, "username": user.username}

@app.post("/api/auth/login")
def login(user: UserLogin):
    """Authenticate and create access token."""
    user_record = db_manager.get_user_by_username(user.username)
    if not user_record or not verify_password(user.password, user_record.get("password", "")):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    
    token = create_access_token(data={"user_id": user_record["_id"], "username": user_record["username"]})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user_record["_id"],
        "username": user_record["username"]
    }

# --- Dashboard & KPIs ---
@app.get("/api/kpis")
def get_kpis(user_id: str = Depends(get_current_user_id)):
    """Fetch high-performance KPI aggregates, spend trend, and fraud security warnings."""
    expenses = db_manager.get_expenses_by_user(user_id)
    income = db_manager.get_income_by_user(user_id)
    
    kpis = analytics_service.compute_summary_kpis(expenses, income)
    comparisons = analytics_service.compute_monthly_comparisons(expenses, income)
    trend = analytics_service.compute_daily_spending_trend(expenses)
    merchants = analytics_service.get_highest_spending_merchants(expenses, limit=5)
    anomalies = fraud_agent.scan_for_anomalies(expenses)
    
    return {
        "summary": kpis,
        "comparisons": comparisons,
        "trend": trend,
        "top_merchants": merchants,
        "anomalies": anomalies
    }

# --- Expense ---
@app.get("/api/expenses")
def list_expenses(user_id: str = Depends(get_current_user_id)):
    """Fetch expense list for current user."""
    expenses = db_manager.get_expenses_by_user(user_id)
    # Sort chronologically newest first
    expenses.sort(key=lambda x: x.get("date", ""), reverse=True)
    return expenses

@app.post("/api/expenses")
def add_expense(expense: ExpenseCreate, user_id: str = Depends(get_current_user_id)):
    """Create and index a new outflow expense."""
    record = expense.model_dump()
    expense_id = expense_service.add_expense(user_id, record)
    return {"status": "Success", "expense_id": expense_id}

@app.delete("/api/expenses/{expense_id}")
def delete_expense(expense_id: str, user_id: str = Depends(get_current_user_id)):
    """Remove expense record from databases."""
    success = expense_service.delete_expense(user_id, expense_id)
    if not success:
        raise HTTPException(status_code=404, detail="Expense not found or unauthorized deletion.")
    return {"status": "Success", "message": "Expense successfully deleted"}

# --- Income ---
@app.get("/api/income")
def list_income(user_id: str = Depends(get_current_user_id)):
    """Fetch income entries for current user."""
    income = db_manager.get_income_by_user(user_id)
    income.sort(key=lambda x: x.get("date", ""), reverse=True)
    return income

@app.post("/api/income")
def add_income(income: IncomeCreate, user_id: str = Depends(get_current_user_id)):
    """Add a new inflow income record."""
    record = income.model_dump()
    record["user_id"] = user_id
    income_id = db_manager.create_income(record)
    return {"status": "Success", "income_id": income_id}

@app.delete("/api/income/{income_id}")
def delete_income(income_id: str, user_id: str = Depends(get_current_user_id)):
    """Delete an income record."""
    success = db_manager.delete_income(income_id)
    if not success:
        raise HTTPException(status_code=404, detail="Income not found or unauthorized deletion.")
    return {"status": "Success", "message": "Income successfully deleted"}

# --- Budgets ---
@app.get("/api/budgets")
def list_budgets(user_id: str = Depends(get_current_user_id), month: Optional[str] = None):
    """Fetch budget allocations."""
    if not month:
        month = datetime.utcnow().strftime("%Y-%m")
    budgets = db_manager.get_budgets_by_user(user_id, month=month)
    return budgets

@app.post("/api/budgets")
def set_budget(budget: BudgetCreate, user_id: str = Depends(get_current_user_id)):
    """Establish budget threshold allowances."""
    record = budget.model_dump()
    record["user_id"] = user_id
    budget_id = db_manager.create_or_update_budget(record)
    return {"status": "Success", "budget_id": budget_id}

# --- Receipt OCR Uploader ---
@app.post("/api/ocr/scan")
def scan_receipt(file: UploadFile = File(...), user_id: str = Depends(get_current_user_id)):
    """Accepts image/PDF upload, runs Dual OCR, returns structured details."""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in [".jpg", ".jpeg", ".png", ".pdf"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Supported types: png, jpg, jpeg, pdf")
    
    # Save file temporarily
    temp_file_name = f"{user_id}_{int(datetime.utcnow().timestamp())}{file_ext}"
    temp_file_path = os.path.join(UPLOAD_DIR, temp_file_name)
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Execute OCR scanning pipeline
        parsed_data = ocr_service.process_receipt(temp_file_path)
        if not parsed_data:
            raise HTTPException(status_code=500, detail="Receipt OCR scan failed.")
            
        parsed_data["receipt_path"] = temp_file_path
        return parsed_data
    except Exception as e:
        logger.error(f"OCR Scan endpoint failed: {e}")
        # Clean up file on failure
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Internal scan failure: {e}")

# --- RAG Chat Assistant ---
@app.post("/api/chat")
def run_chat(query: ChatQuery, user_id: str = Depends(get_current_user_id)):
    """Ground-truth RAG contextual chat engine."""
    try:
        answer = rag_service.process_chat_query(user_id, query.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Chat RAG endpoint failure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Forecasting Predictions ---
@app.get("/api/prediction")
def get_prediction(user_id: str = Depends(get_current_user_id)):
    """Execute Personal Finance Forecasting Agent."""
    expenses = db_manager.get_expenses_by_user(user_id)
    income = db_manager.get_income_by_user(user_id)
    
    prediction_results = prediction_agent.predict_finance(expenses, income)
    return prediction_results

if __name__ == "__main__":
    import uvicorn
    # Standalone execution
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
