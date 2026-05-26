from pydantic import BaseModel, Field, EmailStr, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)

class UserLogin(BaseModel):
    username: str
    password: str

class UserDB(BaseModel):
    id: str = Field(default=None, alias="_id")
    username: str
    password_hash: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class TransactionItem(BaseModel):
    name: str
    price: float
    quantity: int = 1

class ExpenseCreate(BaseModel):
    amount: float = Field(..., gt=0, description="Expense amount must be positive")
    category: str = Field(default="Other")
    date: str = Field(..., description="Date of transaction in YYYY-MM-DD format")
    payment_mode: str = Field(default="Cash")
    notes: Optional[str] = ""
    merchant: Optional[str] = "Unknown"
    items: Optional[List[TransactionItem]] = []
    receipt_path: Optional[str] = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return value
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

    @field_validator("category")
    @classmethod
    def validate_category(cls, value: str) -> str:
        allowed = {"Food", "Travel", "Shopping", "Rent", "Entertainment", "Education", "Healthcare", "Subscriptions", "Utilities", "Other"}
        title_val = value.strip().title()
        if title_val in allowed:
            return title_val
        return "Other"

class ExpenseDB(ExpenseCreate):
    id: str = Field(..., alias="_id")
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class IncomeCreate(BaseModel):
    amount: float = Field(..., gt=0, description="Income amount must be positive")
    source: str = Field(..., min_length=1)
    date: str = Field(..., description="Date of receipt in YYYY-MM-DD format")
    notes: Optional[str] = ""

    @field_validator("date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m-%d")
            return value
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

class IncomeDB(IncomeCreate):
    id: str = Field(..., alias="_id")
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class BudgetCreate(BaseModel):
    category: str
    limit: float = Field(..., gt=0)
    month: str = Field(..., description="Month in YYYY-MM format")

    @field_validator("month")
    @classmethod
    def validate_month(cls, value: str) -> str:
        try:
            datetime.strptime(value, "%Y-%m")
            return value
        except ValueError:
            raise ValueError("Month must be in YYYY-MM format")

class BudgetDB(BudgetCreate):
    id: str = Field(..., alias="_id")
    user_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

class ChatMessage(BaseModel):
    role: str # 'user' or 'assistant'
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ChatHistoryDB(BaseModel):
    id: str = Field(..., alias="_id")
    user_id: str
    messages: List[ChatMessage] = []
    updated_at: datetime = Field(default_factory=datetime.utcnow)
