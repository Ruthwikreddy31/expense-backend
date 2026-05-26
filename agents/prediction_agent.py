import os
import json
import logging
from typing import Dict, Any, List
from services.gemini_service import gemini_service
from services.groq_service import groq_service

logger = logging.getLogger("finance_app.prediction_agent")

class PredictionAgent:
    """Agent predicting future expenditures and listing subscription calendar dates."""
    def __init__(self):
        current_dir = os.path.dirname(os.path.dirname(__file__))
        self.prompt_path = os.path.join(current_dir, "prompts", "prediction_prompt.txt")

    def _load_prompt_template(self) -> str:
        if os.path.exists(self.prompt_path):
            with open(self.prompt_path, "r") as f:
                return f.read()
        logger.warning(f"Prediction prompt file not found at {self.prompt_path}. Using standard template.")
        return "Analyze spending patterns in {historical_records} and forecast next month. Return JSON."

    def predict_finance(self, expenses: List[Dict[str, Any]], income: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compile transaction arrays and call LLM projection models to forecast next month's flows."""
        logger.info("PredictionAgent: Calculating future trends...")
        
        if not expenses and not income:
            return {
                "projected_total_expense": 0.0,
                "projected_total_income": 0.0,
                "projected_savings": 0.0,
                "predicted_trends": [],
                "upcoming_subscription_renewals": []
            }

        # Format historical summaries
        history = {
            "recent_expenses": [
                {
                    "merchant": e.get("merchant", "Unknown"),
                    "amount": e.get("amount", 0.0),
                    "category": e.get("category", "Other"),
                    "date": e.get("date", ""),
                    "notes": e.get("notes", "")
                }
                for e in expenses[-30:] # Last 30 expenses
            ],
            "recent_income": [
                {
                    "source": i.get("source", "Salary"),
                    "amount": i.get("amount", 0.0),
                    "date": i.get("date", "")
                }
                for i in income[-10:] # Last 10 incomes
            ]
        }

        template = self._load_prompt_template()
        prompt = template.format(historical_records=json.dumps(history, indent=2))

        sys_prompt = "You are a forecasting algorithm designed to predict personal finance balances and return JSON models."

        result = None
        if gemini_service.initialized:
            result = gemini_service.generate_structured_json(prompt, system_instruction=sys_prompt)
        elif groq_service.initialized:
            result = groq_service.generate_structured_json(prompt, system_prompt=sys_prompt)

        if result:
            return result

        # Basic linear projection fallback
        total_exp = sum(e.get("amount", 0.0) for e in expenses)
        total_inc = sum(i.get("amount", 0.0) for i in income)
        
        # Simple projection (averages or direct values)
        projected_exp = round(total_exp if len(expenses) > 0 else 0.0, 2)
        projected_inc = round(total_inc if len(income) > 0 else 0.0, 2)
        projected_sav = round(max(0.0, projected_inc - projected_exp), 2)

        # Detect potential subscription entries
        sub_list = []
        for e in expenses:
            cat = e.get("category", "")
            notes = e.get("notes", "").lower()
            merchant = e.get("merchant", "").lower()
            if cat == "Subscriptions" or "subscription" in notes or "monthly" in notes:
                # Approximate due date to next month
                date_str = e.get("date", "2026-05-25")
                try:
                    parts = date_str.split("-")
                    # Bump month to June
                    due = f"2026-06-{parts[2]}"
                except Exception:
                    due = "2026-06-05"
                
                # Prevent duplicate subscriptions in projected checklist
                if not any(s["merchant"].lower() == e.get("merchant", "").lower() for s in sub_list):
                    sub_list.append({
                        "merchant": e.get("merchant", "Recurring Service"),
                        "amount": e.get("amount", 0.0),
                        "due_date": due,
                        "is_essential": True if cat != "Entertainment" else False
                    })

        return {
            "projected_total_expense": projected_exp,
            "projected_total_income": projected_inc,
            "projected_savings": projected_sav,
            "predicted_trends": [
                {"category": "Food", "trend": "Stable", "explanation": "Spending tracks inline with historical baseline."}
            ],
            "upcoming_subscription_renewals": sub_list
        }

# Singleton Instance
prediction_agent = PredictionAgent()
