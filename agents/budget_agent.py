import os
import json
import logging
from typing import Dict, Any, List
from services.gemini_service import gemini_service
from services.groq_service import groq_service

logger = logging.getLogger("finance_app.budget_agent")

class BudgetAgent:
    """Agent running budget analysis and limit optimizations."""
    def __init__(self):
        current_dir = os.path.dirname(os.path.dirname(__file__))
        self.prompt_path = os.path.join(current_dir, "prompts", "budget_prompt.txt")

    def _load_prompt_template(self) -> str:
        if os.path.exists(self.prompt_path):
            with open(self.prompt_path, "r") as f:
                return f.read()
        logger.warning(f"Budget prompt file not found at {self.prompt_path}. Using standard backup.")
        return "Compare spending {expense_history} vs limits {active_limits} and recommend new targets. Return JSON."

    def analyze_budgets(self, expenses: List[Dict[str, Any]], budgets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Verify spending status against budgets and generate optimization recommendations."""
        logger.info("BudgetAgent: Analyzing budget levels...")
        
        # Calculate spending totals per category for this month
        category_spending = {}
        for e in expenses:
            cat = e.get("category", "Other")
            amount = e.get("amount", 0.0)
            category_spending[cat] = category_spending.get(cat, 0.0) + amount

        # Prepare active limits
        limits_simplified = []
        for b in budgets:
            category = b.get("category")
            limit_val = b.get("limit", 0.0)
            spent = category_spending.get(category, 0.0)
            limits_simplified.append({
                "category": category,
                "limit": limit_val,
                "spent": spent
            })

        history_simplified = []
        for e in expenses[-30:]:  # Take last 30 transactions
            history_simplified.append({
                "merchant": e.get("merchant", "Unknown"),
                "amount": e.get("amount", 0.0),
                "category": e.get("category", "Other"),
                "date": e.get("date", "")
            })

        template = self._load_prompt_template()
        prompt = (
            template
            .replace("{expense_history}", json.dumps(history_simplified, indent=2))
            .replace("{active_limits}", json.dumps(limits_simplified, indent=2))
        )

        sys_prompt = "You are an advanced budgeting intelligence agent. Monitor category thresholds and output strict JSON status alerts."

        result = None
        if gemini_service.initialized:
            result = gemini_service.generate_structured_json(prompt, system_instruction=sys_prompt)
        elif groq_service.initialized:
            result = groq_service.generate_structured_json(prompt, system_prompt=sys_prompt)

        if result:
            return result

        # Basic default structures if AI keys are missing
        status_checks = []
        for b in budgets:
            category = b.get("category")
            limit_val = b.get("limit", 1.0)
            spent = category_spending.get(category, 0.0)
            pct = round((spent / limit_val) * 100, 1)
            status = "Safe"
            msg = f"You have used {pct}% of your {category} budget."
            if pct >= 100:
                status = "Critical"
                msg = f"Overspent! You have exceeded your {category} allowance by ${round(spent - limit_val, 2)}."
            elif pct >= 80:
                status = "Warning"
                msg = f"Warning: You have used {pct}% of your {category} allowance."
            
            status_checks.append({
                "category": category,
                "active_limit": limit_val,
                "actual_spent": spent,
                "percent_used": pct,
                "status": status,
                "message": msg
            })

        return {
            "status_checks": status_checks,
            "savings_target": {
                "monthly_goal": 100.0,
                "achievable_projection": 80.0,
                "pathway": "Tracking transactions is active. Configure AI keys to get targeted recommendations."
            },
            "recommended_limits": [
                {"category": "Food", "suggested_limit": 300.0, "rationale": "Base target recommendation."}
            ]
        }

# Singleton Instance
budget_agent = BudgetAgent()
