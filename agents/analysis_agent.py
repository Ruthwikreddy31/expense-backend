import os
import json
import logging
from typing import Dict, Any, List
from services.gemini_service import gemini_service
from services.groq_service import groq_service

logger = logging.getLogger("finance_app.analysis_agent")

class AnalysisAgent:
    """Agent running detailed behavior reviews over full transaction batches."""
    def __init__(self):
        current_dir = os.path.dirname(os.path.dirname(__file__))
        self.prompt_path = os.path.join(current_dir, "prompts", "analysis_prompt.txt")

    def _load_prompt_template(self) -> str:
        if os.path.exists(self.prompt_path):
            with open(self.prompt_path, "r") as f:
                return f.read()
        logger.warning(f"Analysis prompt file not found at {self.prompt_path}. Using basic template.")
        return "Analyze this expense data: {expense_data}. Return JSON with summary, category_insights, anomalies, savings_opportunities, health_score."

    def analyze_expenses(self, expenses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate transaction entries and trigger LLM analysis for saving opportunities & anomalies."""
        logger.info(f"AnalysisAgent: Analyzing {len(expenses)} transactions...")
        
        if not expenses:
            return {
                "summary": "No transactions recorded yet. Add some expenses to get started!",
                "category_insights": [],
                "anomalies": [],
                "savings_opportunities": [],
                "health_score": 100
            }

        # Truncate expense elements to keep prompts tidy and compact
        simplified_expenses = []
        for e in expenses:
            simplified_expenses.append({
                "merchant": e.get("merchant", "Unknown"),
                "amount": e.get("amount", 0.0),
                "category": e.get("category", "Other"),
                "date": e.get("date", ""),
                "notes": e.get("notes", "")
            })

        expense_data_str = json.dumps(simplified_expenses, indent=2)
        template = self._load_prompt_template()
        prompt = template.format(expense_data=expense_data_str)

        sys_prompt = "You are a professional personal finance advisor agent designed to identify costs leakage and return JSON summaries."

        result = None
        if gemini_service.initialized:
            result = gemini_service.generate_structured_json(prompt, system_instruction=sys_prompt)
        elif groq_service.initialized:
            result = groq_service.generate_structured_json(prompt, system_prompt=sys_prompt)

        if result:
            return result

        # Basic default structures if no AI engines are available
        return {
            "summary": "Basic offline summary: You have recorded expenses across multiple categories. Set up API keys to enable deep AI insights.",
            "category_insights": [
                {"category": "Other", "percentage": 100.0, "assessment": "No custom insights available. Add API keys to activate analysis."}
            ],
            "anomalies": [],
            "savings_opportunities": [
                {"actionable_tip": "Keep tracking your daily expenses to gain a clearer picture of your outflows.", "estimated_savings": 10.0}
            ],
            "health_score": 90
        }

# Singleton Instance
analysis_agent = AnalysisAgent()
