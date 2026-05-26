import os
import logging
from typing import Dict, Any
from services.gemini_service import gemini_service
from services.groq_service import groq_service

logger = logging.getLogger("finance_app.categorization_agent")

class CategorizationAgent:
    """Agent running few-shot prompts to classify transaction ledger entries."""
    def __init__(self):
        # Resolve prompt file path
        current_dir = os.path.dirname(os.path.dirname(__file__))
        self.prompt_path = os.path.join(current_dir, "prompts", "categorization_prompt.txt")

    def _load_prompt_template(self) -> str:
        if os.path.exists(self.prompt_path):
            with open(self.prompt_path, "r") as f:
                return f.read()
        logger.warning(f"Categorization prompt file not found at {self.prompt_path}. Using hardcoded default.")
        return """
        Categorize the transaction.
        Merchant: {merchant}
        Amount: ${amount}
        Notes: {notes}
        Return JSON with keys "reasoning" and "category".
        Category must be one of: Food, Travel, Shopping, Rent, Entertainment, Education, Healthcare, Subscriptions, Utilities, Other.
        """

    def categorize_transaction(self, merchant: str, amount: float, notes: str) -> Dict[str, Any]:
        """Classify merchant, notes, and amount into a single finance category."""
        logger.info(f"Categorizing transaction: Merchant='{merchant}', Amount={amount}")
        template = self._load_prompt_template()
        prompt = (
            template
            .replace("{merchant}", merchant)
            .replace("{amount}", str(amount))
            .replace("{notes}", notes)
        )

        sys_instruction = "You are a professional ledger classification engine designed to return strict JSON data."

        # Use Gemini first, then fall back to Groq
        result = None
        if gemini_service.initialized:
            result = gemini_service.generate_structured_json(prompt, system_instruction=sys_instruction)
        elif groq_service.initialized:
            result = groq_service.generate_structured_json(prompt, system_prompt=sys_instruction)

        if result and "category" in result:
            # Standardize category text
            cat = result["category"].strip().title()
            allowed = {"Food", "Travel", "Shopping", "Rent", "Entertainment", "Education", "Healthcare", "Subscriptions", "Utilities", "Other"}
            if cat in allowed:
                return result
            # Map misspellings or custom answers to other
            result["category"] = "Other"
            return result

        # Basic default fallback if LLMs fail
        return {
            "reasoning": "Fallback classification due to AI service disruption.",
            "category": "Other"
        }

# Singleton Instance
categorization_agent = CategorizationAgent()
