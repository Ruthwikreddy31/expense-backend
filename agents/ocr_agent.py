import logging
from typing import Dict, Any, Optional
from services.gemini_service import gemini_service
from services.groq_service import groq_service

logger = logging.getLogger("finance_app.ocr_agent")

class OCRAgent:
    """Agent specialized in refining OCR texts into structured JSON schemas."""
    
    def parse_text(self, ocr_text: str) -> Optional[Dict[str, Any]]:
        """Process flat text scan using AI prompts to produce normalized JSON dictionaries."""
        logger.info("OCRAgent: Structuring flat scanned characters...")
        
        prompt = f"""
        Extract clean personal financial fields from this OCR scanned receipt/invoice text:
        ---
        {ocr_text}
        ---

        Identify:
        1. Merchant Name
        2. Total Amount (float)
        3. Tax/VAT Amount (float, default to 0.0)
        4. Transaction Date (YYYY-MM-DD format - infer year relative to 2026 if missing)
        5. Payment Mode (Cash, Card, UPI, Bank Transfer)
        6. Purchased line items (list of items, each with 'name', 'price', and optional 'quantity')
        7. Brief transaction summary notes.

        Return a valid JSON object matching this structure EXACTLY:
        {{
            "merchant": "Merchant Name",
            "amount": 0.0,
            "tax": 0.0,
            "date": "YYYY-MM-DD",
            "payment_mode": "Card",
            "notes": "Purchased items description",
            "items": [
                {{"name": "Apples", "price": 12.00, "quantity": 3}}
            ],
            "category": "Other"
        }}
        Category must be one of: Food, Travel, Shopping, Rent, Entertainment, Education, Healthcare, Subscriptions, Utilities, Other.
        """
        
        # Use Gemini structured mode first, fallback to Groq JSON Mode
        if gemini_service.initialized:
            return gemini_service.generate_structured_json(
                prompt=prompt,
                system_instruction="You are an expert OCR schema structuring agent."
            )
        elif groq_service.initialized:
            return groq_service.generate_structured_json(
                prompt=prompt,
                system_prompt="You are an expert OCR schema structuring agent."
            )
            
        # Fallback dictionary if no LLM configured
        return {
            "merchant": "Unknown Merchant",
            "amount": 0.0,
            "tax": 0.0,
            "date": "2026-05-25",
            "payment_mode": "Cash",
            "notes": "Auto-parsed fallback due to missing LLM keys",
            "items": [],
            "category": "Other"
        }

# Singleton Instance
ocr_agent = OCRAgent()
