import os
import logging
from typing import Dict, Any, List
from services.gemini_service import gemini_service
from services.groq_service import groq_service

logger = logging.getLogger("finance_app.chat_agent")

class ChatAgent:
    """Agent conducting contextual finance dialogs using RAG results."""
    def __init__(self):
        current_dir = os.path.dirname(os.path.dirname(__file__))
        self.prompt_path = os.path.join(current_dir, "prompts", "chat_prompt.txt")

    def _load_prompt_template(self) -> str:
        if os.path.exists(self.prompt_path):
            with open(self.prompt_path, "r") as f:
                return f.read()
        logger.warning(f"Chat prompt template not found at {self.prompt_path}. Using standard backup.")
        return """
        Answer the user's question based strictly on context.
        Context: {retrieved_records}
        Budgets: {active_budgets}
        Question: {user_question}
        User ID: {user_id}
        """

    def answer_question(self, user_id: str, question: str, retrieved_records: List[Dict[str, Any]], active_budgets: List[Dict[str, Any]]) -> str:
        """Construct the system instruction prompt and trigger LLM generation to answer the user query."""
        logger.info(f"ChatAgent: Answering query for User {user_id}: '{question}'")
        
        # Format the retrieved documents
        if retrieved_records:
            records_str = ""
            for idx, r in enumerate(retrieved_records):
                meta = r.get("metadata", {})
                records_str += f"{idx+1}. Merchant: {meta.get('merchant', 'Unknown')}, Amount: ${meta.get('amount', 0.0)}, Category: {meta.get('category', 'Other')}, Date: {meta.get('date', '')}, Notes: {meta.get('notes', '')}\n"
        else:
            records_str = "No transaction records found matching this context."

        # Format budgets
        if active_budgets:
            budgets_str = ""
            for b in active_budgets:
                budgets_str += f"- Category: {b.get('category', 'Other')}, Limit: ${b.get('limit', 0.0)}, Month: {b.get('month', '')}\n"
        else:
            budgets_str = "No active budgets set for this period."

        # Load and assemble template
        template = self._load_prompt_template()
        system_instruction = template.format(
            user_id=user_id,
            retrieved_records=records_str,
            active_budgets=budgets_str
        )

        user_prompt = f"User Question: {question}"

        # Determine which model is configured in sidebar and use it
        # If Gemini is configured, use it. If not, use Groq.
        gemini_ready = gemini_service.refresh()
        groq_ready = groq_service.refresh()

        if gemini_ready:
            logger.info("ChatAgent: Running Gemini for conversational synthesis...")
            # We pass the system_instruction to Gemini GenerativeModel
            return gemini_service.generate_text(
                prompt=user_prompt,
                system_instruction=system_instruction,
                model_name=None
            )
        elif groq_ready:
            logger.info("ChatAgent: Running Groq/Llama for conversational synthesis...")
            return groq_service.generate_text(
                prompt=user_prompt,
                system_prompt=system_instruction
            )
            
        return "I could not find a working Gemini or Groq API key in the local environment. Add one to the .env file and restart Streamlit if this message continues."

# Singleton Instance
chat_agent = ChatAgent()
