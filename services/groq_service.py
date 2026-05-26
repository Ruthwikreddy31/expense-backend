import os
import json
import logging
from typing import Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("finance_app.groq_service")

class GroqService:
    """Service wrapper for Groq API."""
    def __init__(self):
        load_dotenv(override=True)
        self.api_key = os.getenv("GROQ_API_KEY")
        self.default_model = "llama-3.3-70b-versatile"
        self.client = None
        self.initialized = False
        self._configure()

    def _configure(self):
        if self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                self.initialized = True
                logger.info("Groq API client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
        else:
            logger.warning("GROQ_API_KEY not found in environment. Groq functions will be bypassed.")

    def refresh(self) -> bool:
        """Reload environment keys and initialize Groq if a key was added after startup."""
        load_dotenv(override=True)
        latest_key = os.getenv("GROQ_API_KEY")
        if latest_key != self.api_key or not self.initialized:
            self.api_key = latest_key
            self.client = None
            self.initialized = False
            self._configure()
        return self.initialized

    def generate_text(self, prompt: str, system_prompt: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """Generate text response using Groq."""
        self.refresh()
        if not self.initialized:
            logger.error("Groq client not initialized.")
            return "Error: Groq API key missing."
        
        model_to_use = model_name or self.default_model
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=model_to_use,
                temperature=0.1
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API call failed: {e}")
            return f"Error in Groq execution: {e}"

    def generate_structured_json(self, prompt: str, system_prompt: Optional[str] = None, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate structured JSON using Groq, utilizing JSON Mode where possible."""
        self.refresh()
        if not self.initialized:
            logger.error("Groq client not initialized.")
            return None
        
        model_to_use = model_name or self.default_model
        
        # Enforce JSON formatting
        enforced_prompt = f"{prompt}\n\nReturn ONLY a valid JSON object. Do not include any introductory or concluding text."
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": enforced_prompt})

        try:
            # Groq supports json_object response format
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=model_to_use,
                response_format={"type": "json_object"},
                temperature=0.1
            )
            text = chat_completion.choices[0].message.content
            clean_text = self._clean_json_markdown(text)
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"Groq structured JSON generation failed: {e}. Retrying without JSON response_format...")
            try:
                # Regular text generation fallback
                chat_completion = self.client.chat.completions.create(
                    messages=messages,
                    model=model_to_use,
                    temperature=0.1
                )
                text = chat_completion.choices[0].message.content
                clean_text = self._clean_json_markdown(text)
                return json.loads(clean_text)
            except Exception as e2:
                logger.error(f"Groq fallback JSON parsing failed: {e2}")
                return None

    def _clean_json_markdown(self, text: str) -> str:
        """Strip markdown code block indicators from LLM output."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```json") or lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return text

# Singleton Instance
groq_service = GroqService()
