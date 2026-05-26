import os
import json
import logging
import google.generativeai as genai
from typing import Dict, Any, Optional, Union, List
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("finance_app.gemini_service")

class GeminiService:
    """Service wrapper for Google Gemini API."""
    def __init__(self):
        load_dotenv(override=True)
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.default_model_name = os.getenv("MODEL_NAME", "gemini-3.5-flash")
        self.initialized = False
        self._configure()

    def _configure(self):
        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.initialized = True
                logger.info("Gemini API initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to configure Gemini API: {e}")
        else:
            logger.warning("GEMINI_API_KEY not found in environment. Gemini features will be disabled or bypassed.")

    def refresh(self) -> bool:
        """Reload environment keys and initialize Gemini if a key was added after startup."""
        load_dotenv(override=True)
        latest_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        latest_model = os.getenv("MODEL_NAME", self.default_model_name)
        if latest_key != self.api_key or latest_model != self.default_model_name or not self.initialized:
            self.api_key = latest_key
            self.default_model_name = latest_model
            self.initialized = False
            self._configure()
        return self.initialized

    def generate_text(self, prompt: str, system_instruction: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """Generate standard text from a prompt."""
        self.refresh()
        if not self.initialized:
            logger.error("Gemini API is not initialized.")
            return "Error: Gemini API key missing."
        
        model_to_use = model_name or self.default_model_name
        try:
            model = genai.GenerativeModel(
                model_name=model_to_use,
                system_instruction=system_instruction
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini text generation failed: {e}")
            return f"Error in Gemini execution: {e}"

    def generate_structured_json(self, prompt: str, system_instruction: Optional[str] = None, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Generate a structured JSON response from Gemini, ensuring parsing robustness."""
        self.refresh()
        if not self.initialized:
            logger.error("Gemini API is not initialized.")
            return None
        
        model_to_use = model_name or self.default_model_name
        
        # We append a strong instruction to enforce JSON
        enforced_prompt = f"{prompt}\n\nReturn ONLY a valid JSON object. Do not include any introductory or concluding text. Double check brackets and format."
        
        try:
            # We can request JSON output type from Gemini 1.5+
            generation_config = {"response_mime_type": "application/json"}
            model = genai.GenerativeModel(
                model_name=model_to_use,
                system_instruction=system_instruction,
                generation_config=generation_config
            )
            response = model.generate_content(enforced_prompt)
            clean_text = self._clean_json_markdown(response.text)
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"Structured JSON generation failed with generation_config: {e}. Retrying with regex cleanup...")
            # Fallback text generation if config fails
            try:
                model = genai.GenerativeModel(model_name=model_to_use, system_instruction=system_instruction)
                response = model.generate_content(enforced_prompt)
                clean_text = self._clean_json_markdown(response.text)
                return json.loads(clean_text)
            except Exception as e2:
                logger.error(f"Gemini fallback JSON parsing failed: {e2}")
                return None

    def analyze_image(self, image_bytes: bytes, mime_type: str, prompt: str, system_instruction: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Perform multimodal receipt analysis with Gemini's vision capability, returning structured JSON."""
        self.refresh()
        if not self.initialized:
            logger.error("Gemini API is not initialized.")
            return None
        
        model_name = self.default_model_name
        
        try:
            generation_config = {"response_mime_type": "application/json"}
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
                generation_config=generation_config
            )
            
            image_part = {
                "mime_type": mime_type,
                "data": image_bytes
            }
            
            response = model.generate_content([prompt, image_part])
            clean_text = self._clean_json_markdown(response.text)
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"Gemini Image Multimodal analysis failed: {e}")
            return None

    def _clean_json_markdown(self, text: str) -> str:
        """Strip markdown code block indicators (```json ... ```) from LLM output."""
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
gemini_service = GeminiService()
