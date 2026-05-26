import os
import logging
import pdfplumber
from PIL import Image
import cv2
import numpy as np
from typing import Dict, Any, Optional
from services.gemini_service import gemini_service

logger = logging.getLogger("finance_app.ocr_service")

class OCRService:
    """Orchestrates image/PDF document text and structured finance extraction."""
    def __init__(self):
        # We can configure tesseract command path if local environment variable exists
        self.tesseract_cmd = os.getenv("TESSERACT_CMD_PATH")
        if self.tesseract_cmd:
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
                logger.info(f"Tesseract path bound to: {self.tesseract_cmd}")
            except Exception as e:
                logger.error(f"Error setting custom pytesseract path: {e}")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract native text characters from a PDF file using pdfplumber."""
        logger.info(f"Scanning PDF receipt: {pdf_path}")
        text_content = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_content.append(extracted)
            return "\n".join(text_content)
        except Exception as e:
            logger.error(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return ""

    def process_receipt(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Orchestrate OCR analysis on an uploaded image/PDF file path.
        Returns a structured dictionary with merchant, amount, category, taxes, date, and items.
        """
        logger.info(f"Beginning OCR scanning pipeline on {file_path}")
        ext = os.path.splitext(file_path)[1].lower()

        # Handle PDF natively first
        if ext == ".pdf":
            pdf_text = self.extract_text_from_pdf(file_path)
            if pdf_text.strip():
                return self._parse_raw_text_via_llm(pdf_text)
            # If native extraction was empty, convert first page of PDF to image and do multimodal OCR?
            # For simplicity, if PDF text is empty, we fall back to image-based processors if possible.
            logger.warning("PDF has no native text. Treating as scanned document.")

        # Read file bytes for Gemini Vision OCR
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except Exception as e:
            logger.error(f"Failed to read file bytes: {e}")
            return None

        # Determine MIME type
        mime_type = "image/jpeg"
        if ext == ".png":
            mime_type = "image/png"
        elif ext == ".pdf":
            mime_type = "application/pdf"

        # --- Stage 1: Cloud Multimodal Gemini Vision OCR ---
        if gemini_service.initialized:
            logger.info("Stage 1 OCR: Triggering Gemini multimodal vision extraction...")
            ocr_prompt = """
            Extract structured details from this receipt, bill, or payment screenshot.
            Locate and parse:
            1. Merchant name
            2. Total transaction amount
            3. Tax / VAT amount
            4. Purchase date (YYYY-MM-DD format - infer year if missing, relative to current date 2026-05-25)
            5. Payment mode (Cash, Card, UPI, Bank Transfer)
            6. Itemized list of purchased products (each with 'name', 'price', and optional 'quantity')
            7. Brief helpful note summarizing the transaction

            Return a valid JSON object matching this structure EXACTLY:
            {
                "merchant": "Merchant Name",
                "amount": 42.50,
                "tax": 3.40,
                "date": "2026-05-25",
                "payment_mode": "Card",
                "notes": "Purchased groceries and household items",
                "items": [
                    {"name": "Apples", "price": 12.00, "quantity": 3},
                    {"name": "Laundry Detergent", "price": 14.50, "quantity": 1}
                ],
                "category": "Food"
            }
            Make sure 'amount' and 'tax' are floats. Category must be one of: Food, Travel, Shopping, Rent, Entertainment, Education, Healthcare, Subscriptions, Utilities, Other.
            """
            
            structured_data = gemini_service.analyze_image(
                image_bytes=file_bytes,
                mime_type=mime_type,
                prompt=ocr_prompt,
                system_instruction="You are an expert OCR receipt parsing intelligence designed to extract clean, precise transaction data into JSON."
            )
            if structured_data:
                logger.info(f"Gemini Stage 1 OCR parsed data successfully: {structured_data.get('merchant')}")
                return structured_data

        # --- Stage 2: Local Preprocessing & Pytesseract OCR Fallback ---
        logger.info("Stage 2 OCR: Cloud key missing or failed. Initiating local Tesseract OCR engine...")
        local_text = self._run_local_tesseract_ocr(file_path)
        if local_text.strip():
            logger.info("Local OCR successfully read text bytes. Parsing structure via LLM text models...")
            return self._parse_raw_text_via_llm(local_text)
        
        # If both fail, return structured fallback mock response to avoid breaking client forms
        logger.warning("All OCR pipelines failed or returned empty results. Generating blank manual form.")
        return {
            "merchant": "Unknown Merchant",
            "amount": 0.0,
            "tax": 0.0,
            "date": "2026-05-25",
            "payment_mode": "Cash",
            "notes": "OCR Failed. Please enter details manually.",
            "items": [],
            "category": "Other"
        }

    def _run_local_tesseract_ocr(self, img_path: str) -> str:
        """Preprocess image and execute local Tesseract OCR extraction."""
        try:
            import pytesseract
            # Read image via OpenCV
            img = cv2.imread(img_path)
            if img is None:
                # If cv2 fails, try opening with Pillow directly
                pil_img = Image.open(img_path)
                return pytesseract.image_to_string(pil_img)

            # Gray scaling
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Denoising
            denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
            # Thresholding
            thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            # Convert to PIL Image for Pytesseract
            pil_img = Image.fromarray(thresh)
            text = pytesseract.image_to_string(pil_img)
            return text
        except ImportError:
            logger.warning("pytesseract package is not installed. Skipping local engine.")
            return ""
        except Exception as e:
            logger.error(f"Local PyTesseract execution failed: {e}")
            return ""

    def _parse_raw_text_via_llm(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Pass raw extracted text block to LLM to structure as clean JSON."""
        parsing_prompt = f"""
        Extract clean financial receipt fields from this OCR scanned text:
        ---
        {raw_text}
        ---

        Return ONLY a valid JSON object of this format:
        {{
            "merchant": "Merchant Name",
            "amount": 0.0,
            "tax": 0.0,
            "date": "YYYY-MM-DD",
            "payment_mode": "Card",
            "notes": "Short details",
            "items": [
                {{"name": "item", "price": 0.0, "quantity": 1}}
            ],
            "category": "Other"
        }}
        Category must be one of: Food, Travel, Shopping, Rent, Entertainment, Education, Healthcare, Subscriptions, Utilities, Other.
        """
        
        # Try Gemini text parsing first, then fallback to Groq if available
        if gemini_service.initialized:
            return gemini_service.generate_structured_json(parsing_prompt)
        # Add Groq LLM parsing fallback
        from services.groq_service import groq_service
        if groq_service.initialized:
            return groq_service.generate_structured_json(parsing_prompt)
        
        # Absolute basic manual fallback
        return {
            "merchant": "Scanned Document",
            "amount": 0.0,
            "tax": 0.0,
            "date": "2026-05-25",
            "payment_mode": "Cash",
            "notes": "Auto-parsed fallback due to missing LLM keys",
            "items": [],
            "category": "Other"
        }

# Singleton Instance
ocr_service = OCRService()
