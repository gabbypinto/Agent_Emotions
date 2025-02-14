from PIL import Image
import os
from typing import Dict
from langchain.prompts import PromptTemplate
from modules.image_processor import ImageProcessor
from modules.llm_model import LLMModel
from langchain_core.messages import HumanMessage


import sys


class ImageAnalyzer:
    def __init__(self, model: str = "llama3.2-vision:11b", google_api_key=None, persona: str = "default"):
        """
        Initialize ImageAnalyzer with LLMModel.
        """
        self.image_processor = ImageProcessor() ### ImageProcessor instance created
        self.llm_model = LLMModel(model, google_api_key, persona)

    def analyze_image(self, image_path: str) -> Dict:
        """
        Generate Multimodal Prompt
        """
        # Ensure the image exists before processing
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Image not found: {image_path}")
        
        # extracted_text = self.llm_model.extract_text_from_image(image_path)
        # print(f"📌 Extracted Text: {extracted_text}")


        prompt = (
            "You are a vision model. Look at this image carefully and describe it in detail.\n\n"
            "- What objects do you see?\n"
            "- Are there people in the image? If so, what are they doing?\n"
            "- What is the setting of the image (indoor, outdoor, etc.)?\n"
            "- Describe any text or logos present.\n\n"
            "Provide a detailed, structured description."
        )

        
        description = self.llm_model.execute_prompt(prompt, image_path)
        sys.exit()

        return description


    def extract_metadata(self, image_path: str, image: Image.Image) -> Dict:
        """
        Extract metadata from an image.
        """
        width, height = image.size
        return {
            "format": image.format or "Unknown",
            "mode": image.mode,
            "dimensions": f"{width}x{height}",
            "size_kb": round(os.path.getsize(image_path) / 1024, 2) if os.path.exists(image_path) else None,
        }
