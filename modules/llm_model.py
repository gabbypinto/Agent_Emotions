import os
from typing import Dict, Optional
import ollama
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import base64
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage


# from langchain_ollama import Ollama



class LLMModel:
    def __init__(self, model: str = "ollama", google_api_key: Optional[str] = None, persona: str = "default"):
        self.model_name = model.lower()
        self.persona = persona

        if self.model_name == "ollama":
            self.llm = ChatOllama(model="llama3",temperature=0)
        elif self.model_name == "llama3.2-vision:11b":
            print("Initializing Ollama Llama3.2 Vision Model")
            # self.llm = ollama
            # self.llm = Ollama(model="llama3.2-vision")
            # self.llm = OllamaLLM(model="llama3.2-vision:11b")
            self.llm = ChatOllama(model="llama3.2-vision", temperature=0)
        elif self.model_name == "gemini":
            api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set.")
            self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
    def encode_image(self, image_path: str) -> str:
        """Convert an image to base64 encoding for API input."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
        
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extracts literal text from an image using Llama3.2-Vision and `StrOutputParser()`.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"❌ Image file not found: {image_path}")

        # ✅ Convert image to Base64
        image_b64 = self.encode_image(image_path)

        # ✅ Use LangChain's `StrOutputParser()`
        parser = StrOutputParser()

        # ✅ Define prompt function to pass image as Base64
        def prompt_func(data):
            text = data["text"]
            image = data["image"]

            image_part = {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{image}",  # ✅ Correct format for Llama3.2-Vision
            }

            text_part = {"type": "text", "text": text}

            return [HumanMessage(content=[image_part, text_part])]

        # ✅ Create a structured prompt for text extraction
        chain = prompt_func | self.llm | parser

        # ✅ Invoke model with text extraction query
        response = chain.invoke(
            {"text": "Only extract text contained in the image.", "image": image_b64}
        )

        # ✅ Invoke model with text extraction query
        # response = chain.invoke({"text": "Only extract text of this image.", "image": image_b64})

        # print(f"📌 Extracted Text: {response.strip()}")
        return response.strip()

    
    def execute_prompt(self, prompt: str, image_path: Optional[str] = None ) -> str:
        """
        Run the LLM on a given prompt and return the response.
        """
        if self.model_name == "llama3.2-vision:11b":
            if not os.path.exists(image_path) or not image_path:
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # ✅ Convert image to Base64
            image_b64 = self.encode_image(image_path)

            # ✅ Use LangChain's `StrOutputParser()`
            parser = StrOutputParser()

            # ✅ Define prompt function to pass image as Base64
            def prompt_func(data):
                text = data["text"]
                image = data["image"]

                image_part = {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{image}",  # ✅ Correct format for Llama3.2-Vision
                }

                text_part = {"type": "text", "text": text}

                return [HumanMessage(content=[image_part, text_part])]

            # ✅ Create a structured prompt for text extraction
            chain = prompt_func | self.llm | parser

           # ✅ Invoke model with text extraction query
            response = chain.invoke(
                    {"text": prompt, "image": image_b64}
            )

            # ✅ Invoke model with text extraction query
            # response = chain.invoke({"text": prompt, "image": image_b64})

            print(f"📌 Extracted Description: {response.strip()}")
            return response.strip()


        # ✅ Default behavior for text-based LLMs
        elif self.model_name == "gemini":
            
            if not os.path.exists(image_path) or not image_path:
                raise FileNotFoundError(f"Image file not found: {image_path}")

            print("Image path:",image_path)
            # ✅ Fix: Use correct prompt formatting for multimodal input
            multimodal_prompt = f"{prompt}"

            response = self.llm.invoke(multimodal_prompt) 
            print("response)")

            return response
        return self.llm.invoke(input=prompt).strip()