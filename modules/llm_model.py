import os
from typing import Dict, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI


class LLMModel:
    def __init__(self, model: str = "ollama", google_api_key: Optional[str] = None, persona: str = "default"):
        self.model_name = model.lower()
        self.persona = persona
        if self.model_name == "ollama":
            self.llm = ChatOllama(model="llama3")
        elif self.model_name == "gemini":
            api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set.")
            self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)
        else:
            raise ValueError(f"Unsupported model: {model}")
    
    def run_chain(self, prompt: PromptTemplate, input_vars: Dict) -> str:
        input_vars["persona"] = self.persona  # Inject persona into prompt
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        return llm_chain.run(**input_vars).strip()
