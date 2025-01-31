import os
import cv2
import torch
import numpy as np
import faiss
from PIL import Image
from typing import List, Dict, Optional
import json
import re

from langsmith import trace  # Import LangSmith tracing
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema



class VideoAnalyzer:
    """
    Modular Video Analyzer for generating descriptions & contextual questions using multiple LLMs (Ollama, Gemini).
    """

    def __init__(self, model: str = "ollama", google_api_key: Optional[str] = None):
        """
        Initialize with the chosen model (ollama or gemini) and API key if needed.
        """
        self.model_name = model.lower()

        if self.model_name == "ollama":
            print("Loading Ollama LLM...")
            self.llm = ChatOllama(model="llama3")

        elif self.model_name == "gemini":
            print("Loading Gemini LLM...")
            api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set. Please set it as an environment variable or pass it directly.")
            self.llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=api_key)

        else:
            raise ValueError(f"Unsupported model: {model}. Use 'ollama' or 'gemini'.")

        self.question_types = ["What", "Where", "Who", "When", "Why", "How"]

        self.video_analysis_prompt = PromptTemplate(
            template="""
            System: You are a video content analyzer specializing in understanding video context.

            User: Analyze this video content and provide a detailed description:
            {video_description}

            Generate a comprehensive description that covers:
            1. What's happening in the video
            2. Main subjects/people
            3. Setting and environment
            4. Notable actions or events
            5. Overall context and purpose

            Description:
            """,
            input_variables=["video_description"]
        )

    def extract_frame(self, video_path: str) -> Image.Image:
        """
        Extracts a middle frame from the given video file.
        """
        print(f"\nExtracting a frame from {video_path}...")
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)  # Grab middle frame
            ret, frame = cap.read()
            cap.release()

            if not ret:
                raise ValueError("Failed to extract frame from video.")

            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        except Exception as e:
            print(f"Error extracting frame: {e}")
            raise

    def analyze_frame(self, frame: Image.Image) -> Dict:
        """
        Analyzes a single frame and returns a textual description.
        """
        print("\nAnalyzing frame...")
        try:
            return {"description": "Frame extracted successfully. Contains visual elements related to human activity.", "frame_count": 1}
        except Exception as e:
            print(f"Error analyzing frame: {e}")
            raise

    def generate_description(self, video_analysis: Dict) -> str:
        """
        Generates a textual description of the video frame.
        """
        print(f"\nGenerating video description using {self.model_name}...")
        try:
            llm_chain = LLMChain(llm=self.llm, prompt=self.video_analysis_prompt)
            response = llm_chain.run(video_description=video_analysis["description"])
            return response.strip()
        except Exception as e:
            print(f"Error generating description: {e}")
            return ""

    def clean_json_output(self, raw_output: str) -> str:
        """
        Cleans JSON output by removing unnecessary formatting from Gemini responses.
        """
        return re.sub(r"```json|```", "", raw_output).strip()

    def generate_questions(self, video_description: str) -> Dict[str, List[str]]:
        """
        Generates contextual questions based on the video description.
        """
        print(f"\nGenerating questions using {self.model_name}...")
        questions = {}

        response_schema = ResponseSchema(
            name="questions",
            description="A list of exactly three questions related to the given question type."
        )
        parser = StructuredOutputParser.from_response_schemas([response_schema])

        question_prompt = PromptTemplate(
            input_variables=["question_type", "video_description", "format_instructions"],
            template="""
            You are an AI designed to generate structured JSON output. Do not include any explanations or introductory text.

            Generate exactly three questions of type "{question_type}" based on the following video description:

            Video Description:
            {video_description}

            Format the output as valid JSON:
            {format_instructions}
            """
        )

        for qtype in self.question_types:
            try:
                llm_chain = LLMChain(llm=self.llm, prompt=question_prompt)
                response = llm_chain.run(
                    question_type=qtype,
                    video_description=video_description,
                    format_instructions=parser.get_format_instructions()
                )

                if self.model_name == "gemini":
                    response = self.clean_json_output(response)

                parsed_output = parser.parse(response)
                questions[qtype] = parsed_output.get("questions", [])

            except json.JSONDecodeError:
                print(f"Warning: {self.model_name} did not return valid JSON for {qtype}.")
                questions[qtype] = [q.strip() for q in response.split("\n") if q.strip()]

            except Exception as e:
                print(f"Error generating questions for {qtype}: {e}")
                questions[qtype] = []

        return questions

    def select_questions(self, questions: Dict[str, List[str]], video_description: str, k: int = 2) -> List[str]:
        """
        Selects the most relevant questions using FAISS.
        """
        print("\nSelecting the most relevant questions...")

        dimension = 384  # Assuming a text embedding dimension of 384
        index = faiss.IndexFlatL2(dimension)

        # Convert video description to vector (simulated)
        desc_vector = np.random.rand(1, dimension).astype('float32')
        index.add(desc_vector)

        selected_questions = []
        for qlist in questions.values():
            if not qlist:
                continue

            q_vectors = np.random.rand(len(qlist), dimension).astype('float32')
            D, I = index.search(q_vectors, k)

            for idx in I[0][:k]:
                if idx < len(qlist):
                    selected_questions.append(qlist[idx])

        return selected_questions


# Ensure API key is set
if "GOOGLE_API_KEY" not in os.environ:
    raise ValueError("GOOGLE_API_KEY is not set. Please configure it in your environment.")
# Debugging (optional): Verify API key is correctly set
print("Using GOOGLE_API_KEY:", os.getenv("GOOGLE_API_KEY"))  # Remove this in production

def main(model_name="ollama"):
    # Configuration
    MODEL_NAME = "meta-llama/Llama-3.2-3B"

    video_path = "/nas/eclairnas01/users/gpinto/Agent_Emotions/vids/@_americannana__video_7305501771227516202.mp4"

    # # Choose model: "ollama" or "gemini"
    # model_name = "gemini"  # Change to "ollama" for Ollama
    
    # Ensure API key is set for Gemini
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if model_name == "gemini" and not google_api_key:
        print("Error: GOOGLE_API_KEY is not set. Set it in your environment before running this script.")
        return

    # Initialize analyzer
    video_analyzer = VideoAnalyzer(model=model_name, google_api_key=google_api_key)

    # Extract & analyze video frame
    frame = video_analyzer.extract_frame(video_path)
    video_analysis = video_analyzer.analyze_frame(frame)

    # Generate description
    description = video_analyzer.generate_description(video_analysis)
    print("\nVideo Description:\n", description)

    # Generate questions
    questions = video_analyzer.generate_questions(description)
    print("\nGenerated Questions:\n", questions)

    # Select relevant questions
    selected_questions = video_analyzer.select_questions(questions, description)
    print("\nSelected Questions:\n", selected_questions)

if __name__ == "__main__":
    print("*"*50,"OLLAMA","*"*50)
    main(model_name='ollama')
    print("\n\n")
    print("*"*50,"GEMINI","*"*50)
    main(model_name='gemini')