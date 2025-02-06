import os
import cv2
import torch
import numpy as np
import faiss
from PIL import Image
from typing import List, Dict, Optional
import json
import re
from collections import defaultdict
from datetime import datetime
import wandb
import pandas as pd
from datasets import load_dataset

from langsmith import trace
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


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


class VideoProcessor:
    def extract_frame(self, video_path: str) -> Image.Image:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Failed to extract frame.")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def analyze_frame(self, frame: Image.Image) -> Dict:
        return {"description": "Extracted frame content with human activity.", "frame_count": 1}


class DatasetHandler:
    def __init__(self, dataset_name: str = "denny3388/VHD11K", csv_path: str = "/nas/eclairnas01/users/gpinto/Agent_Emotions/datasets/harmful_image_10000_ann.csv", image_dir: str = "/nas/eclairnas01/users/gpinto/Agent_Emotions/datasets/all_10000_evenHarmfulUnharmful"):
        print(f"Loading dataset: {dataset_name}")
        
        # Load metadata CSV
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.metadata = pd.read_csv(csv_path, encoding="ISO-8859-1")

    def get_sample(self, index: int = 0) -> Dict:
        """Retrieve an annotated image sample with metadata, skipping missing images."""
        for index, row in self.metadata.iterrows():
            image_path = os.path.join(self.image_dir, row["imagePath"])
            
            if os.path.exists(image_path):  # Only return valid images
                return {
                    "image": Image.open(image_path),
                    "image_path": image_path,
                    "decision": row["decision"],
                    "harmfulType": row["harmfulType"],
                    "affirmative_argument_0": row["affirmativeDebater_argument_0"],
                    "affirmative_argument_1": row["affirmativeDebater_argument_1"],
                    "negative_argument_0": row["negativeDebater_argument_0"],
                    "negative_argument_1": row["negativeDebater_argument_1"],
                }
        
        print("No valid images found in dataset.")
        return None  # If no valid images are found


class QuestionGenerator:
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model
        self.question_types = ["What", "Where", "Who", "When", "Why", "How"]
        self.question_prompt = PromptTemplate(
            input_variables=["question_type", "video_description", "format_instructions", "persona"],
            template="""
            You are an AI with the persona of {persona}, generating structured JSON output. 
            Do not include any explanations or introductory text.

            Generate exactly three questions of type "{question_type}" based on the following video description:

            Video Description:
            {video_description}

            Format the output as valid JSON:
            {format_instructions}
            """
        )
        self.response_schema = ResponseSchema(
            name="questions",
            description="A list of exactly three questions related to the given question type."
        )
        self.parser = StructuredOutputParser.from_response_schemas([self.response_schema])

    def generate_questions(self, video_description: str) -> Dict[str, List[str]]:
        print(f"\nGenerating questions using {self.llm_model.model_name} with persona {self.llm_model.persona}...")
        questions = {}
        for qtype in self.question_types:
            try:
                response = self.llm_model.run_chain(self.question_prompt, {
                    "question_type": qtype,
                    "video_description": video_description,
                    "format_instructions": self.parser.get_format_instructions()
                })
                parsed_output = self.parser.parse(response)
                questions[qtype] = parsed_output.get("questions", [])
            except Exception as e:
                print(f"Error generating questions for {qtype}: {e}")
                questions[qtype] = []
        return questions

    def select_questions(self, questions: Dict[str, List[str]], video_description: str, k: int = 2) -> List[str]:
        print("\nSelecting the most relevant questions...")
        dimension = 384  # Assuming a text embedding dimension of 384
        index = faiss.IndexFlatL2(dimension)
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


class VideoAnalyzer:
    def __init__(self, model: str = "ollama", google_api_key: Optional[str] = None, persona: str = "default"):
        self.llm_model = LLMModel(model, google_api_key, persona)
        self.video_processor = VideoProcessor()
        self.dataset_handler = DatasetHandler()
        self.question_generator = QuestionGenerator(self.llm_model)
        self.memory = []  # 🧠 Store past analyses

        self.pre_event_prompt = PromptTemplate(
            template="""
            System: You are an AI agent with the persona of {persona}, preparing for video analysis.
            
            User: Before analyzing the video, summarize your understanding of the topic:
            {topic}
            """,
            input_variables=["topic", "persona"]
        )

        self.post_event_prompt = PromptTemplate(
            template="""
            System: You are an AI agent with the persona of {persona}, having just analyzed a video.
            
            User: After analyzing the video, summarize any changes in your understanding of the topic:
            {topic}

            Additionally, recall past analyses and reflect on any patterns or shifts in interpretation.
            """,
            input_variables=["topic", "persona"]
        )

    def analyze_video(self, video_path: Optional[str] = None, dataset_sample: Optional[Dict] = None) -> Dict:
        """Analyze a video or dataset sample and store the results in memory."""
        if video_path:
            print("\nAnalyzing local video file...")
            analysis = self.video_processor.analyze_frame(self.video_processor.extract_frame(video_path))
        elif dataset_sample:
            print("\nAnalyzing dataset sample...")
            analysis = {
                "image_path": dataset_sample["image_path"],
                "decision": dataset_sample["decision"],
                "harmfulType": dataset_sample["harmfulType"],
                "affirmative_argument_0": dataset_sample["affirmative_argument_0"],
                "affirmative_argument_1": dataset_sample["affirmative_argument_1"],
                "negative_argument_0": dataset_sample["negative_argument_0"],
                "negative_argument_1": dataset_sample["negative_argument_1"],
            }
        else:
            raise ValueError("Either a video file path or a dataset sample must be provided.")

        # 🧠 Store analysis in memory
        self.memory.append(analysis)
        return analysis

    def analyze_topic_before(self, topic: str) -> str:
        """Analyze a topic before processing any video/image."""
        response = self.llm_model.run_chain(self.pre_event_prompt, {"topic": topic})
        self.memory.append({"topic": topic, "before": response})  # Store initial topic perception
        return response

    def analyze_topic_after(self, topic: str) -> str:
        """Analyze a topic after processing and recall past analyses."""
        past_analyses = json.dumps(self.memory, indent=4)  # Convert memory to readable format
        response = self.llm_model.run_chain(self.post_event_prompt, {"topic": topic, "past_analyses": past_analyses})
        
        # Update memory with post-analysis reflections
        self.memory.append({"topic": topic, "after": response})
        return response

    def generate_summary(self) -> Dict:
        """Generate a final summary including all past analyses."""
        return {"topic_history": self.memory}


def main(model_name="ollama", persona="default", video_path: Optional[str] = None, use_dataset: bool = False):
    wandb.init(project="video-analysis", name=f"{model_name}-{persona}-run")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    video_analyzer = VideoAnalyzer(model=model_name, google_api_key=google_api_key, persona=persona)
    dataset_handler = DatasetHandler()

    topic = "2024 U.S. Presidential Election"
    print("\nBefore Analysis:")
    before_analysis = video_analyzer.analyze_topic_before(topic)
    print(before_analysis)

    if use_dataset:
        # Fetch an image sample with metadata
        dataset_sample = dataset_handler.get_sample()

        while dataset_sample is None:  # Keep trying until a valid image is found
            print("Skipping missing sample... Searching for a valid image.")
            dataset_sample = dataset_handler.get_sample()

        print("\n===== Analyzing Dataset Sample =====")
        analysis_results = video_analyzer.analyze_video(dataset_sample=dataset_sample)
        print(analysis_results)

    elif video_path:
        analysis_results = video_analyzer.analyze_video(video_path=video_path)

    else:
        raise ValueError("Either provide a video path or enable dataset usage.")

    print("\nAfter Analysis:")
    after_analysis = video_analyzer.analyze_topic_after(topic)
    print(after_analysis)

    summary = video_analyzer.generate_summary()
    print("Final Summary:", json.dumps(summary, indent=4))
    wandb.log(summary)  # 🧠 Log everything, including memory


if __name__ == "__main__":
    print("\nRunning Analysis with Ollama on Dataset:")
    main(model_name='ollama', persona='expert', use_dataset=True)
    print("\nRunning Analysis with Gemini on Local Video:")
    main(model_name='gemini', persona='neutral', video_path='/nas/eclairnas01/users/gpinto/Agent_Emotions/vids/@_americannana__video_7305501771227516202.mp4')



#/nas/eclairnas01/users/gpinto/Agent_Emotions/datasets/all_10000_evenHarmfulUnharmful/
#/nas/eclairnas01/users/gpinto/Agent_Emotions/datasets/8_all_videos_v3_downsampled/