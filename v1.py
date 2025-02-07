import os
import cv2
import torch
import numpy as np
import faiss
from PIL import Image
from typing import List, Dict, Optional
import json
import wandb
import pandas as pd
from datasets import load_dataset

from langsmith import trace
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
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
        response = llm_chain.invoke(input_vars)  # Returns a dictionary
        return response["text"].strip() if "text" in response else str(response).strip()



class ImageProcessor:
    """Handles all image-related processing."""
    
    def analyze_image(self, image_path: str) -> Dict:
        """Processes an image from a dataset."""
        try:
            image = Image.open(image_path)
            return {
                "image_path": image_path,
                "description": "Analyzed image",
            }
        except Exception as e:
            print(f"Error processing image: {e}")
            return {}


class VideoProcessor:
    """Handles all video-related processing."""
    
    def extract_frame(self, video_path: str) -> Image.Image:
        """Extracts a single frame from a video."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError("Failed to extract frame.")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def analyze_frame(self, frame: Image.Image) -> Dict:
        """Analyzes a single frame."""
        return {"description": "Extracted frame content with human activity.", "frame_count": 1}


class DatasetHandler:
    """Handles dataset loading and sample retrieval."""

    def __init__(self, dataset_name: str = "denny3388/VHD11K",
                 csv_path: str = "/nas/eclairnas01/users/gpinto/Agent_Emotions/datasets/harmful_image_10000_ann.csv",
                 image_dir: str = "/nas/eclairnas01/users/gpinto/Agent_Emotions/datasets/all_10000_evenHarmfulUnharmful"):
        print(f"Loading dataset: {dataset_name}")
        self.dataset_name = dataset_name
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.metadata = pd.read_csv(csv_path, encoding="ISO-8859-1")

    
    def get_sample(self, index: int = 0) -> Optional[Dict]:
        """
        Retrieve an annotated image sample with metadata, skipping missing images.
        :param index: The index of the sample in the CSV.
        :return: Dictionary containing image and metadata.
        """
        while index < len(self.metadata):
            row = self.metadata.iloc[index]
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
            else:
                print(f"⚠️ Skipping missing file: {image_path}")
                index += 1  # Move to next image

        print("No valid images found in dataset.")
        return None

    # def get_sample(self) -> Optional[Dict]:
    #     """Retrieves an annotated image sample, skipping missing images."""
    #     for _, row in self.metadata.iterrows():
    #         image_path = os.path.join(self.image_dir, row["imagePath"])
    #         if os.path.exists(image_path):  # Only return valid images
    #             return {
    #                 "image_path": image_path,
    #                 "decision": row["decision"],
    #                 "harmfulType": row["harmfulType"],
    #                 "affirmative_argument_0": row["affirmativeDebater_argument_0"],
    #                 "affirmative_argument_1": row["affirmativeDebater_argument_1"],
    #                 "negative_argument_0": row["negativeDebater_argument_0"],
    #                 "negative_argument_1": row["negativeDebater_argument_1"],
    #             }
    #     print("No valid images found in dataset.")
    #     return None
    
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
    
class ExplanationStep:
    def __init__(self, llm_model: LLMModel):
        self.llm_model = llm_model
        self.explanation_prompt = PromptTemplate(
            template="""
            System: You are an AI agent with the persona of {persona}, tasked with explaining the content of an image.

            User: Describe this image in detail. What do you see? Include key elements, context, and possible meaning.

            Image Path: {image_path}
            """,
            input_variables=["image_path", "persona"]
        )

    def generate_explanation(self, image_path: str, persona: str) -> str:
        """Generate an explanation of the image content before further processing."""
        print(f"\n Explaining Image Content: {image_path}")
        explanation = self.llm_model.run_chain(self.explanation_prompt, {"image_path": image_path, "persona": persona})
        return explanation


class VideoAnalyzer:
    """Manages video and image analysis, generating insights via LLMs."""
    
    def __init__(self, model: str = "ollama", google_api_key: Optional[str] = None, persona: str = "default"):
        self.llm_model = LLMModel(model, google_api_key, persona)
        self.video_processor = VideoProcessor()
        self.dataset_handler = DatasetHandler()
        self.explanation_step = ExplanationStep(self.llm_model) 
        self.question_generator = QuestionGenerator(self.llm_model)
        self.memory = []

        self.pre_event_prompt = PromptTemplate(
            template="System: You are an AI agent preparing for video analysis.\n\nUser: Before analyzing the video, summarize your understanding of {topic}.",
            input_variables=["topic"]
        )

        self.post_event_prompt = PromptTemplate(
            template="System: You are an AI agent, reflecting after video analysis.\n\nUser: After analyzing {topic}, summarize any changes in your understanding and recall past analyses.",
            input_variables=["topic"]
        )

    def analyze_video(self, dataset_sample: Dict) -> Dict:
        """Explain and analyze a video/image."""
        image_path = dataset_sample["image_path"]

        # 🆕 Step 1: Explain what is in the image
        explanation = self.explanation_step.generate_explanation(image_path, self.llm_model.persona)
        print("\n📝 Explanation:", explanation)

        # 🧠 Store explanation in memory
        self.memory.append({"image_path": image_path, "explanation": explanation})

        # Step 2: Analyze the image
        analysis = self.video_processor.analyze_frame(self.video_processor.extract_frame(image_path))
        self.memory.append({"image_path": image_path, "analysis": analysis})

        return analysis

    # def analyze_video(self, video_path: str) -> Dict:
    #     """Analyzes a video file by extracting and analyzing a frame."""
    #     frame = self.video_processor.extract_frame(video_path)
    #     analysis = self.video_processor.analyze_frame(frame)
    #     self.memory.append(analysis)
    #     return analysis

    def analyze_image(self, image_path: str) -> Dict:
        """Analyzes an image from a dataset."""
        analysis = self.image_processor.analyze_image(image_path)
        self.memory.append(analysis)
        return analysis

    def analyze_topic_before(self, topic: str) -> str:
        """Analyze a topic before processing."""
        response = self.llm_model.run_chain(self.pre_event_prompt, {"topic": topic})
        self.memory.append({"topic": topic, "before": response})
        return response

    def analyze_topic_after(self, topic: str) -> str:
        """Analyze a topic after processing."""
        response = self.llm_model.run_chain(self.post_event_prompt, {"topic": topic})
        self.memory.append({"topic": topic, "after": response})
        return response

    def generate_summary(self) -> Dict:
        """Generate a final summary including all past analyses."""
        return {"topic_history": self.memory}
    


# def main(model_name="ollama", persona="default", video_path: Optional[str] = None, use_dataset: bool = False):
#     wandb.init(project="video-analysis", name=f"{model_name}-{persona}-run")
#     google_api_key = os.getenv("GOOGLE_API_KEY")
#     video_analyzer = VideoAnalyzer(model=model_name, google_api_key=google_api_key, persona=persona)
#     dataset_handler = DatasetHandler()

#     topic = "Emotion expressed from the given image"
#     print("\nBefore Analysis:")
#     before_analysis = video_analyzer.analyze_topic_before(topic)
#     print(before_analysis)

#     if use_dataset:
#         # Fetch an image sample with metadata
#         dataset_sample = dataset_handler.get_sample()
#         if dataset_sample:
#             print("\n===== Analyzing Dataset Sample =====")
#             analysis_results = video_analyzer.analyze_sample(dataset_sample=dataset_sample)
#             print(analysis_results)
#         else:
#             print("Error: Could not retrieve dataset sample.")
#             return

#     elif video_path:
#         # Process a local video file
#         analysis_results = video_analyzer.analyze_video(video_path=video_path)

#     else:
#         raise ValueError("Either provide a video path or enable dataset usage.")

#     print("\nAfter Analysis:")
#     after_analysis = video_analyzer.analyze_topic_after(topic)
#     print(after_analysis)

#     summary = video_analyzer.generate_summary()
#     wandb.log(summary)

def main(model_name="ollama", persona="default", num_experiments: int = 10, repeat_exposure: int = 3):
    """Run multiple experiments using dataset samples as input, including repeated exposure to the same image."""
    wandb.init(project="video-analysis", name=f"{model_name}-{persona}-experiments")
    google_api_key = os.getenv("GOOGLE_API_KEY")
    video_analyzer = VideoAnalyzer(model=model_name, google_api_key=google_api_key, persona=persona)
    dataset_handler = DatasetHandler()

    topic = "The impact of violent content on AI perception"
    print("\nBefore Analysis:")
    before_analysis = video_analyzer.explanation_step.generate_explanation(topic, persona)
    print(before_analysis)

    #Step 1: Run repeated exposure experiment
    dataset_sample = dataset_handler.get_sample(index=0)  # Ensure valid sample
    if dataset_sample is None:
        print("No valid samples found in dataset!")
        return

    print("\n===== Running Repeated Exposure Experiment =====")
    for exposure in range(repeat_exposure):
        print(f"\nExposure {exposure + 1}/{repeat_exposure}: {dataset_sample['image_path']}")
        analysis_results = video_analyzer.analyze_video(dataset_sample=dataset_sample)
        print(analysis_results)

    #Step 2: Run additional experiments with new images
    for experiment_num in range(1, num_experiments + 1):  # Start at 1 to avoid duplicate
        print(f"\n===== Running Experiment {experiment_num}/{num_experiments} =====")

        dataset_sample = dataset_handler.get_sample(index=experiment_num)
        if dataset_sample is None:
            print("No more valid samples found, stopping experiments.")
            break

        print(f"Processing Sample {experiment_num}: {dataset_sample['image_path']}")
        analysis_results = video_analyzer.analyze_video(dataset_sample=dataset_sample)
        print(f"\nExperiment {experiment_num} Analysis:\n", analysis_results)

    #Step 3: Generate final summary
    summary = video_analyzer.generate_summary()
    print("\n Final Summary:", json.dumps(summary, indent=4))
    wandb.log(summary)


if __name__ == "__main__":
    print("\nRunning Analysis on Dataset:")
    # main(model_name='ollama', persona='expert', use_dataset=True)
    print("\nRunning Analysis with Ollama on Dataset:")
    main(model_name='ollama', persona='expert', num_experiments=10, repeat_exposure=5)

    print("\nRunning Analysis with Gemini on Local Video:")
    # main(model_name='gemini', persona='neutral', num_experiments=5, repeat_exposure=3)


    # print("\nRunning Analysis on Local Video:")
    # main(model_name="gemini", persona="neutral", video_path="/nas/eclairnas01/users/gpinto/Agent_Emotions/vids/sample_video.mp4")
    # main(model_name='gemini', persona='neutral', video_path='/nas/eclairnas01/users/gpinto/Agent_Emotions/vids/sample.mp4')
    # main(model_name='gemini', persona='neutral', video_path='/nas/eclairnas01/users/gpinto/Agent_Emotions/vids/@_americannana__video_7305501771227516202.mp4')
