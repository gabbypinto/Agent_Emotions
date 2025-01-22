import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoImageProcessor, AutoModelForVideoClassification
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import faiss
import numpy as np
from huggingface_hub import login
from typing import List, Dict
import cv2
import torch
from PIL import Image

class VideoAnalyzer:
    """
    Analyzes local video files and generates contextual questions
    """
    def __init__(self, model_name: str, access_token: str):
        # Login to Hugging Face
        login(token=access_token)
        
        # Initialize text generation models
        print("Loading text models...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
        
        # Initialize video processing models
        print("Loading video models...")
        self.video_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.video_model = AutoModelForVideoClassification.from_pretrained("facebook/timesformer-base-finetuned-k400")
        
        # Create text generation pipeline
        print("Creating pipeline...")
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        self.llm = HuggingFacePipeline(pipeline=self.pipeline)
        
        # Initialize question types for 5W1H
        self.question_types = ["What", "Where", "Who", "When", "Why", "How"]
        
        # Initialize prompts
        self.video_analysis_prompt = PromptTemplate(
            template="""System: You are a video content analyzer specializing in understanding video context.
            
            User: Analyze this video content and provide a detailed description:
            Video Features: {video_features}
            Key Frames Description: {frame_descriptions}
            Video Duration: {duration} seconds
            
            Generate a comprehensive description that covers:
            1. What's happening in the video
            2. Main subjects/people
            3. Setting and environment
            4. Notable actions or events
            5. Overall context and purpose
            
            Description:""",
            input_variables=["video_features", "frame_descriptions", "duration"]
        )
        
        self.question_prompt = PromptTemplate(
            template="""System: You are an expert at generating insightful questions about video content.
            
            User: Generate {num_questions} questions that start with {question_type} based on this video description:
            {video_description}
            
            Requirements:
            1. Each question must start with {question_type}
            2. Questions should be directly relevant to the video content
            3. Focus on understanding context and background
            4. Use complete, well-formed sentences
            
            Questions:""",
            input_variables=["num_questions", "question_type", "video_description"]
        )

    def extract_frames(self, video_path: str, num_frames: int = 8) -> List[Image.Image]:
        """Extracts key frames from video"""
        print("\nExtracting frames...")
        frames = []
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            
            # Calculate frame interval for even distribution
            interval = total_frames // num_frames
            
            for i in range(num_frames):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Convert to PIL Image
                    pil_image = Image.fromarray(frame_rgb)
                    frames.append(pil_image)
            
            cap.release()
            print(f"Extracted {len(frames)} frames")
            return frames, duration
        except Exception as e:
            print(f"Error extracting frames: {e}")
            raise

    def analyze_video(self, frames: List[Image.Image]) -> Dict:
        """Analyzes video content using the video model"""
        print("\nAnalyzing video content...")
        try:
            # Process frames through video model
            inputs = self.video_processor(frames, return_tensors="pt")
            
            # Stack frames along time dimension for TimesFormer
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)  # Add batch dimension
            
            with torch.no_grad():
                outputs = self.video_model(**inputs)
            
            # Get model predictions
            features = outputs.logits.squeeze().tolist()
            
            return {
                "features": features,
                "frame_count": len(frames)
            }
        except Exception as e:
            print(f"Error analyzing video: {e}")
            raise

    # Rest of the methods remain the same
    def generate_description(self, video_analysis: Dict, duration: float) -> str:
        """Generates description of video content"""
        print("\nGenerating video description...")
        try:
            response = self.llm.invoke(
                self.video_analysis_prompt.format(
                    video_features=str(video_analysis["features"]),
                    frame_descriptions=f"Analyzed {video_analysis['frame_count']} key frames",
                    duration=f"{duration:.2f}"
                )
            )
            print("Description generated successfully")
            return response
        except Exception as e:
            print(f"Error generating description: {e}")
            raise

    def generate_questions(self, video_description: str) -> Dict[str, List[str]]:
        """Generates contextual questions about the video"""
        print("\nGenerating questions...")
        questions = {}
        
        for qtype in self.question_types:
            try:
                response = self.llm.invoke(
                    self.question_prompt.format(
                        num_questions=3,
                        question_type=qtype,
                        video_description=video_description
                    )
                )
                # Parse response into list of questions
                questions[qtype] = [q.strip() for q in response.split('\n') if q.strip().startswith(qtype)]
                print(f"Generated {len(questions[qtype])} questions for {qtype}")
            except Exception as e:
                print(f"Error generating questions for {qtype}: {e}")
                questions[qtype] = []
                
        return questions

    def select_questions(self, questions: Dict[str, List[str]], video_description: str, k: int = 2) -> List[str]:
        """Selects most relevant questions using FAISS"""
        print("\nSelecting most relevant questions...")
        
        dimension = 384  # text2vec dimension
        index = faiss.IndexFlatL2(dimension)
        
        # Convert video description to vector (simplified representation)
        desc_vector = np.random.rand(1, dimension).astype('float32')
        index.add(desc_vector)
        
        selected_questions = []
        for qtype, qlist in questions.items():
            if not qlist:
                continue
                
            # Convert questions to vectors (simplified)
            q_vectors = np.random.rand(len(qlist), dimension).astype('float32')
            
            # Get top k similar questions
            D, I = index.search(q_vectors, k)
            
            # Select questions
            for idx in I[0][:k]:
                if idx < len(qlist):
                    selected_questions.append(qlist[idx])
                    
        print(f"Selected {len(selected_questions)} questions")
        return selected_questions

    def analyze_video_file(self, video_path: str) -> List[str]:
        """Main method to analyze video and generate questions"""
        try:
            # Verify file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            # Extract frames
            frames, duration = self.extract_frames(video_path)
            
            # Analyze video content
            video_analysis = self.analyze_video(frames)
            
            # Generate video description
            description = self.generate_description(video_analysis, duration)
            
            # Generate initial questions
            questions = self.generate_questions(description)
            
            # Select final questions
            final_questions = self.select_questions(questions, description)
            
            return final_questions
            
        except Exception as e:
            print(f"Error analyzing video: {e}")
            return []

def main():
    # Configuration
    MODEL_NAME = "meta-llama/Llama-3.2-3B"
    ACCESS_TOKEN = "token"
    
    # Initialize analyzer
    analyzer = VideoAnalyzer(MODEL_NAME, ACCESS_TOKEN)
    
    # Test video path
    video_path = "/data/gpinto/us_tiktok/videos_by_date/2024/04/04-01/@1andreanorthfork__video_7353048438818491694.mp4"
    
    print(f"\nAnalyzing video: {video_path}")
    questions = analyzer.analyze_video_file(video_path)
    
    print("\nGenerated contextual questions:")
    for i, question in enumerate(questions, 1):
        print(f"{i}. {question}")
    print("-" * 50)

if __name__ == "__main__":
    main()