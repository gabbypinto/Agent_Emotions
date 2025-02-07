import json
from modules.llm_model import LLMModel
from modules.video_processor import VideoProcessor
from modules.dataset_handler import DatasetHandler
from modules.question_generator import QuestionGenerator


class VideoAnalyzer:
    def __init__(self, model: str = "ollama", google_api_key=None, persona: str = "default"):
        self.llm_model = LLMModel(model, google_api_key, persona)
        self.video_processor = VideoProcessor()
        self.dataset_handler = DatasetHandler()
        self.question_generator = QuestionGenerator(self.llm_model)
        self.memory = []

    def analyze_video(self, dataset_sample):
        analysis = {
            "image_path": dataset_sample["image_path"],
            "decision": dataset_sample["decision"],
            "harmfulType": dataset_sample["harmfulType"],
        }
        self.memory.append(analysis)
        return analysis
