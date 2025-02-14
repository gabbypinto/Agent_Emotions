import cv2
from PIL import Image
from typing import Dict
from modules.image_processor import ImageProcessor


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

    # def analyze_frame(self, frame: Image.Image) -> Dict:
    #     return {"description": "Extracted frame content.", "frame_count": 1}
    
    def analyze_image(self, image_path: str):   #synonymous to analyze_frame
        """Analyze an image by generating a textual description using LLM."""
        image = self.image_processor.load_image(image_path)
        image_metadata = self.image_processor.analyze_image(image)

        # Generate a description using LLM
        llm_prompt = f"Describe the following image in detail: {image_metadata['description']}"
        description = self.llm_model.run_chain(prompt=llm_prompt, input_vars={})

        analysis = {
            "image_path": image_path,
            "description": description,
            "metadata": image_metadata,
        }
        self.memory.append(analysis)
        return analysis
