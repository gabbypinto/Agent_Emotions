import cv2
from PIL import Image
from typing import Dict


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
