from modules.video_processor import VideoProcessor

class VideoAnalyzer:
    def __init__(self):
        """
        Initialize VideoAnalyzer with VideoProcessor.
        """
        self.video_processor = VideoProcessor()

    def analyze_video(self, video_path: str):
        """
        Extract a frame from the video and analyze it.
        """
        frame = self.video_processor.extract_frame(video_path)
        frame_analysis = self.video_processor.analyze_frame(frame)

        analysis = {
            "video_path": video_path,
            "frame_analysis": frame_analysis,
        }
        return analysis


   