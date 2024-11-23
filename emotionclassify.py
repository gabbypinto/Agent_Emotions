import os
import json
import logging
from typing import Dict, Any, Optional
import speech_recognition as sr
from pydub import AudioSegment
from transformers import pipeline
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('emotion_analyzer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AudioEmotionAnalyzer:
    def __init__(self, model_name: str = "michellejieli/emotion_text_classifier"):
        """
        Initialize the AudioEmotionAnalyzer with specified models and configurations.
        
        Args:
            model_name (str): Name of the emotion classification model to use
        """
        self.recognizer = sr.Recognizer()
        try:
            self.emotion_classifier = pipeline(
                "text-classification", 
                model=model_name, 
                return_all_scores=True
            )
            logger.info("Emotion classifier loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load emotion classifier: {str(e)}")
            raise

    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """
        Extract audio from video file.
        
        Args:
            video_path (str): Path to input video file
            output_path (str): Path to save extracted audio
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            audio = AudioSegment.from_file(video_path)
            audio.export(output_path, format="wav")
            logger.info(f"Audio extracted successfully to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to extract audio: {str(e)}")
            return False

    def transcribe_audio(self, audio_path: str) -> Optional[str]:
        """
        Convert audio to text using speech recognition.
        
        Args:
            audio_path (str): Path to audio file
            
        Returns:
            Optional[str]: Transcribed text or None if failed
        """
        try:
            with sr.AudioFile(audio_path) as source:
                audio_data = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio_data)
                logger.info("Audio transcribed successfully")
                return text
        except sr.UnknownValueError:
            logger.warning("Speech Recognition could not understand the audio")
            return None
        except sr.RequestError as e:
            logger.error(f"Speech Recognition service error: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return None

    def analyze_emotions(self, text: str) -> Optional[list]:
        """
        Analyze emotions in the given text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Optional[list]: List of emotion scores or None if failed
        """
        try:
            emotions = self.emotion_classifier(text)
            logger.info("Emotion analysis completed successfully")
            return emotions
        except Exception as e:
            logger.error(f"Emotion analysis failed: {str(e)}")
            return None

    def process_video(self, video_path: str, workspace_dir: str = "temp") -> Dict[str, Any]:
        """
        Process a video file and return emotion analysis results.
        
        Args:
            video_path (str): Path to input video file
            workspace_dir (str): Directory for temporary files
            
        Returns:
            Dict[str, Any]: Results dictionary containing status and analysis
        """
        # Create workspace directory if it doesn't exist
        Path(workspace_dir).mkdir(parents=True, exist_ok=True)
        
        audio_path = os.path.join(workspace_dir, "temp_audio.wav")
        
        try:
            # Extract audio
            if not self.extract_audio(video_path, audio_path):
                return {"status": "error", "message": "Audio extraction failed"}

            # Transcribe audio
            transcript = self.transcribe_audio(audio_path)
            if not transcript:
                return {"status": "error", "message": "Transcription failed"}

            # Analyze emotions
            emotions = self.analyze_emotions(transcript)
            if not emotions:
                return {"status": "error", "message": "Emotion analysis failed"}

            result = {
                "status": "success",
                "transcript": transcript,
                "emotions": emotions
            }

        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            result = {"status": "error", "message": str(e)}

        finally:
            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
                logger.info("Temporary files cleaned up")

        return result

def main():
    """
    Main function to demonstrate usage.
    """
    # Example usage
    video_path = "data/gpinto/us_tiktok/videos_by_date/2024/07/07-14/@22mary48_video_7391644336737226026.mp4"
    analyzer = AudioEmotionAnalyzer()
    
    results = analyzer.process_video(video_path)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":