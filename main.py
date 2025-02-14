import wandb
from modules.video_analyzer import VideoAnalyzer
from modules.image_analyzer import ImageAnalyzer
from modules.dataset_handler import DatasetHandler


def main(model_name="llama3.2-vision:11b", persona="default"):
    wandb.init(project="video-analysis", name=f"{model_name}-{persona}-run")

    video_analyzer = VideoAnalyzer()
    image_analyzer = ImageAnalyzer(model=model_name, persona=persona)
    dataset_handler = DatasetHandler()

    # Retrieve two samples (either images or videos)
    dataset_samples = dataset_handler.get_samples(num_samples=1)

    analysis_results = []
    for sample in dataset_samples:
        try:
            if sample["type"] == "image":
                print("*****Sample is an image*****")
                analysis = image_analyzer.analyze_image(sample["image_path"])
            elif sample["type"] == "video":
                analysis = video_analyzer.analyze_video(sample["video_path"])
            else:
                raise ValueError("Invalid sample format.")
            
            analysis_results.append(analysis)
            print(analysis)
        except Exception as e:
            print(f"Error processing sample: {e}")

    # Log analyses to Weights & Biases
    wandb.log({"analysis_results": analysis_results})

if __name__ == "__main__":
    main(model_name="llama3.2-vision:11b", persona="expert")
    # main(model_name="gemini", persona="expert")

