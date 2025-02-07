import wandb
from modules.video_analyzer import VideoAnalyzer
from modules.dataset_handler import DatasetHandler


def main(model_name="ollama", persona="default"):
    wandb.init(project="video-analysis", name=f"{model_name}-{persona}-run")
    video_analyzer = VideoAnalyzer(model=model_name, persona=persona)
    dataset_handler = DatasetHandler()

    dataset_sample = dataset_handler.get_sample()
    if dataset_sample:
        analysis_results = video_analyzer.analyze_video(dataset_sample)
        print(analysis_results)

    wandb.log({"analysis": analysis_results})


if __name__ == "__main__":
    main(model_name="ollama", persona="expert")
