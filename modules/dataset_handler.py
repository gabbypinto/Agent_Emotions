import os
import pandas as pd
from PIL import Image
from typing import Dict, Optional, List


class DatasetHandler:
    def __init__(self, 
                 csv_path: str = "/nas/eclairnas01/users/gpinto/Agent_Emotions/datasets/harmful_image_10000_ann.csv", 
                 media_dir: str = "/nas/eclairnas01/users/gpinto/Agent_Emotions/datasets/all_10000_evenHarmfulUnharmful"):
        """
        Initialize DatasetHandler to load dataset metadata and validate file existence.
        """
        print(f"📂 Loading dataset metadata from: {csv_path}")
        self.csv_path = csv_path
        self.media_dir = media_dir
        self.metadata = pd.read_csv(csv_path, encoding="ISO-8859-1")
        self.missing_files = 0
        self.found_files = 0

    def get_sample(self, index: int = 0) -> Optional[Dict]:
        """
        Retrieve a single valid sample from the dataset, ensuring the file exists.
        Supports both images and videos.
        """
        if index >= len(self.metadata):
            return None
        
    
        row = self.metadata.iloc[index]
        file_path = os.path.join(self.media_dir, row["imagePath"])

        if not os.path.exists(file_path):
            print(f"⚠️ Skipping missing file: {file_path}")
            self.missing_files+=1
            return None

        self.found_files+=1

        if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            return {
                    "type": "image",
                    "image_path": file_path,
                    "decision": row["decision"],
                    "harmfulType": row["harmfulType"],
                    "affirmative_argument_0": row["affirmativeDebater_argument_0"],
                    "affirmative_argument_1": row["affirmativeDebater_argument_1"],
                    "negative_argument_0": row["negativeDebater_argument_0"],
                    "negative_argument_1": row["negativeDebater_argument_1"],
            }

        elif file_path.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            return {
                    "type": "video",
                    "video_path": file_path,
                    "decision": row["decision"],
                    "harmfulType": row["harmfulType"],
                    "affirmative_argument_0": row["affirmativeDebater_argument_0"],
                    "affirmative_argument_1": row["affirmativeDebater_argument_1"],
                    "negative_argument_0": row["negativeDebater_argument_0"],
                    "negative_argument_1": row["negativeDebater_argument_1"],
            }


        print("❌ No valid media files found in dataset.")
        return None

    def get_samples(self, num_samples: int = 2) -> List[Dict]:
        """
        Retrieve multiple valid samples from the dataset.
        :param num_samples: Number of samples to return.
        :return: List of valid dataset samples.
        """
        samples = []
        index = 0

        while len(samples) < num_samples and index < len(self.metadata):
            sample = self.get_sample(index)
            if sample:
                samples.append(sample)
            index += 1  # Move to the next sample

        # Final check to ensure we got exactly `num_samples`
        if len(samples) < num_samples:
            print(f"⚠️ WARNING: Only {len(samples)} valid samples found out of {num_samples} requested.")

        print(f"✅ Found {self.found_files} valid files. ❌ Skipped {self.missing_files} missing files.")
        return samples
