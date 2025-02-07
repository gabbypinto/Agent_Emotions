import os
import pandas as pd
from PIL import Image
from typing import Dict, Optional


class DatasetHandler:
    def __init__(self, 
                 csv_path: str = "/nas/eclairnas01/users/gpinto/Agent_Emotions/datasets/harmful_image_10000_ann.csv", 
                 image_dir: str = "/nas/eclairnas01/users/gpinto/Agent_Emotions/datasets/all_10000_evenHarmfulUnharmful"):
        print(f"📂 Loading dataset metadata from: {csv_path}")
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.metadata = pd.read_csv(csv_path, encoding="ISO-8859-1")

    def get_sample(self, index: int = 0) -> Optional[Dict]:
        while index < len(self.metadata):
            row = self.metadata.iloc[index]
            image_path = os.path.join(self.image_dir, row["imagePath"])
            
            if os.path.exists(image_path):  # ✅ Only return valid images
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

        print("❌ No valid images found in dataset.")
        return None
