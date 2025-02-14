from PIL import Image
import os

class ImageProcessor:
    def __init__(self, supported_formats=("JPEG", "PNG")):
        """
        Initialize the ImageProcessor with supported image formats.
        """
        self.supported_formats = supported_formats

    def load_image(self, image_path: str) -> Image.Image:
        """
        Load an image from the specified file path and verify the format.
        :param image_path: Path to the image file.
        :return: PIL Image object if valid, else raises an error.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        try:
            image = Image.open(image_path)
            if image.format not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {image.format}. Supported formats: {self.supported_formats}")
            return image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

