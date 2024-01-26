# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

MODEL_NAME = 'amrul-hzz/watermark_detector'        

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = ViTForImageClassification.from_pretrained(MODEL_NAME)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(MODEL_NAME)

    def predict(
        self,
        image: Path = Input(description="Input image"),
    ) -> str:
        """Run a single prediction on the model"""
        img = Image.open(image)
        inputs = self.feature_extractor(images=img, return_tensors="pt")
        outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        result = "Predicted:" + self.model.config.id2label[predicted_class_idx]
        return result
    