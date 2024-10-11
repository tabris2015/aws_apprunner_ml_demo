from pydantic import BaseModel
from enum import Enum


class PredictionType(str, Enum):
    classification = "CLS"
    object_detection = "OD"
    segmentation = "SEG"


class GeneralPrediction(BaseModel):
    pred_type: PredictionType


class Detection(GeneralPrediction):
    n_detections: int
    boxes: list[list[int]]
    label: str
    confidences: list[float]
