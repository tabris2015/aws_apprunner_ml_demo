from ultralytics import YOLO
from src.models import Detection, PredictionType
from src.config import get_settings

SETTINGS = get_settings()


class PedestrianDetector:
    def __init__(self) -> None:
        self.model = YOLO(SETTINGS.yolo_version)

    def predict_image(self, image_array, threshold):
        results = self.model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        #   0 = person
        # [0, 2, 0, 3, 0]
        #  0  1  2  3  4
        # [0, 2, 4]
        #

        indexes = [
            i for i in range(len(labels)) if labels[i] in [0, 15, 16]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        detection = Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            label="person",
            confidences=confidences,
        )
        return detection