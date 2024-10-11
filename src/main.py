import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image
from src.predictor import PedestrianDetector, Detection
from src.config import get_settings

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_object_detector():
    print("Creating model...")
    return PedestrianDetector()


def predict_uploadfile(predictor, file, threshold):
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    img_obj = Image.open(img_stream)
    # crear array de numpy
    img_array = np.array(img_obj)
    return predictor.predict_image(img_array, threshold), img_array


def annotate_image(image_array, prediction: Detection):
    ann_color = (255, 255, 0)
    annotated_img = image_array.copy()
    for box, conf in zip(prediction.boxes, prediction.confidences):
        cv2.rectangle(annotated_img, (box[0], box[1]), (box[2], box[3]), ann_color, 3)
        cv2.putText(
            annotated_img,
            prediction.label,
            (box[0], box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
        cv2.putText(
            annotated_img,
            f"{conf:.1f}",
            (box[0], box[3] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    ## annotation
    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)

    return image_stream


@app.post("/objects")
def detect_pedestrians(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    predictor: PedestrianDetector = Depends(get_object_detector),
) -> Detection:
    results, _ = predict_uploadfile(predictor, file, threshold)

    return results


@app.post("/annotate_objects")
def annotate_pedestrians(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    predictor: PedestrianDetector = Depends(get_object_detector),
) -> Response:
    results, img = predict_uploadfile(predictor, file, threshold)
    annotated_img_stream = annotate_image(img, results)
    # return {"success": True}
    return Response(content=annotated_img_stream.read(), media_type="image/jpeg")


@app.get("/info")
def get_model_info():
    data = {
        "model_name": "Pedestrian detection",
        "model_arquitecture": SETTINGS.yolo_version,
        "classes": ["person"],
        "input_type": "image",
    }
    return data

@app.get("/greeting")
def greet(name: str | None = None):
    return {
        "status": "OK",
        "message": f"Hello, {name} how are you?"
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")
