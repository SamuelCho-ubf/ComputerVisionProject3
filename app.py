import cv2
import inference
import supervision as sv
import os
from dotenv import load_dotenv

load_dotenv()
VIDEO_ROUTE = os.getenv("VIDEO_ROUTE")

cap = cv2.VideoCapture(VIDEO_ROUTE)

frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (frameWidth, frameHeight)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

out = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

annotator = sv.BoxAnnotator()

def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    out.write(image)
    cv2.imshow(
        "Prediction", 
        annotator.annotate(
            scene=image, 
            detections=detections,
            labels=labels
        )
    ),
    cv2.waitKey(1)

inference.Stream(
    source=VIDEO_ROUTE, # or rtsp stream or camera id
    model="bikes-ped-scooters/4", # from Universe
    output_channel_order="BGR",
    use_main_thread=True, # for opencv display
    on_prediction=on_prediction, 
)
