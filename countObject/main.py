import numpy as np
import supervision as sv

from ultralytics import YOLO
from supervision.assets import download_assets, VideoAssets
download_assets(VideoAssets.VEHICLES)

model = YOLO("yolov8x.pt")

byte_tracker = sv.ByteTrack()
bounding_box_annotator = sv.BoxAnnotator(thickness=4) #BoundingBoxAnnotator
label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
# trace_annotator = sv.TraceAnnotator(thickness=4)

START = sv.Point(0, 1500)
END = sv.Point(3840, 1500)
line_zone = sv.LineZone(start=START, end=END)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_thickness=4,
    text_scale=2)

def callback(frame: np.ndarray, index:int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for confidence, class_id, tracker_id
        in zip(detections.confidence, detections.class_id, detections.tracker_id)
    ]

    annotated_frame = frame.copy()
    # annotated_frame = trace_annotator.annotate(
    #     scene=annotated_frame,
    #     detections=detections)
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame,
        detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels)

    line_zone.trigger(detections)

    return  line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)

sv.process_video(
    source_path = "vehicles.mp4",
    target_path = "output.mp4",
    callback=callback
)

