from ultralytics import YOLO
import supervision as sv
import cv2

def main():
    # args = parse_arguments()
    # frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )
        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()