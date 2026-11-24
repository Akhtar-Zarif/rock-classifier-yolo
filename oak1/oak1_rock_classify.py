import time
import cv2
import depthai as dai
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = r"weights\best.pt"
CONF_THRESHOLD = 0.30  # Minimum confidence to log prediction in terminal
# ---------------------


def main():
    print(f"Loading YOLO Classification model from: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # Build DepthAI Pipeline (v2.x Syntax)
    pipeline = dai.Pipeline()

    camRgb = pipeline.createColorCamera()
    camRgb.setPreviewSize(640, 640)
    camRgb.setInterleaved(False)
    camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    print("Connecting to OAK-1 camera...")
    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        print("OAK-1 Rock Classifier Active! Press 'q' in the window to exit.")

        prev_time = 0.0

        while True:
            inRgb = qRgb.get()
            frame = inRgb.getCvFrame()

            results = model.predict(source=frame, verbose=False)
            probs = results[0].probs
            top1_idx = probs.top1
            top1_conf = float(probs.top1conf)
            rock_name = results[0].names[top1_idx]

            if top1_conf >= CONF_THRESHOLD:
                print(f"Top Prediction: {rock_name} | Confidence: {top1_conf:.2f}")

            annotated_frame = results[0].plot()

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time > 0 else 0.0
            prev_time = curr_time

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow("OAK-1 Real-time Rock Classification", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()
    print("Camera stream closed.")


if __name__ == "__main__":
    main()
