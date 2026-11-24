from pathlib import Path
import gradio as gr
from ultralytics import YOLO

WEIGHTS_PATH = Path(__file__).parent / "weights" / "best.pt"


def load_model():
    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Trained weights not found at {WEIGHTS_PATH}. "
            "Place your trained YOLO classification weights at weights/best.pt."
        )
    return YOLO(str(WEIGHTS_PATH))


model = load_model()


def classify_rock(image):
    if image is None:
        return {}

    results = model.predict(source=image, verbose=False)
    probs = results[0].probs
    names = results[0].names

    top5_indices = probs.top5
    top5_confs = probs.top5conf.tolist()
    return {names[idx]: float(conf) for idx, conf in zip(top5_indices, top5_confs)}


demo = gr.Interface(
    fn=classify_rock,
    inputs=gr.Image(type="pil", label="Upload a Rock Photo"),
    outputs=gr.Label(num_top_classes=5, label="Top-5 Rock Predictions"),
    title="YOLO11 Rock Classifier (42 Classes)",
    description=(
        "Upload a close-up photo of a rock sample to predict its rock class "
        "(across igneous, metamorphic, and sedimentary specimens). "
        "Best results come from clear, well-lit close-ups where the rock fills most of the frame."
    ),
)

if __name__ == "__main__":
    demo.launch()
