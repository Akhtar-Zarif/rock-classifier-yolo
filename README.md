# Rock Classifier (YOLO11 Classification)

Classify rock samples across **42 rock classes** spanning **igneous**, **metamorphic**, and **sedimentary** specimens using a fine-tuned `yolo11n-cls` model. Includes a browser-based photo upload app (`app.py`), a CLI image classifier (`predict_image.py`), dataset preparation utility (`prepare_rocks_dataset.py`), and an optional Luxonis OAK-1 real-time camera script (`oak1/oak1_rock_classify.py`).

## Repository Structure

```text
rock-classifier-yolo/
├── app.py                     # Gradio web UI for uploading rock photos
├── predict_image.py           # Command-line single-image inference script
├── prepare_rocks_dataset.py   # Splits nested category/rock folders into train/val
├── oak1/
│   └── oak1_rock_classify.py  # Optional real-time camera inference on OAK-1
├── weights/
│   ├── README.md              # Instructions for placing best.pt
│   └── best.pt                # Add your trained weights here
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

The custom dataset used to train the YOLOv11n-cls model (394 images across 42 rock classes spanning igneous, metamorphic, and sedimentary rocks) is hosted on Hugging Face:

- **Hugging Face Dataset:** [`zarifbinakhtar/rocks-classification-42`](https://huggingface.co/datasets/zarifbinakhtar/rocks-classification-42)

```python
from datasets import load_dataset

ds = load_dataset("zarifbinakhtar/rocks-classification-42")
print(ds)

## Quick Start (Photo Upload Web App)

1. **Clone the repository and create a virtual environment:**

   ```bash
   git clone https://github.com/Akhtar-Zarif/rock-classifier-yolo.git
   cd rock-classifier-yolo
   python -m venv .venv
   ```

   Activate the environment:
   - **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
   - **macOS / Linux:** `source .venv/bin/activate`

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Add trained model weights:**

   Copy your trained `best.pt` file into `weights/best.pt`.

4. **Launch the interactive photo classifier:**

   ```bash
   python app.py
   ```

   Open the local URL printed in your terminal (typically `http://127.0.0.1:7860`) and upload a clear, well-lit close-up photo of a rock.

## Command-Line Inference

```bash
python predict_image.py path	o
ock_sample.jpg --top-k 5
```

## Training & Dataset Notes

- **Dataset scale & accuracy:** Trained on 394 images across 42 rock classes at `224x224` resolution (`41.5%` Top-1 validation accuracy and `79.8%` Top-5 validation accuracy). Because sample counts per individual rock class are small, viewing **Top-5 predictions** is recommended, and close-up photos where the rock fills most of the frame work best.
- **Re-training:** Organize source photos under `rocks/<category>/<rock_type>/`, run `python prepare_rocks_dataset.py` to create an 80/20 `train`/`val` split, and train with:

  ```bash
  yolo classify train data=rocks_cls_dataset model=yolo11n-cls.pt epochs=50 imgsz=224
  ```

## Optional: OAK-1 Real-Time Camera

If you have a Luxonis OAK-1 camera and `depthai` (`2.x`) installed in your environment:

```bash
python oak1/oak1_rock_classify.py
```
