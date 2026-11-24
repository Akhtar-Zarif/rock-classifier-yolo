import argparse
from pathlib import Path
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path(__file__).parent / "weights" / "best.pt"


def main():
    parser = argparse.ArgumentParser(description="Classify a rock image using trained YOLO weights.")
    parser.add_argument("image", type=Path, help="Path to the input rock image")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="Path to best.pt weights")
    parser.add_argument("--top-k", type=int, default=5, help="Number of top predictions to display")
    args = parser.parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Weights file not found: {args.weights}")
    if not args.image.exists():
        raise FileNotFoundError(f"Input image not found: {args.image}")

    model = YOLO(str(args.weights))
    results = model.predict(source=str(args.image), verbose=False)
    probs = results[0].probs
    names = results[0].names

    k = min(args.top_k, len(probs.top5))
    top_indices = probs.top5[:k]
    top_confs = probs.top5conf[:k].tolist()

    print(f"Image: {args.image}")
    for rank, (idx, conf) in enumerate(zip(top_indices, top_confs), start=1):
        print(f"  {rank}. {names[idx]}: {conf:.2%}")


if __name__ == "__main__":
    main()
