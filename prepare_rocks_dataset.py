import random
import shutil
from pathlib import Path

# --- CONFIGURATION ---
SOURCE_DIR = Path("rocks")              # Nested rocks/<category>/<rock_type>/images
TARGET_DIR = Path("rocks_cls_dataset")  # Output train/val split for YOLO classification
SPLIT_RATIO = 0.8                       # 80% train, 20% val
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# ---------------------


def main():
    random.seed(42)

    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Source folder '{SOURCE_DIR}' not found. Expected nested folders: rocks/<category>/<rock_name>."
        )

    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    total_classes = 0
    total_images = 0

    for category_dir in sorted(SOURCE_DIR.iterdir()):
        if not category_dir.is_dir():
            continue

        for rock_dir in sorted(category_dir.iterdir()):
            if not rock_dir.is_dir():
                continue

            images = [f for f in rock_dir.iterdir() if f.suffix.lower() in VALID_EXTS]
            if not images:
                continue

            random.shuffle(images)
            split_idx = max(1, int(len(images) * SPLIT_RATIO)) if len(images) > 1 else 1
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]

            rock_name = rock_dir.name
            train_class_dir = TARGET_DIR / "train" / rock_name
            val_class_dir = TARGET_DIR / "val" / rock_name
            train_class_dir.mkdir(parents=True, exist_ok=True)
            val_class_dir.mkdir(parents=True, exist_ok=True)

            for img in train_imgs:
                shutil.copy2(img, train_class_dir / img.name)
            for img in val_imgs:
                shutil.copy2(img, val_class_dir / img.name)

            total_classes += 1
            total_images += len(images)
            print(f"Processed '{rock_name}': {len(train_imgs)} train, {len(val_imgs)} val")

    print(f"\nDone! Prepared {total_images} images across {total_classes} rock classes in '{TARGET_DIR}'.")


if __name__ == "__main__":
    main()
