import os
import argparse
from ultralytics import YOLO
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ===== CONFIG =====
PERSON_MODEL_PATH = "yolov8n.pt"          # generic person detector (COCO pretrained)
SEATBELT_MODEL_PATH = "best.pt"           # <-- put the path to YOUR trained seatbelt-detector weights here
                                           #     (runs/detect/seatbelt_detector/weights/best.pt after training)

# names in your trained model's class list that mean "wearing a seatbelt"
# check with: print(seatbelt_model.names) after loading, and adjust these to match exactly
WITH_SEATBELT_LABELS = {"seat belt", "seatbelt", "with seatbelt", "seat_belt"}
WITHOUT_SEATBELT_LABELS = {"no seat belt", "no_seatbelt", "no seatbelt", "without seatbelt"}

person_model = YOLO(PERSON_MODEL_PATH)
seatbelt_model = YOLO(SEATBELT_MODEL_PATH)


def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    return img


def detect_person_crop(image_path):
    """Find the most confident person box and return a padded crop of it.
    Returns (full_img, crop, box) or (full_img, None, None) if no person found."""
    img = load_image(image_path)
    results = person_model(image_path, imgsz=640, conf=0.25, classes=[0], verbose=False)

    boxes = results[0].boxes
    if len(boxes) == 0:
        return img, None, None

    valid_boxes = []
    for box in boxes:
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        if conf < 0.25:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        if x2 <= x1 or y2 <= y1:
            continue
        if (x2 - x1) * (y2 - y1) < 1000:
            continue
        valid_boxes.append((conf, x1, y1, x2, y2))

    if not valid_boxes:
        return img, None, None

    valid_boxes.sort(key=lambda item: item[0], reverse=True)
    _, x1, y1, x2, y2 = valid_boxes[0]

    width, height = x2 - x1, y2 - y1
    pad_x, pad_y = width * 0.15, height * 0.15
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img.width, x2 + pad_x)
    y2 = min(img.height, y2 + pad_y)

    crop = img.crop((x1, y1, x2, y2))
    return img, crop, (x1, y1, x2, y2)


def run_seatbelt_detector(pil_image):
    """Run the trained seatbelt-detector on a PIL image.
    Returns list of (label, confidence, box_xyxy_in_image_coords)."""
    results = seatbelt_model(pil_image, imgsz=640, conf=0.25, verbose=False)
    detections = []
    names = results[0].names
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy()
        detections.append((label, conf, xyxy))
    return detections


def classify_seatbelt(image_path):
    img, person_crop, person_box = detect_person_crop(image_path)

    # Prefer running the seatbelt detector on the cropped person (higher effective
    # resolution on the belt region); fall back to the full image if no person
    # was found, since the belt detector may still catch it directly.
    target_image = person_crop if person_crop is not None else img
    detections = run_seatbelt_detector(target_image)

    if not detections:
        if person_crop is None:
            return "no human in image", None, None, None, None
        return "no seatbelt region detected", None, person_crop, None, None

    # pick the single most confident detection
    detections.sort(key=lambda d: d[1], reverse=True)
    label, confidence, box = detections[0]

    label_lower = label.lower().strip()
    if label_lower in WITH_SEATBELT_LABELS:
        prediction = "with seatbelt"
        phrase = "This person is wearing a seatbelt."
    elif label_lower in WITHOUT_SEATBELT_LABELS:
        prediction = "without seatbelt"
        phrase = "This person is not wearing a seatbelt."
    else:
        # label didn't match either known set -- surface it raw so you can fix
        # WITH_SEATBELT_LABELS / WITHOUT_SEATBELT_LABELS above to match your model
        prediction = f"unrecognized label: {label}"
        phrase = f"Model returned an unmapped class '{label}' -- update the label sets in the script."

    result = {"label": label, "confidence": confidence, "all_detections": detections}
    return prediction, result, target_image, confidence, phrase


def test_image(image_path):
    if not os.path.exists(image_path):
        print(f"The image path does not exist: {image_path}")
        return

    prediction, result, display_img, confidence, phrase = classify_seatbelt(image_path)

    if result is None:
        print(prediction)
        return

    print("result:", prediction)
    print("confidence:", round(confidence, 3))
    print("details:", result)
    print(phrase)

    fig, ax = plt.subplots(1, figsize=(7, 7))
    ax.imshow(display_img)
    ax.set_title(f"{phrase}\n(conf: {confidence:.2f})")
    ax.axis("off")

    # draw box for the winning detection, in the displayed image's own coordinates
    for label, conf, box in result["all_detections"]:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="lime" if label == result["label"] else "gray",
            facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(x1, max(0, y1 - 5), f"{label} {conf:.2f}", color="lime", fontsize=9)

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Test the seatbelt detection model on an image")
    parser.add_argument("image_path", nargs="?", help="Path to the image file to test")
    args = parser.parse_args()

    image_path = args.image_path
    if not image_path:
        image_path = input("Enter the image path: ").strip().strip('"')

    test_image(image_path)


if __name__ == "__main__":
    main()