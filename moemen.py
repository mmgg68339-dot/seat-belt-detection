import os
import argparse
import torch
import clip
from ultralytics import YOLO
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# YOLOv8 لاكتشاف الأشخاص
yolo_model = YOLO("yolov8n.pt")  # نسخة خفيفة وسريعة

# CLIP للتصنيف
clip_model, preprocess = clip.load("ViT-B/32", device=device)

with_prompts = [
    "a close-up photo of a person wearing a seatbelt across their chest",
    "a person sitting in a car wearing a seatbelt",
    "a driver in a car wearing a seat belt",
    "a person with a seat belt fastened over their shoulder"
]
without_prompts = [
    "a close-up photo of a person not wearing a seatbelt, chest visible without any strap",
    "a person sitting in a car without a seatbelt",
    "a driver in a car with no seat belt",
    "a person with no seat belt across their chest"
]
text_prompts = with_prompts + without_prompts
text_tokens = clip.tokenize(text_prompts).to(device)

with torch.no_grad():
    text_features = clip_model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    with_features = text_features[:len(with_prompts)].mean(dim=0)
    without_features = text_features[len(with_prompts):].mean(dim=0)
    class_text_features = torch.stack([with_features, without_features], dim=0)
    # re-normalize after mean-pooling, since averaging unit vectors doesn't stay unit-length
    class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)


# ===== دالة الكشف عن الشخص وقص منطقة الصدر =====
def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    return img


def detect_and_crop_person(image_path):
    img = load_image(image_path)
    results = yolo_model(image_path, imgsz=640, conf=0.25, classes=[0], verbose=False)

    boxes = results[0].boxes
    if len(boxes) == 0:
        return None, None, None, None

    valid_boxes = []
    for box in boxes:
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        if conf < 0.25:
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        if x2 <= x1 or y2 <= y1:
            continue
        width = x2 - x1
        height = y2 - y1
        area = width * height
        if area < 1000:
            continue
        valid_boxes.append((conf, x1, y1, x2, y2))

    if not valid_boxes:
        return None, None, None, None

    # choose the most confident person box
    valid_boxes.sort(key=lambda item: item[0], reverse=True)
    _, x1, y1, x2, y2 = valid_boxes[0]

    width = x2 - x1
    height = y2 - y1

    # add a small padding around the box
    pad_x = width * 0.1
    pad_y = height * 0.1
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img.width, x2 + pad_x)
    y2 = min(img.height, y2 + pad_y)

    # recompute after padding
    width = x2 - x1
    height = y2 - y1
    aspect = height / width if width > 0 else 1.0

    # chest crop: adapt the vertical fraction to the box's aspect ratio,
    # since a fixed 15%-80% slice assumes a tall, full-body box and will
    # cut off the chest/shoulder region on waist-up or wide/short boxes.
    if aspect > 2.2:
        # tall / full-body box (head to feet visible) -> chest is upper portion
        frac_top, frac_bottom = 0.10, 0.55
    elif aspect > 1.0:
        # waist-up / half-body shot -> chest occupies most of the box
        frac_top, frac_bottom = 0.05, 0.90
    else:
        # wide/short box (sideways crop, close-up, unusual framing) -> don't guess
        frac_top, frac_bottom = 0.0, 1.0

    chest_y1 = y1 + height * frac_top
    chest_y2 = y1 + height * frac_bottom
    chest_y1 = min(max(y1, chest_y1), y2 - 10)
    chest_y2 = min(max(chest_y1 + 10, chest_y2), y2)

    chest_crop = img.crop((x1, chest_y1, x2, chest_y2))
    full_crop = img.crop((x1, y1, x2, y2))
    return img, chest_crop, full_crop, (x1, y1, x2, y2)

# ===== دالة التصنيف الكاملة =====
def classify_seatbelt(image_path):
    img, chest_crop, full_crop, box = detect_and_crop_person(image_path)

    if chest_crop is None:
        return "no human in image", None, None, None

    crops = [("chest", chest_crop), ("full", full_crop)]
    probs_list = []
    logit_scale = clip_model.logit_scale.exp()

    for _, crop in crops:
        image_input = preprocess(crop).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # scale raw cosine similarities before softmax (standard CLIP zero-shot
            # recipe) -- without this, similarities are too close together and the
            # softmax output collapses to ~0.5/0.5 regardless of image content
            logits = logit_scale * image_features @ class_text_features.t()  # shape (1, 2)
            probs = logits.softmax(dim=-1).cpu().numpy().reshape(-1)
        probs_list.append(probs)

    # combine both views (chest crop + full body crop) for a more stable prediction
    combined_probs = np.mean(np.vstack(probs_list), axis=0)

    result = {
        "WITH SEATBELT": float(combined_probs[0]),
        "WITHOUT SEATBELT": float(combined_probs[1])
    }

    confidence = float(max(combined_probs))

    if combined_probs[0] > combined_probs[1]:
        prediction = "with seatbelt"
        phrase = "This person is wearing a seatbelt."
    else:
        prediction = "without seatbelt"
        phrase = "This person is not wearing a seatbelt."

    display_crop = chest_crop if chest_crop.size[0] > 0 and chest_crop.size[1] > 0 else full_crop
    return prediction, result, display_crop, confidence, phrase

# ===== اختبار الصورة من خلال إدخال مسار محلي =====

def test_image(image_path):
    if not os.path.exists(image_path):
        print(f"The image path does not exist: {image_path}")
        return

    output = classify_seatbelt(image_path)

    if output[1] is None:
        print(output[0])
    else:
        pred, scores, cropped, confidence, phrase = output
        print("results:", pred)
        print("confidence:", round(confidence, 3))
        print("details:", scores)
        print(phrase)

        # عرض الصورة الأصلية + المنطقة المقصوصة
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(load_image(image_path))
        axes[0].set_title("original image")
        axes[0].axis('off')

        axes[1].imshow(cropped)
        axes[1].set_title(f"checked part\n{phrase}")
        axes[1].axis('off')
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