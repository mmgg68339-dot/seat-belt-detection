import os
import tempfile

import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps

import clip
from ultralytics import YOLO

# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Seatbelt Detector",
    page_icon="🚗",
    layout="centered",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

WITH_PROMPTS = [
    "a close-up photo of a person wearing a seatbelt across their chest",
    "a person sitting in a car wearing a seatbelt",
    "a driver in a car wearing a seat belt",
    "a person with a seat belt fastened over their shoulder",
]
WITHOUT_PROMPTS = [
    "a close-up photo of a person not wearing a seatbelt, chest visible without any strap",
    "a person sitting in a car without a seatbelt",
    "a driver in a car with no seat belt",
    "a person with no seat belt across their chest",
]


# =========================================================
# Cached model loading (runs once per server process)
# =========================================================
@st.cache_resource(show_spinner="Loading models (first run only)...")
def load_models():
    yolo_model = YOLO("yolov8n.pt")
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)

    text_prompts = WITH_PROMPTS + WITHOUT_PROMPTS
    text_tokens = clip.tokenize(text_prompts).to(DEVICE)

    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        with_features = text_features[: len(WITH_PROMPTS)].mean(dim=0)
        without_features = text_features[len(WITH_PROMPTS):].mean(dim=0)
        class_text_features = torch.stack([with_features, without_features], dim=0)
        class_text_features = class_text_features / class_text_features.norm(dim=-1, keepdim=True)

    return yolo_model, clip_model, preprocess, class_text_features


# =========================================================
# Core detection / classification logic (from original script)
# =========================================================
def load_image_from_path(image_path):
    img = Image.open(image_path).convert("RGB")
    img = ImageOps.exif_transpose(img)
    return img


def detect_and_crop_person(image_path, yolo_model):
    img = load_image_from_path(image_path)
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

    valid_boxes.sort(key=lambda item: item[0], reverse=True)
    _, x1, y1, x2, y2 = valid_boxes[0]

    width = x2 - x1
    height = y2 - y1

    pad_x = width * 0.1
    pad_y = height * 0.1
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img.width, x2 + pad_x)
    y2 = min(img.height, y2 + pad_y)

    width = x2 - x1
    height = y2 - y1
    aspect = height / width if width > 0 else 1.0

    if aspect > 2.2:
        frac_top, frac_bottom = 0.10, 0.55
    elif aspect > 1.0:
        frac_top, frac_bottom = 0.05, 0.90
    else:
        frac_top, frac_bottom = 0.0, 1.0

    chest_y1 = y1 + height * frac_top
    chest_y2 = y1 + height * frac_bottom
    chest_y1 = min(max(y1, chest_y1), y2 - 10)
    chest_y2 = min(max(chest_y1 + 10, chest_y2), y2)

    chest_crop = img.crop((x1, chest_y1, x2, chest_y2))
    full_crop = img.crop((x1, y1, x2, y2))
    return img, chest_crop, full_crop, (x1, y1, x2, y2)


def classify_seatbelt(image_path, yolo_model, clip_model, preprocess, class_text_features):
    img, chest_crop, full_crop, box = detect_and_crop_person(image_path, yolo_model)

    if chest_crop is None:
        return "no human in image", None, None, None, None

    crops = [("chest", chest_crop), ("full", full_crop)]
    probs_list = []
    logit_scale = clip_model.logit_scale.exp()

    for _, crop in crops:
        image_input = preprocess(crop).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ class_text_features.t()
            probs = logits.softmax(dim=-1).cpu().numpy().reshape(-1)
        probs_list.append(probs)

    combined_probs = np.mean(np.vstack(probs_list), axis=0)

    result = {
        "WITH SEATBELT": float(combined_probs[0]),
        "WITHOUT SEATBELT": float(combined_probs[1]),
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


# =========================================================
# Streamlit UI
# =========================================================
def main():
    st.title("🚗 Seatbelt Detection")
    st.write(
        "Upload a photo of a driver/passenger and the app will detect the person "
        "and classify whether they are wearing a seatbelt (YOLOv8 + CLIP zero-shot)."
    )

    yolo_model, clip_model, preprocess, class_text_features = load_models()

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"]
    )

    if uploaded_file is not None:
        # Save to a temp file since YOLO/PIL logic works off a path
        suffix = os.path.splitext(uploaded_file.name)[1] or ".jpg"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            with st.spinner("Running detection..."):
                output = classify_seatbelt(
                    tmp_path, yolo_model, clip_model, preprocess, class_text_features
                )

            if output[1] is None:
                st.warning(output[0])
                st.image(Image.open(tmp_path), caption="Uploaded image", use_container_width=True)
            else:
                prediction, scores, cropped, confidence, phrase = output

                col1, col2 = st.columns(2)
                with col1:
                    st.image(load_image_from_path(tmp_path), caption="Original image", use_container_width=True)
                with col2:
                    st.image(cropped, caption="Region analyzed", use_container_width=True)

                if prediction == "with seatbelt":
                    st.success(f"✅ {phrase}")
                else:
                    st.error(f"⚠️ {phrase}")

                st.metric("Confidence", f"{confidence * 100:.1f}%")

                st.subheader("Detailed scores")
                st.bar_chart(scores)
                st.json(scores)
        finally:
            os.remove(tmp_path)
    else:
        st.info("👆 Upload an image to get started.")


if __name__ == "__main__":
    main()