import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import ViTImageProcessor, AutoModelForImageClassification
from camel_tools.utils.dediac import dediac_ar
from camel_tools.tokenizers.word import simple_word_tokenize

# ---------------- LOAD MODEL AND PROCESSOR ---------------- #
model_path = "output_final_vit"
processor = ViTImageProcessor.from_pretrained(model_path, local_files_only=True)
model = AutoModelForImageClassification.from_pretrained(model_path, local_files_only=True)
model.eval()

# ---------------- ARABIC LABEL MAPPINGS ---------------- #
arabic_letters = [
    "Zain", "Zah", "Meem", "Seen", "Teh", "Lam", "Dad", "Teh_Marbuta", "Reh", "Sad",
    "Dal", "Sheen", "Hah", "Beh", "Tah", "Alef", "Waw", "Qaf", "Al", "Ghain",
    "Heh", "Ain", "Kaf", "Thal", "Feh", "Khah", "Yeh", "Jeem", "Theh", "Noon", "Laa"
]

translation_to_arabic = {
    'Zain': 'ز', 'Zah': 'ظ', 'Meem': 'م', 'Seen': 'س', 'Teh': 'ت', 'Lam': 'ل',
    'Dad': 'ض', 'Teh_Marbuta': 'ة', 'Reh': 'ر', 'Sad': 'ص', 'Dal': 'د', 'Sheen': 'ش',
    'Hah': 'ح', 'Beh': 'ب', 'Tah': 'ط', 'Alef': 'ا', 'Waw': 'و', 'Qaf': 'ق', 'Al': 'ال',
    'Ghain': 'غ', 'Heh': 'ه', 'Ain': 'ع', 'Kaf': 'ك', 'Thal': 'ذ', 'Feh': 'ف',
    'Khah': 'خ', 'Yeh': 'ي', 'Jeem': 'ج', 'Theh': 'ث', 'Noon': 'ن', 'Laa': 'لا'
}

# ---------------- CAMEL TOOLS CLEANING ---------------- #
def clean_arabic_text(input_text):
    normalized = dediac_ar(input_text)
    cleaned = []
    prev = None
    for char in normalized:
        if char != prev:
            cleaned.append(char)
            prev = char
    return ' '.join(simple_word_tokenize(''.join(cleaned)))

# ---------------- OPENCV HAND DETECTION ---------------- #
def detect_hand_opencv(image_np):
    hsv = cv2.cvtColor(image_np, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    pad = 20
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, image_np.shape[1])
    y2 = min(y + h + pad, image_np.shape[0])
    return image_np[y1:y2, x1:x2]

# ---------------- LETTER BUFFER ---------------- #
letter_buffer = []

# ---------------- PREDICTION FUNCTION ---------------- #
def process_and_predict(image_pil):
    global letter_buffer

    # If the image is empty (i.e., no valid photo is provided), handle it
    if image_pil is None:
        return "No hand detected!", clean_arabic_text(''.join(letter_buffer))

    image_np = np.array(image_pil.convert("RGB"))
    hand_crop = detect_hand_opencv(image_np)
    if hand_crop is None:
        return "No hand detected!", clean_arabic_text(''.join(letter_buffer))

    resized = cv2.resize(hand_crop, (224, 224))
    cropped_pil = Image.fromarray(resized)

    inputs = processor(images=cropped_pil, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        pred_id = torch.argmax(logits, dim=1).item()

    label = arabic_letters[pred_id]
    arabic_char = translation_to_arabic.get(label, "?")

    letter_buffer.append(arabic_char)
    sentence = clean_arabic_text(''.join(letter_buffer))

    return f"🔤 Letter: {arabic_char}", sentence

# ---------------- FUNCTION TO ADD SPACE ---------------- #
def add_space():
    global letter_buffer
    letter_buffer.append(" ")
    sentence = clean_arabic_text(''.join(letter_buffer))
    return sentence

# ---------------- CLEAR FUNCTION ---------------- #
def clear_sentence():
    global letter_buffer
    letter_buffer = []
    return "Buffer cleared", ""  # Clear the output fields

# ---------------- GRADIO UI ---------------- #
with gr.Blocks() as demo:
    gr.Markdown("## 🤟 Arabic Sign Language → Arabic Sentence ✨")
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload or Capture Gesture")
        output_letter = gr.Textbox(label="Predicted Letter")
        output_sentence = gr.Textbox(label="Reconstructed Sentence")
    with gr.Row():
        btn_clear = gr.Button("🗑️ Clear Sentence")
        btn_space = gr.Button("Space")

    image_input.change(fn=process_and_predict, inputs=image_input, outputs=[output_letter, output_sentence])
    btn_clear.click(fn=clear_sentence, outputs=[output_letter, output_sentence])
    btn_space.click(fn=add_space, outputs=[output_sentence])

demo.launch()
