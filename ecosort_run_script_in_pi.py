import os, time, datetime, base64, qrcode
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from picamera2 import Picamera2
import firebase_admin
from firebase_admin import credentials, firestore
import RPi.GPIO as GPIO
from luma.core.interface.serial import i2c
from luma.oled.device import sh1106
import board
import neopixel

# ---------------------------------------------------
# Firebase Setup
# ---------------------------------------------------
cred = credentials.Certificate("firebase-key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# ---------------------------------------------------
# PyTorch model helpers 
# ---------------------------------------------------
import torch.nn as nn
from torchvision import models

class ResNetMultiHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_feats, num_classes))
        self.recycle_head = nn.Sequential(nn.Dropout(p=0.2), nn.Linear(in_feats, 1))
    def forward(self, x):
        feats = self.backbone(x)
        class_logits = self.classifier(feats)
        recycle_logit = self.recycle_head(feats).squeeze(-1)
        return class_logits, recycle_logit

def load_from_ckpt(ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt.get("meta", {}).get("classes", None)
    if not classes:
        raise RuntimeError("Checkpoint meta missing 'classes'")
    model = ResNetMultiHead(len(classes))
    model.load_state_dict(ckpt["model"])
    model.eval()
    torch.set_num_threads(2)
    return model, classes

def preprocess_bgr(bgr, img_size=224):
    """Crop-center and normalize BGR image."""
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side)//2
    x0 = (w - side)//2
    crop = bgr[y0:y0+side, x0:x0+side]
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb).permute(2,0,1).float()/255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    x = (x - mean) / std
    return x.unsqueeze(0)
    
# Load model
MODEL_PATH = "best.pt"
INFER_IMG_SIZE = 224
RECYCLE_THRESHOLD = 0.2
model, classes = load_from_ckpt(MODEL_PATH)

# ---------------------------------------------------
# Load model 
# ---------------------------------------------------
# MODEL_PATH = "best.pt"
# INFER_IMG_SIZE = 224
# RECYCLE_THRESHOLD = 0.2 
# model, classes = load_from_ckpt(MODEL_PATH)

# ---------------------------------------------------
# GPIO + Servo Setup
# ---------------------------------------------------
servo_sort_pin = 17
servo_lid_pin = 27
ir_sensor_pin = 22

GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_sort_pin, GPIO.OUT)
GPIO.setup(servo_lid_pin, GPIO.OUT)
GPIO.setup(ir_sensor_pin, GPIO.IN)

pwm_sort = GPIO.PWM(servo_sort_pin, 50)
pwm_lid = GPIO.PWM(servo_lid_pin, 50)
pwm_sort.start(0)
pwm_lid.start(0)

def set_angle(pwm, angle):
    duty = 2 + (angle / 18)
    pwm.ChangeDutyCycle(duty)
    time.sleep(0.5)
    pwm.ChangeDutyCycle(0)

# Neutral positions
set_angle(pwm_sort, 102)
set_angle(pwm_lid, 190)

# -------------------------------
# WS2812B / NeoPixel Setup (Single LED Stick)
# -------------------------------
led_pin = board.D18  # GPIO18 (Pin 12)
led_count = 8        # number of LEDs on the stick
brightness = 0.2     # adjustable brightness (0.0 - 1.0)

led_strip = neopixel.NeoPixel(
    led_pin,
    led_count,
    brightness=brightness,
    auto_write=False
)

def set_leds(color):
    """Set LED stick to a given RGB color."""
    led_strip.fill(color)
    led_strip.show()

def leds_off():
    """Turn off all LEDs."""
    set_leds((0, 0, 0))

# ---------------------------------------------------
# OLED Setup
# ---------------------------------------------------
serial = i2c(port=1, address=0x3C)
oled = sh1106(serial, width=128, height=64)
font = ImageFont.load_default()

def show_message(lines, duration=2):
    """Helper: show multiple lines of text on OLED."""
    msg_screen = Image.new("1", (oled.width, oled.height), "black")
    msg_draw = ImageDraw.Draw(msg_screen)
    for i, line in enumerate(lines):
        msg_draw.text((5, 15 + i * 12), line, font=font, fill=255)
    oled.display(msg_screen)
    time.sleep(duration)

# ---------------------------------------------------
# Empty Frame Detection
# ---------------------------------------------------
EMPTY_STD_THR  = 12.0      # adjust if needed (10–15 typical)
EMPTY_EDGE_THR = 400       # fewer edges => empty

def is_empty_frame(frame_bgr):
    """
    Detect empty frame using brightness variance + edge density.
    Works for BGR images (OpenCV / Picamera2).
    """
    if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3 or frame_bgr.size == 0:
        return True
    small = cv2.resize(frame_bgr, (320, 180), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    std = float(gray.std())
    edges = cv2.Canny(gray, 60, 120)
    edge_count = int((edges > 0).sum())
    print(f"[INFO] Empty check — std: {std:.1f}, edges: {edge_count}")
    return (std < EMPTY_STD_THR) or (edge_count < EMPTY_EDGE_THR)

# ---------------------------------------------------
# Camera
# ---------------------------------------------------
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format":"BGR888","size":(1280,720)}))

# ---------------------------------------------------
# Continuous Main Loop
# ---------------------------------------------------
while True:
    # Step 1: Wait for Hand Detection
    show_message(["Wave your hand", "to open the lid"], duration=2)
    print("[INFO] Waiting for hand detection...")

    while GPIO.input(ir_sensor_pin) == GPIO.HIGH:
        time.sleep(0.1)

    print("[INFO] Hand detected!")

    # Step 2: Open Lid
    open_time = 5
    set_angle(pwm_lid, 120)
    for remaining in range(open_time, 0, -1):
        msg_screen = Image.new("1", (oled.width, oled.height), "black")
        msg_draw = ImageDraw.Draw(msg_screen)
        msg_draw.text((10, 15), "Lid opened!", font=font, fill=255)
        msg_draw.text((10, 30), f"Closing in {remaining}s", font=font, fill=255)
        oled.display(msg_screen)
        time.sleep(1)
    set_angle(pwm_lid, 155)
    set_angle(pwm_lid, 190)
    print("[INFO] Lid closed")
    set_leds((255, 255, 255))

    # Step 3: Capture Image
    show_message(["Scanning..."], duration=2)
    picam2.start()
    time.sleep(1.0) 
    frame_bgr = picam2.capture_array()
    picam2.capture_file("capture.jpg")
    picam2.stop()

    img = Image.open("capture.jpg").convert("RGB")
    img_np = np.array(img)

    # Step 4: Empty Check 
    if is_empty_frame(frame_bgr):
        print("[WARN] Empty frame detected — skipping.")
        show_message(["Scan empty!", "Please try again."], duration=3)
        leds_off()
        show_message(["Returning to standby..."], duration=2)
        continue

    # Step 5: Inference 
    print("[INFO] Running inference...")
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    x = preprocess_bgr(bgr, INFER_IMG_SIZE)
    with torch.no_grad():
        logits, rlogit = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        cls_idx = int(torch.argmax(probs).item())
        waste_class = classes[cls_idx]
        class_conf = float(probs[cls_idx].item())
        recycle_prob = float(torch.sigmoid(rlogit).item())
        is_recyclable = int(recycle_prob >= RECYCLE_THRESHOLD)

    # Optional: show top-3 predictions
    topk = min(3, len(classes))
    top_probs, top_idx = torch.topk(probs, topk)
    top3 = [(classes[int(i)], float(p)) for p, i in zip(top_probs.tolist(), top_idx.tolist())]

    print(f"[RESULT] Waste Type: {waste_class} | "
        f"Recyclable: {bool(is_recyclable)} | "
        f"Confidence: {class_conf:.2f} | Recycle_Prob: {recycle_prob:.2f}")
    print(f"[TOP-3] {top3}")

    # Step 6: Sorting Servo Action
    if is_recyclable:
        print("[ACTION] ♻️ Recyclable → Servo_sort = 0°")
        set_leds((0, 255, 0))   # Green light
        set_angle(pwm_sort, 10)
    else:
        print("[ACTION] 🗑️ Non-Recyclable → Servo_sort = 180°")
        set_leds((255, 0, 0))   # Red light
        set_angle(pwm_sort, 180)

    time.sleep(2)
    set_angle(pwm_sort, 102)
    print("[INFO] Servo returned to neutral position")
    leds_off()

    # Step 7: Save to Firebase
    with open("capture.jpg", "rb") as f:
        img_bytes = f.read()
    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    # label
    if is_recyclable:
        prediction_label = "Recyclable"
        points = 20
    else:
        prediction_label = "Non-Recyclable"
        points = 0

    result_data = {
        "waste_class": waste_class,
        "prediction": prediction_label,    
        "confidence": float(class_conf),
        "recycle_prob": float(recycle_prob),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_base64": img_base64,
        "points": points,
        "collected": False
    }
    doc_ref = db.collection("waste_data").add(result_data)
    waste_doc_id = doc_ref[1].id
    print(f"[FIREBASE] Saved Waste ID: {waste_doc_id}")

    # Step 8: Generate QR Code
    # qr = qrcode.QRCode(
    #     version=2,
    #     error_correction=qrcode.constants.ERROR_CORRECT_L,
    #     box_size=1,
    #     border=0
    # )
    # qr.add_data(waste_doc_id)
    # qr.make(fit=True)
    # matrix = qr.get_matrix()
    # qr_size = len(matrix)

    # Step 8 & 9: Display Result and Generate QR (for recyclable only)
    if is_recyclable:
        # Show messages
        show_message(["Congratulations!", "Waste recyclable!"], duration=3)
        show_message(["Scan QR code", "to collect points"], duration=3)

        # Generate QR code for recyclable waste only
        qr = qrcode.QRCode(
            version=2,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=0
        )
        qr.add_data(waste_doc_id)
        qr.make(fit=True)
        matrix = qr.get_matrix()
        qr_size = len(matrix)

        # Display QR code on OLED
        scale = min(oled.width // qr_size, oled.height // qr_size)
        shiftx = (oled.width - qr_size * scale) // 2
        shifty = (oled.height - qr_size * scale) // 2 - 8

        for remaining in range(20, 0, -1):
            image = Image.new("1", (oled.width, oled.height), "black")
            draw = ImageDraw.Draw(image)
            for y in range(qr_size):
                for x in range(qr_size):
                    if matrix[y][x]:
                        x1 = shiftx + x * scale
                        y1 = shifty + y * scale
                        x2 = x1 + scale - 1
                        y2 = y1 + scale - 1
                        draw.rectangle((x1, y1, x2, y2), fill=255)
            countdown_text = f"Time left: {remaining}s"
            draw.text((5, 54), countdown_text, font=font, fill=255)
            oled.display(image)
            time.sleep(1)
    else:
        show_message(["Sorry!", "Waste non-recyclable!"], duration=3)
        show_message(["No point to collect...", "Please try again!"], duration=3)

    # Step 10: Final Goodbye
    show_message(["Bye bye,", "See you next time!"], duration=3)
    oled.display(Image.new("1", (oled.width, oled.height), "black"))
    print("✅ Cycle complete. Waiting for next user...")

# ---------------------------------------------------
# Cleanup
# ---------------------------------------------------
pwm_sort.stop()
pwm_lid.stop()
GPIO.cleanup()
print("[INFO] GPIO cleaned up.")
