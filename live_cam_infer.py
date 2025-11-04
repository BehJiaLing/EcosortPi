import os, time, argparse, sys
import torch
import torch.nn as nn
from torchvision import models
import numpy as np
import cv2

# ---------- Optional LEDs ----------
HAS_LEDS = False
try:
    import board, neopixel
    LED_PIN = board.D18
    LED_COUNT = 8
    BRIGHTNESS = 0.2
    _led = neopixel.NeoPixel(LED_PIN, LED_COUNT, brightness=BRIGHTNESS, auto_write=False)
    def set_leds(rgb): _led.fill(rgb); _led.show()
    def leds_off(): set_leds((0,0,0))
    HAS_LEDS = True
except Exception:
    def set_leds(rgb): pass
    def leds_off(): pass

COLOR_IDLE       = (0, 0, 64)     # blue (empty / waiting)
COLOR_SEEING     = (64, 32, 0)    # amber (seeing content)
COLOR_PREDICT    = (128, 128, 128)# white-ish (predicting)
COLOR_RECYCLE    = (0, 64, 0)     # green
COLOR_NONRECYCLE = (64, 0, 0)     # red
COLOR_ERROR      = (64, 0, 64)    # magenta

# ---------- Camera backends ----------
try:
    from picamera2 import Picamera2
    HAS_PICAM2 = True
except Exception:
    HAS_PICAM2 = False

def eprint(*a, **k): print(*a, file=sys.stderr, **k)

# ---------- Model ----------
class ResNetMultiHead(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_feats = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_feats, num_classes))
        self.recycle_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_feats, 1))
    def forward(self, x):
        f = self.backbone(x)
        return self.classifier(f), self.recycle_head(f).squeeze(-1)

def load_from_ckpt(ckpt_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt.get("meta", {}).get("classes", None)
    if not classes: raise RuntimeError("Checkpoint meta missing 'classes'.")
    m = ResNetMultiHead(len(classes))
    m.load_state_dict(ckpt["model"])
    m.eval()
    torch.set_num_threads(2)
    return m, classes

# ---------- Frame utils ----------
def preprocess_bgr(bgr, img_size=224):
    h, w = bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side)//2; x0 = (w - side)//2
    crop = bgr[y0:y0+side, x0:x0+side]
    rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb  = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb).permute(2,0,1).float()/255.0
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    x = (x-mean)/std
    return x.unsqueeze(0)

# Heuristic “empty” detector: low contrast + few edges
EMPTY_STD_THR  = 12.0     # stddev in grayscale
EMPTY_EDGE_THR = 400      # number of edge pixels
def is_empty(frame):
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.size == 0:
        return True
    small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    std   = float(gray.std())
    edges = cv2.Canny(gray, 60, 120)
    edge_count = int((edges > 0).sum())
    return (std < EMPTY_STD_THR) or (edge_count < EMPTY_EDGE_THR)

# ---------- Camera ----------
def open_camera(prefer_picam=True):
    if prefer_picam and HAS_PICAM2:
        cam = Picamera2()
        cam.configure(cam.create_preview_configuration(main={"format":"RGB888","size":(1280,720)}))
        cam.start()
        def grab():
            try:
                f = cam.capture_array()
                return f if isinstance(f, np.ndarray) and f.size>0 else None
            except Exception: return None
        def release():
            try: cam.stop()
            except Exception: pass
        print("[INFO] Picamera2 started (BGR888 1280x720).")
        return grab, release, "picamera2"
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    if not cap.isOpened(): raise RuntimeError("Cannot open camera.")
    def grab():
        ok, f = cap.read()
        return f if ok and isinstance(f, np.ndarray) and f.size>0 else None
    def release():
        try: cap.release()
        except Exception: pass
    print("[INFO] OpenCV camera started (BGR 1280x720).")
    return grab, release, "opencv"

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--recycle_threshold", type=float, default=0.5)
    ap.add_argument("--fps_limit", type=float, default=8.0)
    ap.add_argument("--no_window", action="store_true")
    args = ap.parse_args()

    # LEDs: idle/empty by default
    set_leds(COLOR_PREDICT)

    try:
        model, classes = load_from_ckpt(args.ckpt)
    except Exception as ex:
        set_leds(COLOR_PREDICT); eprint(f"[ERROR] load ckpt: {ex}"); sys.exit(1)

    try:
        grab, release, source = open_camera(prefer_picam=True)
    except Exception as ex:
        set_leds(COLOR_PREDICT); eprint(f"[ERROR] camera: {ex}"); sys.exit(2)

    min_dt = 1.0 / max(args.fps_limit, 1e-3)
    last = 0.0

    try:
        while True:
            frame = grab()
            if frame is None:
                # keep idle LEDs
                time.sleep(0.01)
                if not args.no_window and cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # Empty/blank detection
            if is_empty(frame):
                set_leds(COLOR_PREDICT)
                if not args.no_window:
                    cv2.imshow("EcoSort Live", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # We see content
            set_leds(COLOR_PREDICT)

            now = time.time()
            if now - last < min_dt:
                if not args.no_window:
                    cv2.imshow("EcoSort Live", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue
            last = now

            # Predict
            set_leds(COLOR_PREDICT)
            try:
                x = preprocess_bgr(frame, args.img_size)
            except Exception as ex:
                eprint(f"[WARN] preprocess: {ex}")
                set_leds(COLOR_PREDICT)
                continue

            with torch.no_grad():
                try:
                    logits, rlogit = model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                    cls_idx = int(torch.argmax(probs).item())
                    cls_p   = float(probs[cls_idx].item())
                    rprob   = float(torch.sigmoid(rlogit).item())
                    recyclable = int(rprob >= args.recycle_threshold)
                except Exception as ex:
                    eprint(f"[WARN] inference: {ex}")
                    set_leds(COLOR_PREDICT)
                    continue

            cls_name = classes[cls_idx]
            # LEDs by decision
            set_leds(COLOR_PREDICT if recyclable else COLOR_PREDICT)

            cv2.putText(frame, f"class: {cls_name} (p={cls_p:.2f})", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"recyclable: {recyclable} (prob={rprob:.2f})", (20,75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)

            if not args.no_window:
                cv2.imshow("EcoSort Live", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # loop end
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        try: release()
        except Exception: pass
        if not args.no_window:
            cv2.destroyAllWindows()
        leds_off()  # always turn off

if __name__ == "__main__":
    main()
