import argparse, os, json, time, glob
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Same head as training
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

def load_model(ckpt_path, scripted_path=None):
    device = torch.device("cpu")
    classes = None

    if scripted_path and os.path.isfile(scripted_path):
        model = torch.jit.load(scripted_path, map_location=device)
        # load classes next to scripted file
        cls_json = os.path.splitext(scripted_path)[0] + "_classes.json"
        if os.path.isfile(cls_json):
            with open(cls_json, "r") as f:
                classes = json.load(f)
        return model, classes, device, True

    ckpt = torch.load(ckpt_path, map_location=device)
    meta = ckpt.get("meta", {})
    classes = meta.get("classes", None)
    if classes is None:
        raise RuntimeError("Checkpoint meta missing 'classes'.")

    num_classes = len(classes)
    model = ResNetMultiHead(num_classes)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, classes, device, False

def build_tfms(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

def predict_one(img_path, model, classes, device, recycle_threshold=0.5, scripted=False, tfms=None):
    img = Image.open(img_path).convert("RGB")
    x = tfms(img).unsqueeze(0)
    with torch.no_grad():
        if scripted:
            class_logits, recycle_logit = model(x)  # TorchScript returns same tuple
        else:
            class_logits, recycle_logit = model(x.to(device))
    probs = torch.softmax(class_logits, dim=1)[0]
    cls_idx = int(probs.argmax().item())
    cls_name = classes[cls_idx]
    cls_conf = float(probs[cls_idx].item())

    r_prob = float(torch.sigmoid(recycle_logit).item())
    recyclable = int(r_prob >= recycle_threshold)
    return {
        "image": img_path,
        "class": cls_name,
        "class_conf": cls_conf,
        "recycle_prob": r_prob,
        "recyclable": recyclable
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="best.pt", help="Path to best.pt")
    ap.add_argument("--scripted", default="", help="Optional TorchScript model_scripted.pt")
    ap.add_argument("--path", required=True, help="Image file or folder")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--recycle_threshold", type=float, default=0.5)
    args = ap.parse_args()

    model, classes, device, is_scripted = load_model(args.ckpt, args.scripted if args.scripted else None)
    tfms = build_tfms(args.img_size)
    torch.set_num_threads(2)

    paths = []
    if os.path.isdir(args.path):
        for ext in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
            paths.extend(glob.glob(os.path.join(args.path, ext)))
        paths.sort()
    else:
        paths = [args.path]

    if not paths:
        print("No images found.")
        return

    t0 = time.time()
    for p in paths:
        out = predict_one(p, model, classes, device, args.recycle_threshold, is_scripted, tfms)
        print(f"[{os.path.basename(p)}] class={out['class']} (p={out['class_conf']:.3f}) | "
              f"recycle_prob={out['recycle_prob']:.3f} -> recyclable={out['recyclable']}")
    print(f"Done {len(paths)} images in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
