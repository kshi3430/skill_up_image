import argparse
import json
import torch
from PIL import Image
from torchvision import transforms
from train import device_name, make_model

p = argparse.ArgumentParser()
p.add_argument("image"); p.add_argument("--model", default="artifacts/best.pt")
args = p.parse_args(); device = device_name()
ckpt = torch.load(args.model, map_location=device, weights_only=True)
model = make_model(len(ckpt["classes"]), pretrained=False); model.load_state_dict(ckpt["state_dict"]); model.to(device).eval()
tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
with torch.inference_mode():
    probs = model(tf(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)).softmax(1)[0]
ranked = sorted(zip(ckpt["classes"], probs.cpu().tolist()), key=lambda x: x[1], reverse=True)
print(json.dumps([{"class": c, "probability": round(p, 4)} for c, p in ranked], ensure_ascii=False, indent=2))
