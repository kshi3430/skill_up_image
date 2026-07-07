import argparse
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from train import device_name, make_model


def main():
    parser = argparse.ArgumentParser(description="실시간 손톱 이미지 분류")
    parser.add_argument("--model", default="artifacts/best.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--interval", type=float, default=0.25,
                        help="추론 간격(초)")
    args = parser.parse_args()

    device = device_name()
    checkpoint = torch.load(args.model, map_location=device, weights_only=True)
    classes = checkpoint["classes"]
    model = make_model(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise SystemExit(
            "카메라를 열 수 없습니다. macOS 설정 > 개인정보 보호 및 보안 > "
            "카메라에서 Terminal(또는 사용 중인 IDE)을 허용하세요."
        )

    label, confidence = "손톱을 사각형 안에 놓으세요", 0.0
    last_inference = 0.0
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            height, width = frame.shape[:2]
            side = int(min(height, width) * 0.65)
            x1, y1 = (width - side) // 2, (height - side) // 2
            x2, y2 = x1 + side, y1 + side

            now = time.monotonic()
            if now - last_inference >= args.interval:
                roi = frame[y1:y2, x1:x2]
                rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                image = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(device)
                with torch.inference_mode():
                    probabilities = model(image).softmax(dim=1)[0]
                index = int(probabilities.argmax())
                label = classes[index]
                confidence = float(probabilities[index])
                last_inference = now

            color = (50, 220, 50) if confidence >= 0.7 else (0, 180, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f"{label}  {confidence * 100:.1f}%"
            cv2.rectangle(frame, (12, 12), (min(width - 12, 620), 58), (0, 0, 0), -1)
            cv2.putText(frame, text, (22, 44), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2, cv2.LINE_AA)
            cv2.putText(frame, "Q: quit", (12, height - 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.imshow("Nail disease classifier (research only)", frame)

            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
