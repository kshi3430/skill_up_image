from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from train import device_name, make_model


MODEL_PATH = Path("artifacts/best.pt")
CLASS_NAMES_KO = {
    "Acral_Lentiginous_Melanoma": "말단 흑색점 흑색종",
    "Healthy_Nail": "건강한 손톱",
    "Onychogryphosis": "조갑구만증",
    "blue_finger": "청색 손가락",
    "clubbing": "곤봉지",
    "pitting": "손톱 함몰",
}

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])


@st.cache_resource
def load_model():
    device = device_name()
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = make_model(len(checkpoint["classes"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint["classes"], device


st.set_page_config(page_title="손톱 이미지 분류", page_icon="🔎", layout="centered")
st.title("손톱 이미지 분류")
st.caption("손톱이 선명하고 화면 중앙에 크게 나온 사진을 사용하세요.")

if not MODEL_PATH.exists():
    st.error(f"학습 모델을 찾을 수 없습니다: {MODEL_PATH}")
    st.stop()

input_method = st.radio(
    "사진 입력 방법",
    ["사진 업로드", "카메라 촬영"],
    horizontal=True,
)

if input_method == "사진 업로드":
    uploaded = st.file_uploader(
        "손톱 사진 선택",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
    )
else:
    st.info("한 손톱이 화면 중앙을 크게 채우도록 가까이에서 촬영하세요.")
    uploaded = st.camera_input("손톱 사진 촬영")

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except Exception:
        st.error("이미지를 읽을 수 없습니다. 다른 사진을 선택해 주세요.")
        st.stop()

    caption = "촬영한 사진" if input_method == "카메라 촬영" else "업로드한 사진"
    st.image(image, caption=caption, use_container_width=True)
    model, classes, device = load_model()
    tensor = PREPROCESS(image).unsqueeze(0).to(device)
    with st.spinner("이미지를 분석하고 있습니다..."):
        with torch.inference_mode():
            probabilities = model(tensor).softmax(dim=1)[0].cpu()

    ranked = sorted(zip(classes, probabilities.tolist()), key=lambda x: x[1], reverse=True)
    top_class, top_score = ranked[0]
    top_name = CLASS_NAMES_KO.get(top_class, top_class)

    st.subheader("분류 결과")
    st.metric(top_name, f"{top_score * 100:.1f}%")
    if top_score < 0.6:
        st.warning("모델 확신도가 낮습니다. 손톱을 가까이에서 선명하게 촬영한 사진으로 다시 시도하세요.")

    st.subheader("전체 예측 점수")
    for class_name, score in ranked:
        name = CLASS_NAMES_KO.get(class_name, class_name)
        st.write(f"{name} — {score * 100:.1f}%")
        st.progress(float(score))

st.divider()
st.caption(
    "연구·교육용 모델이며 의료 진단을 대신하지 않습니다. 색 변화, 통증, 출혈 또는 "
    "지속적인 변형이 있다면 피부과 전문의의 진료를 받으세요."
)
