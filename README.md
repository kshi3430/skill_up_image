# 손톱 질환 이미지 분류 베이스라인
## https://kshi3430.streamlit.app

6개 클래스의 사진을 EfficientNet-B0 전이학습으로 분류합니다. 이 모델은 의료 진단기가 아니라 연구용 보조 모델입니다.

## 중요한 데이터 이슈

원본 데이터에는 같은 원본의 Roboflow 증강본이 train/validation 양쪽에 들어간 그룹이 있습니다. 이 경우 검증 점수가 과대평가되므로 `prepare_data.py`가 원본 파일명 단위로 묶어 다시 분할합니다.

## 실행

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python prepare_data.py
python train.py --epochs 15
python predict.py path/to/nail.jpg
```

## 카메라로 실시간 확인

```bash
pip install opencv-python
python camera_predict.py
```

손톱을 화면 중앙 사각형에 놓고 `Q`를 누르면 종료됩니다. 카메라가 열리지 않으면 macOS의 **설정 → 개인정보 보호 및 보안 → 카메라**에서 Terminal 또는 IDE의 권한을 허용하세요. 화면의 확률은 진단 확률이 아니라 현재 데이터셋의 6개 클래스 중 모델이 선택한 상대 점수입니다.

## 사진 업로드 화면

```bash
pip install -r requirements.txt
streamlit run app.py
```

브라우저가 열리면 JPG, PNG 또는 WebP 손톱 사진을 올리거나 기기의 카메라로 직접 촬영해 분류 결과를 확인할 수 있습니다. 배포 주소가 HTTPS여야 브라우저 카메라가 정상 작동하며, 최초 사용 시 카메라 권한을 허용해야 합니다.

평가는 accuracy뿐 아니라 `artifacts/metrics.json`의 클래스별 recall, F1, confusion matrix를 함께 봐야 합니다. 특히 흑색종(Acral Lentiginous Melanoma)은 놓치는 경우(false negative)가 중요합니다. 실제 배포 전에는 환자 단위 외부 테스트셋, 촬영 환경·피부색·연령대별 성능 검증, 전문의 검토가 필요합니다.
