## 1. 라이브러리 설치
pip install -r requirements.txt

## 2. 분류 모델 확인
ls efficientnet_best_classifier_model.pth

## 3. 사용 방법

```python
from classifier_model_inference import load_model, predict_image

# 모델 로드
model, class_names = load_model('animal_classifier_model.pth')

# 이미지 분류
predicted_class, confidence = predict_image('your_image.jpg', model, class_names)

print(f'예측 결과: {predicted_class}')
print(f'신뢰도: {confidence:.2f}%')
```

## 모델 성능
- EfficientNet-B1
- parameters : ~7.8M
- 정확도: 94.5%
- 입력 크기: 224x224
- 추론 속도: ~50ms (CPU)

## 학습 파라미터
- batch : 16
- epoch : 30
- optimizer : Adam(lr=0.001)

## 데이터 증강
- 랜덤 수평 뒤집기
- 랜덤 회전(15도)
- 밝기, 대비 조정

## 시스템 요구사항
- Python 3.7+
- TensorFlow 2.x 또는 PyTorch 1.x
- 메모리: 최소 2GB RAM