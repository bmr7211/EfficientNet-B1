import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# 동물 클래스 정의
CLASS_NAMES = ['00_Goat', '01_Wild boar', '02_Squirrel', '03_Raccoon',
               '04_Asiatic black bear', '05_Hare', '06_Weasel',
               '07_Haron', '08_Dog', '09_Cat']

def load_model(model_path, num_classes=10, device='cuda'):
    """학습된 모델 로드"""
    # 체크포인트 먼저 로드하여 정보 확인
    checkpoint = torch.load(model_path, map_location=device)

    # EfficientNet-B1 모델 구조 생성 (학습 시 사용한 것과 동일)
    from torchvision.models import EfficientNet_B1_Weights
    model = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)

    # Classifier 수정
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)

    # 체크포인트 형식 확인 및 로드
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # 전체 체크포인트 형식
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 모델 로드 완료: {model_path}")
        print(f"📌 Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"📌 정확도: {checkpoint.get('accuracy', 'N/A'):.2f}%")
        if 'class_names' in checkpoint:
            print(f"📌 클래스: {checkpoint['class_names']}")
    else:
        # 모델 가중치만 있는 형식
        model.load_state_dict(checkpoint)
        print(f"✅ 모델 로드 완료: {model_path}")

    model = model.to(device)
    model.eval()  # 평가 모드로 설정

    return model


def preprocess_image(image_path):
    """이미지 전처리"""
    transform = transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # 배치 차원 추가

    return image, image_tensor


def predict(model, image_tensor, device='cuda'):
    """예측 수행"""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return predicted.item(), confidence.item(), probabilities[0]


def show_prediction(image, predicted_class, confidence, all_probs, class_names):
    """예측 결과 시각화"""
    plt.figure(figsize=(12, 5))

    # 원본 이미지 표시
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Class: {class_names[predicted_class]}\nAccuracy: {confidence * 100:.2f}%',
              fontsize=14, fontweight='bold')
    plt.axis('off')

    # 확률 분포 그래프
    plt.subplot(1, 2, 2)
    probs = all_probs.cpu().numpy() * 100
    colors = ['green' if i == predicted_class else 'gray' for i in range(len(class_names))]
    plt.barh(class_names, probs, color=colors)
    plt.xlabel('probability (%)', fontsize=12)
    plt.title('Predicted probability for each class', fontsize=12)
    plt.xlim(0, 100)

    # Top 3 예측 결과 출력
    top3_prob, top3_idx = torch.topk(all_probs, 3)
    print("\n=== Top 3 예측 결과 ===")
    for i, (idx, prob) in enumerate(zip(top3_idx, top3_prob), 1):
        print(f"{i}. {class_names[idx]}: {prob * 100:.2f}%")

    plt.tight_layout()
    plt.show()


def classify_image(image_path, model_path='efficientnet_classifier_model.pth'):
    """이미지 분류 메인 함수"""
    # GPU 사용 가능 여부 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")

    # 모델 로드
    model = load_model(model_path, num_classes=len(CLASS_NAMES), device=device)

    # 이미지 전처리
    print(f"\n이미지 로드 중: {image_path}")
    original_image, image_tensor = preprocess_image(image_path)

    # 예측 수행
    print("예측 수행 중...")
    predicted_class, confidence, all_probs = predict(model, image_tensor, device)

    # 결과 출력
    print(f"\n{'=' * 50}")
    print(f"🎯 예측 결과: {CLASS_NAMES[predicted_class]}")
    print(f"📊 신뢰도: {confidence * 100:.2f}%")
    print(f"{'=' * 50}")

    # 시각화
    show_prediction(original_image, predicted_class, confidence, all_probs, CLASS_NAMES)

    return CLASS_NAMES[predicted_class], confidence


# 사용 예시
if __name__ == "__main__":
    # 분류할 이미지 경로 입력
    image_path = input("분류할 이미지 경로를 입력하세요: ")

    try:
        predicted_animal, confidence = classify_image(
            image_path=image_path,
            model_path='efficientnet_classifier_model.pth'
        )
        print(f"\n최종 결과: 이 동물은 '{predicted_animal}' 입니다! (신뢰도: {confidence * 100:.2f}%)")
    except FileNotFoundError as e:
        print(f"❌ 오류: 파일을 찾을 수 없습니다 - {e}")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")