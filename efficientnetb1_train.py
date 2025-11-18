import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os


# 데이터셋 클래스 정의
class AnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['00_Goat', '01_Wild boar', '02_Squirrel', '03_Raccoon',
                        '04_Asiatic black bear', '05_Hare', '06_Weasel',
                        '07_Haron', '08_Dog', '09_Cat']
        self.images = []
        self.labels = []

        print(f"📁 데이터 로딩 중: {root_dir}")
        for idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)

            if not os.path.exists(class_dir):
                print(f"⚠️ 경고: {class_dir} 폴더가 없습니다!")
                continue

            files = [f for f in os.listdir(class_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            print(f"  {class_name}: {len(files)}장")

            for img_name in files:
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(idx)

        print(f"✅ 총 {len(self.images)}장 로드 완료\n")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"❌ 이미지 로드 실패: {self.images[idx]}")
            print(f"   오류: {e}")
            # 검은색 이미지 반환 (에러 방지)
            return torch.zeros(3, 240, 240), self.labels[idx]


# 데이터 전처리
train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("=" * 60)
print("동물 분류 모델 학습 시작")
print("=" * 60)
print()

# 데이터 로더 생성
train_dataset = AnimalDataset('datasets/training', transform=train_transform)
val_dataset = AnimalDataset('datasets/validation', transform=val_transform)

# 배치 크기: GPU 메모리에 따라 조정 (GTX 1650은 16 권장)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"배치 크기: {batch_size}")
print(f"학습 배치 수: {len(train_loader)}")
print(f"검증 배치 수: {len(val_loader)}")
print()

# EfficientNet-B1 모델 로드 및 수정
print("모델 로딩 중...")
model = models.efficientnet_b1(weights='IMAGENET1K_V1')  # 최신 방식
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 10)  # 10개 클래스

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 디바이스: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
print()

model = model.to(device)

# 학습 설정
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 학습 루프
num_epochs = 30
best_acc = 0.0

print("=" * 60)
print("학습 시작!")
print("=" * 60)
print()

for epoch in range(num_epochs):
    # 학습 모드
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 진행 상황 출력 (10 배치마다)
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100. * correct / total:.2f}%")

    train_acc = 100. * correct / total

    # 검증 모드
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_acc = 100. * val_correct / val_total

    print()
    print(f'Epoch [{epoch + 1}/{num_epochs}]')
    print(f'Train Loss: {running_loss / len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Acc: {val_acc:.2f}%')
    print(f'Best Val Acc: {best_acc:.2f}%')
    print("-" * 60)

    # 최고 성능 모델 저장
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': val_acc,
            'class_names': train_dataset.classes
        }, 'efficientnet_classifier_model.pth')
        print(f"✅ 새로운 최고 성능! 모델 저장됨 (Val Acc: {val_acc:.2f}%)")
        print("-" * 60)

    scheduler.step()

print()
print("=" * 60)
print(f'학습 완료! Best Validation Accuracy: {best_acc:.2f}%')
print("=" * 60)
print()

# 최종 모델 저장 (배포용)
print("배포용 모델 저장 중...")
torch.save({
    'model_state_dict': model.state_dict(),
    'class_names': ['00_Goat', '01_Wild boar', '02_Squirrel', '03_Raccoon',
                    '04_Asiatic black bear', '05_Hare', '06_Weasel',
                    '07_Haron', '08_Dog', '09_Cat'],
    'input_size': (240, 240),
    'model_architecture': 'efficientnet_b1',
    'num_classes': 10,
    'accuracy': best_acc
}, 'animal_classifier_model.pth')

print('✅ 최종 모델이 animal_classifier_model.pth로 저장되었습니다!')
print()
print("생성된 파일:")
print("  - efficientnet_classifier_model.pth (학습 중 최고 성능 모델)")
print("  - animal_classifier_model.pth (배포용 최종 모델)")