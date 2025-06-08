import os
from pathlib import Path
import cv2
import albumentations as A
from tqdm import tqdm

# 전처리 파이프라인 정의
transform = A.Compose([
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
    A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
])

# 디렉토리 설정
train_img_dir = Path('/app/notebooks/pothole-detection-challenge/train/images')
val_img_dir = Path('/app/notebooks/pothole-detection-challenge/valid/images')
train_output_dir = Path('/app/notebooks/pothole-detection-challenge/train/preprocessed_images')
val_output_dir = Path('/app/notebooks/pothole-detection-challenge/valid/preprocessed_images')

# 출력 디렉토리 생성
train_output_dir.mkdir(parents=True, exist_ok=True)
val_output_dir.mkdir(parents=True, exist_ok=True)

# 이미지 전처리 및 저장 함수
def preprocess_and_save(input_dir, output_dir):
    for img_path in tqdm(list(input_dir.glob('*.jpg')), desc=f"Processing {input_dir}"):
        # 이미지 로드
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 전처리 적용
        augmented = transform(image=image)
        image = augmented['image']
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 저장
        output_path = output_dir / img_path.name
        cv2.imwrite(str(output_path), image)

# 학습 및 검증 데이터 전처리
preprocess_and_save(train_img_dir, train_output_dir)
preprocess_and_save(val_img_dir, val_output_dir)

print("Data preprocessing completed.")