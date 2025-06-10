import cv2
import os

print("경로 : ", os.getcwd())

input_dir = 'data/raw_images/'
output_dir = 'data/processed_image/'
#os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if not fname.endswith('.jpg'):
        continue

    img_path = os.path.join(input_dir, fname)
    img = cv2.imread(img_path)

    # 1. 크기 정규화
    img = cv2.resize(img, (224, 224))

    # 2. 밝기 및 채도 조정 (HSV 방식)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 밝기 조정
    v = cv2.equalizeHist(v)

    # 채도 조정
    s = cv2.equalizeHist(s) 

    hsv_normalized = cv2.merge([h, s, v])
    img_normalized = cv2.cvtColor(hsv_normalized, cv2.COLOR_HSV2BGR)

    # 저장
    cv2.imwrite(os.path.join(output_dir, fname), img_normalized)