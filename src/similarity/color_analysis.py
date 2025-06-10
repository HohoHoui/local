import json
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt

# 대표 색상 추출 함수
def extract_dominant_colors(image_path, bbox, k=3):
    img = cv2.imread(image_path)
    x, y, w, h = map(int, bbox)
    roi = img[y:y+h, x:x+w]
    roi = roi.reshape((-1, 3))

    # 너무 작은 박스는 건너뜀
    if roi.shape[0] < k:
        return []

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(roi)
    return kmeans.cluster_centers_.astype(int)

# 시각화 (옵션)
def show_colors(colors, title=None):
    swatch = np.zeros((50, 50 * len(colors), 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        swatch[:, i*50:(i+1)*50, :] = color
    plt.imshow(cv2.cvtColor(swatch, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

# 경로 설정
coco_json_path = '../data/colorAndImg/D6/instances_val2017.json'
image_root = '../data/colorAndImg/D6'

# COCO 로드
coco = COCO(coco_json_path)

# 전체 색상 누적 리스트
all_colors = []

# 전체 annotation에 대해 반복
for ann in coco.dataset['annotations']:
    image_info = coco.loadImgs(ann['image_id'])[0]
    file_name = image_info['file_name']
    image_path = os.path.join(image_root, file_name)

    bbox = ann['bbox']
    colors = extract_dominant_colors(image_path, bbox, k=3)

    print(f"Image: {file_name}, Object ID: {ann['id']}, Dominant Colors (BGR):\n {colors}")
    show_colors(colors)

    for color in colors:
        all_colors.append(color)

# === 전체 색상 중에서 가장 흔한 5가지 색상 ===
if len(all_colors) >= 5:
    all_colors_np = np.array(all_colors)

    final_kmeans = KMeans(n_clusters=5)
    final_kmeans.fit(all_colors_np)
    final_centers = final_kmeans.cluster_centers_.astype(int)

    print("\n 전체 이미지에서 가장 많이 등장한 대표 색상 5가지 (BGR):")
    for i, color in enumerate(final_centers):
        print(f"{i+1}. {color}")

    show_colors(final_centers, title="Top 5 Dominant Colors in All Images")
else:
    print("객체가 너무 적어서 대표 색상 5개를 추출할 수 없습니다.")