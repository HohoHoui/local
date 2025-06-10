import json
import os
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# 경로 설정
image_dir = 'data/train2017'           # 이미지가 저장된 폴더
json_path = 'data/annotations/instances_train2017.json' # COCO JSON 파일 경로

# COCO 객체 불러오기
coco = COCO(json_path)

# 카테고리 정보 가져오기
cat_id_to_name = {cat['id']: cat['name'] for cat in coco.loadCats(coco.getCatIds())}

# 시각화할 이미지 ID 리스트 (전체 중 일부만 테스트)
image_ids = coco.getImgIds()
for img_id in image_ids:  # 처음 5개만 확인
    img_info = coco.loadImgs(img_id)[0]
    image_path = os.path.join(image_dir, img_info['file_name'])
    
    # 이미지 불러오기
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지를 찾을 수 없음: {image_path}")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 해당 이미지에 대한 annotation 불러오기
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    # annotation 그리기
    for ann in anns:
        x, y, w, h = ann['bbox']
        category_id = ann['category_id']
        label = cat_id_to_name[category_id]

        # 바운딩 박스 그리기
        cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 2)
        # 라벨 텍스트
        cv2.putText(image, label, (int(x), int(y-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.title(f"Image ID: {img_id}")
    plt.axis('off')
    plt.show()
