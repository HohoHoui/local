import albumentations as A
import cv2
import numpy as np
import os
import json
import shutil

IMAGE_DIR = "../data/train2017"
ANNOTATION_PATH = "../data/annotations/instances_train2017.json"
OUTPUT_IMAGE_DIR = "../data/augmented/train2017"
OUTPUT_ANNOTATION_PATH = "../data/augmented/instances_train2017.json"

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.3),
    A.Blur(p=0.1),
    A.ColorJitter(p=0.3),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

with open(ANNOTATION_PATH) as f:
    coco = json.load(f)

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
new_images = []
new_annotations = []
ann_id = 0
image_id = 0

for img in coco['images']:
    image_path = os.path.join(IMAGE_DIR, img['file_name'])
    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    anns = [ann for ann in coco['annotations'] if ann['image_id'] == img['id']]
    bboxes = [ann['bbox'] for ann in anns]
    category_ids = [ann['category_id'] for ann in anns]

    for i in range(5):  # 증강 5배
        augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        aug_img = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_category_ids = augmented['category_ids']

        new_file_name = f"{image_id:06d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_IMAGE_DIR, new_file_name), aug_img)

        new_images.append({
            "id": image_id,
            "file_name": new_file_name,
            "height": height,
            "width": width
        })

        for box, cat in zip(aug_bboxes, aug_category_ids):
            new_annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "bbox": box,
                "category_id": cat,
                "iscrowd": 0,
                "area": box[2] * box[3]
            })
            ann_id += 1

        image_id += 1

augmented_coco = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": coco["categories"]
}
with open(OUTPUT_ANNOTATION_PATH, "w") as f:
    json.dump(augmented_coco, f)
