# dataset.py
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import os
from pycocotools.coco import COCO

class BuildingDataset(Dataset):
    def __init__(self, json_file, image_dir, processor):
        self.coco = COCO(json_file)
        self.image_dir = image_dir  # 이제 여기에는 명확한 경로가 들어와야 함
        self.processor = processor
        self.image_ids = self.coco.getImgIds()

    def __getitem__(self, idx):
        # image_id = self.image_ids[idx]
        # ann_ids = self.coco.getAnnIds(imgIds=image_id)
        # anns = self.coco.loadAnns(ann_ids)
        # img_info = self.coco.loadImgs(image_id)[0]

        # # 여기서 이미지 경로 직접 연결
        # path = os.path.join(self.image_dir, img_info['file_name'])

        # image = cv2.imread(path)
        # if image is None:
        #     raise FileNotFoundError(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {path}")
        image_id = self.image_ids[idx]
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(image_id)[0]

        # 안전한 경로 처리
        path = os.path.abspath(os.path.join(self.image_dir, img_info['file_name']))
        path = os.path.normpath(path)

        if not os.path.exists(path):
            raise FileNotFoundError(f"[ERROR] 파일 없음: {path}")

        # OpenCV 안전하게 이미지 로딩 (한글 경로 대응 포함)
        with open(path, "rb") as f:
            image_data = f.read()
        image_array = np.asarray(bytearray(image_data), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise FileNotFoundError(f"[ERROR] OpenCV가 이미지를 해독하지 못했습니다: {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # bbox 변환 (xywh → xyxy)
        xyxy_boxes = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            xyxy_boxes.append([x, y, x + w, y + h])

        boxes = torch.tensor(xyxy_boxes, dtype=torch.float32)
        class_labels = torch.tensor([ann['category_id'] for ann in anns], dtype=torch.int64)

        encoding = self.processor(
            images=image,
            return_tensors="pt",
            size={"height": 800, "width": 800}  # 원하는 고정 크기
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "pixel_mask": encoding["pixel_mask"].squeeze(0),
            "labels": {
                "class_labels": class_labels,  # 1D tensor (N,)
                "boxes": boxes                # 2D tensor (N, 4)
            },
            "image_id": torch.tensor(image_id, dtype=torch.int64)
        }

    def __len__(self):
        return len(self.image_ids)
