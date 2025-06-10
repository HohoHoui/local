import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from argparse import Namespace

# 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../detr')))
from models import build_model

# 클래스 정의 (index 순서 중요)
CLASSES = [
    'D2', 'D6', 'D5', 'D15', 'D7', 'N/A'
]

# 이미지 전처리
transform = T.Compose([
    T.Resize(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

def load_model(checkpoint_path, num_classes):
    args = Namespace(
        num_classes=num_classes,
        hidden_dim=256,
        dropout=0.1,
        position_embedding='sine',
        backbone='resnet50',
        dilation=False,
        masks=False,
        device='cpu',
        aux_loss=False,

        # Optim 관련
        lr_backbone=1e-5,
        lr_drop=40,
        weight_decay=1e-4,
        backbone_lr=1e-5,

        # Matching loss weight들
        set_cost_class=1,
        set_cost_bbox=5,
        set_cost_giou=2,
        bbox_loss_coef=5,
        giou_loss_coef=2,
        eos_coef=0.1,

        # Transformer 관련
        nheads=8,
        num_queries=100,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        enc_layers=6,
        dec_layers=6,
        normalize_before=False,
        pre_norm=False,

        # Masks 관련
        mask_loss_coef=1,
        dice_loss_coef=1,
        frozen_weights=None,
        dataset_file="coco"
    )

    model, _, _ = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint)  # 'model' 키 없이 바로 사용
    model.eval()
    return model

def detect_objects(image_path, model, threshold=0.7):
    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)

    logits = outputs['pred_logits'][0]
    boxes = outputs['pred_boxes'][0]

    probs = logits.softmax(-1)
    scores, labels = probs.max(-1)

    keep = scores > threshold
    final_boxes = boxes[keep].cpu().numpy()
    final_labels = labels[keep].cpu().numpy()
    final_scores = scores[keep].cpu().numpy()

    img_w, img_h = image.size
    results = []
    for box, label, score in zip(final_boxes, final_labels, final_scores):
        cx, cy, w, h = box
        xmin = int((cx - w/2) * img_w)
        ymin = int((cy - h/2) * img_h)
        xmax = int((cx + w/2) * img_w)
        ymax = int((cy + h/2) * img_h)
        results.append({
            "bbox": [xmin, ymin, xmax, ymax],
            "class_id": int(label),
            "class_name": CLASSES[int(label)],
            "score": float(score)
        })

    return results
