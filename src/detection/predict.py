# 예측 및 결과 bbox 시각화
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.detection.dataset import BuildingDataset
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def predict():
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model.load_state_dict(torch.load("models/detr_building.pth"))
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    model.eval().to("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BuildingDataset("data/annotations/instances_train2017.json", "data/train2017", processor)

    for i in range(3):
        sample = dataset[i]
        inputs = {k: v.unsqueeze(0).to("cuda") for k, v in sample.items() if k != "labels"}
        output = model(**inputs)
        results = processor.post_process_object_detection(output, target_sizes=[sample["pixel_values"].shape[1:]], threshold=0.9)[0]

        # 시각화
        import numpy as np
        image = sample["pixel_values"].permute(1,2,0).cpu().numpy()
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x, y, w, h = box
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-5, f"Class {label.item()} ({score:.2f})", color='red')
        plt.axis('off')
        plt.savefig(f"results/detr_building/result_{i}.png")
        plt.close()
