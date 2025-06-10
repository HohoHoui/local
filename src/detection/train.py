# train.py
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection
from torch.utils.data import DataLoader
import torch
import os
def detr_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
    labels = [item["labels"] for item in batch]
    image_ids = torch.stack([item["image_id"] for item in batch])
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "labels": labels,
        "image_id": image_ids
    }

def train_detr(dataset):
    train_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=detr_collate_fn
    )

    
    num_classes = 5
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", ignore_mismatched_sizes=True,num_labels=num_classes)
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

   
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

    model.train()
    for epoch in range(30):
        print(f"[Epoch {epoch}]")
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = []
            for label in batch["labels"]:
                labels.append({
                    "class_labels": label["class_labels"].to(device),
                    "boxes": label["boxes"].to(device)
                })
            outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)
        
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"  Loss: {loss.item():.4f}")

    
    save_path = "results/detr_building/checkpoint.pth"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[INFO] 모델 저장 완료: {save_path}")
    
