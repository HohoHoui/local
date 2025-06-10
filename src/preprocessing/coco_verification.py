from pycocotools.coco import COCO

coco = COCO("data/processed_image/D2_01.json")
for ann_id in coco.getAnnIds():
    ann = coco.loadAnns(ann_id)
    print(ann)