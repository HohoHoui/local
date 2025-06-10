# from detection.infer_detr import load_model, detect_objects

# model = load_model("../result/detr_building/checkpoint.pth", num_classes=5)  # ← 실제 클래스 수

# result = detect_objects("data/D2_test2.jpg", model)

# for r in result:
#     print(f"{r} {r['class_name']} ({r['score']:.2f}): {r['bbox']}")

# from detection.infer_detr import load_model, detect_objects
# from similarity.match_building import match_building
# from similarity.color_analysis import extract_dominant_colors
# import numpy as np
# import json
# from visualization.map_plotter import plot_pin_on_map
# #위치 표시
# plot_pin_on_map(
#     building_id="D2",
#     map_path="data/map/map.jpg",
#     pin_path="data/map/pin.png",
#     position_json_path="data/map/building_positions.json",
#     output_path="results/map_with_pin.jpg"
# )
# # 1. 모델 불러오기
# model = load_model("../result/detr_building/checkpoint.pth", num_classes=5)

# # 2. 대상 이미지 경로
# image_path = "data/D2_test2.jpg"
# result = detect_objects(image_path, model)

# # 3. building_color_db.json 불러오기
# with open("data/colorAndImg/building_color_db.json", "r") as f:
#     building_db = json.load(f)

# # 4. 결과 처리
# for r in result:
#     print(f"{r['class_name']} ({r['score']:.2f}): {r['bbox']}")

#     # N/A 무시함
#     # if r['class_name'] == "N/A":
#     #     continue

#     bbox = r['bbox']
#     # [xmin, ymin, xmax, ymax] → [x, y, w, h]
#     x, y, xmax, ymax = bbox
#     w = xmax - x
#     h = ymax - y
#     colors = extract_dominant_colors(image_path, [x, y, w, h], k=5)

#     if len(colors) < 5:
#         print("대표 색상 추출 실패")
#         continue

#     input_color_vector = np.array(colors).flatten()

#     matched = match_building(r["class_id"], input_color_vector, building_db)
#     print(f"예상 건물: {matched}")


# import argparse
# import os
# import json
# import numpy as np

# # 학습
# from detection.train import train_detr

# # 예측
# from detection.infer_detr import load_model, detect_objects
# from similarity.match_building import match_building
# from similarity.color_analysis import extract_dominant_colors
# from visualization.map_plotter import plot_pin_on_map

# # 데이터셋
# from transformers import DetrImageProcessor
# from detection.dataset import BuildingDataset


# def run_train_pipeline():
#     # 절대경로로 프로젝트 루트 잡기
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    
#     json_file = os.path.join(project_root, "data", "augmented", "instances_train2017.json")
#     image_dir = os.path.join(project_root, "data", "augmented", "train2017")

#     # Processor & Dataset 준비
#     processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
#     dataset = BuildingDataset(json_file=json_file, image_dir=image_dir, processor=processor)

#     # train_detr()는 dataset을 인자로 받도록 수정되어야 함
#     train_detr(dataset)


# def run_predict_pipeline():
#     # 위치 표시 (지도에 핀 찍기)
#     plot_pin_on_map(
#         building_id="D2",
#         map_path="data/map/map.jpg",
#         pin_path="data/map/pin.png",
#         position_json_path="data/map/building_positions.json",
#         output_path="results/map_with_pin.jpg"
#     )

#     # 1. 모델 불러오기
#     model = load_model("results/detr_building/checkpoint.pth", num_classes=5)

#     # 2. 대상 이미지 경로
#     image_path = "data/D2_test2.jpg"
#     result = detect_objects(image_path, model)

#     # 3. building_color_db.json 불러오기
#     with open("data/colorAndImg/building_color_db.json", "r") as f:
#         building_db = json.load(f)

#     # 4. 결과 처리
#     for r in result:
#         print(f"{r['class_name']} ({r['score']:.2f}): {r['bbox']}")
#         x, y, xmax, ymax = r['bbox']
#         w, h = xmax - x, ymax - y
#         colors = extract_dominant_colors(image_path, [x, y, w, h], k=5)

#         if len(colors) < 5:
#             print("대표 색상 추출 실패")
#             continue

#         input_color_vector = np.array(colors).flatten()
#         matched = match_building(r["class_id"], input_color_vector, building_db)
#         print(f"예상 건물: {matched}")


# # 명령어 파싱
# parser = argparse.ArgumentParser()
# parser.add_argument('--mode', choices=['train', 'predict'], default='train')
# args = parser.parse_args()

# # 실행 분기
# if args.mode == 'train':
#     run_train_pipeline()
# elif args.mode == 'predict':
#     run_predict_pipeline()


import os
import json
import numpy as np

from detection.infer_detr import load_model, detect_objects
from similarity.match_building import match_building
from similarity.color_analysis import extract_dominant_colors
from visualization.map_plotter import plot_pin_on_map

def predict_and_plot(image_path, model_path, num_classes, map_path, pin_path, position_json_path, output_path, color_db_path):
   
    model = load_model(model_path, num_classes=num_classes)

    
    result = detect_objects(image_path, model)

    
    with open(color_db_path, "r") as f:
        building_db = json.load(f)

   
    for r in result:
        print(f"{r['class_name']} ({r['score']:.2f}): {r['bbox']}")
        x, y, xmax, ymax = r['bbox']
        w, h = xmax - x, ymax - y
        colors = extract_dominant_colors(image_path, [x, y, w, h], k=5)

        if len(colors) < 5:
            print("대표 색상 추출 실패")
            continue

        input_color_vector = np.array(colors).flatten()
        matched = match_building(r["class_id"], input_color_vector, building_db)
        print(f"예상 건물: {matched}")

        plot_pin_on_map(
            building_id=matched,
            map_path=map_path,
            pin_path=pin_path,
            position_json_path=position_json_path,
            output_path=output_path
        )

if __name__ == "__main__":
    image_path = "data/D2_test2.jpg"
    model_path = "results/detr_building/checkpoint.pth"
    num_classes = 5
    map_path = "data/map/map.jpg"
    pin_path = "data/map/pin.png"
    position_json_path = "data/map/building_positions.json"
    output_path = "results/map_with_pin.jpg"
    color_db_path = "data/colorAndImg/building_color_db.json"

    predict_and_plot(
        image_path, model_path, num_classes,
        map_path, pin_path, position_json_path, output_path, color_db_path
    )
