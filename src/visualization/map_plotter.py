# src/visualization/map_plotter.py

import cv2
import json
import os

def plot_pin_on_map(building_id, map_path, pin_path, position_json_path, output_path):
    # 1. 이미지 로드
    map_img = cv2.imread(map_path)
    pin_img = cv2.imread(pin_path, cv2.IMREAD_UNCHANGED)  # alpha 포함 PNG

    if map_img is None:
        raise ValueError("map.jpg를 찾을 수 없습니다.")
    if pin_img is None or pin_img.shape[2] != 4:
        raise ValueError("pin.png는 배경이 투명한 4채널 PNG여야 합니다.")

    # 2. 위치 좌표 불러오기
    with open(position_json_path, "r") as f:
        position_data = json.load(f)

    if building_id not in position_data:
        raise ValueError(f"지도 상에 '{building_id}'의 위치 정보가 없습니다.")

    x, y = position_data[building_id]['x'], position_data[building_id]['y']

    # 3. pin 이미지 크기 및 위치 조정
    pin_h, pin_w = pin_img.shape[:2]
    x_offset = x - pin_w // 2
    y_offset = y - pin_h // 2

    # 4. 알파 블렌딩으로 pin을 지도에 합성
    for c in range(3):  # BGR 채널
        map_img[y_offset:y_offset+pin_h, x_offset:x_offset+pin_w, c] = (
            pin_img[:, :, c] * (pin_img[:, :, 3] / 255.0) +
            map_img[y_offset:y_offset+pin_h, x_offset:x_offset+pin_w, c] * (1.0 - pin_img[:, :, 3] / 255.0)
        ).astype('uint8')

    # 5. 결과 저장
    cv2.imwrite(output_path, map_img)
    print(f"✅ map에 핀 표시 완료 → {output_path}")
