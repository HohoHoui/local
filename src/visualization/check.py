import cv2
import json
import matplotlib.pyplot as plt

# 이미지 및 JSON 경로
map_path = "data/map/map.jpg"
pin_path = "data/map/pin.png"
json_path = "data/map/building_positions.json"

# 이미지 불러오기
map_img = cv2.imread(map_path)
pin_img = cv2.imread(pin_path, cv2.IMREAD_UNCHANGED)  # PNG의 알파 채널 유지

# JSON 불러오기
with open(json_path, 'r') as f:
    positions = json.load(f)

# D2 위치 가져오기
d2_pos = positions.get("D7")
if not d2_pos:
    raise ValueError("D2 위치가 JSON 파일에 없습니다.")

# pin 이미지 크기
pin_h, pin_w = pin_img.shape[:2]

# map 이미지에 pin 오버레이 (투명도 유지)
def overlay_pin(background, pin, x, y):
    """ pin의 중심이 (x, y)에 오도록 오버레이 """
    bh, bw = background.shape[:2]
    ph, pw = pin.shape[:2]

    # 중심 정렬
    x -= pw // 2
    y -= ph // 2

    # 범위 확인
    if x < 0 or y < 0 or x+pw > bw or y+ph > bh:
        print("경고: 핀이 이미지 범위를 벗어납니다.")
        return background

    alpha_s = pin[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s

    for c in range(3):
        background[y:y+ph, x:x+pw, c] = (
            alpha_s * pin[:, :, c] + alpha_l * background[y:y+ph, x:x+pw, c]
        )

    return background

# 핀을 지도 위에 올리기
map_with_pin = overlay_pin(map_img.copy(), pin_img, d2_pos['x'], d2_pos['y'])

# 시각화
plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(map_with_pin, cv2.COLOR_BGR2RGB))
plt.title("Displayed in D2")
plt.axis('off')
plt.show()
