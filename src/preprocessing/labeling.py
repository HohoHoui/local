import xml.etree.ElementTree as ET
import cv2
import os

input_img = 'building_a_01.jpg'
input_xml = 'building_a_01.xml'
new_size = (224, 224)

# 1. 이미지 읽기 및 리사이즈
img = cv2.imread(f"images/{input_img}")
original_size = (img.shape[1], img.shape[0])  # (width, height)
resized_img = cv2.resize(img, new_size)
cv2.imwrite(f"processed_images/{input_img}", resized_img)

# 2. XML 파싱 및 좌표 정규화
tree = ET.parse(f"annotations/{input_xml}")
root = tree.getroot()

for obj in root.iter("object"):
    bbox = obj.find("bndbox")
    xmin = int(bbox.find("xmin").text)
    xmax = int(bbox.find("xmax").text)
    ymin = int(bbox.find("ymin").text)
    ymax = int(bbox.find("ymax").text)

    scale_x = new_size[0] / original_size[0]
    scale_y = new_size[1] / original_size[1]

    # 좌표 스케일 변환
    xmin = int(xmin * scale_x)
    xmax = int(xmax * scale_x)
    ymin = int(ymin * scale_y)
    ymax = int(ymax * scale_y)

    # 필요시 변환된 값을 다른 포맷(txt 등)으로 저장 가능
    print(f"normalized bbox: {xmin}, {ymin}, {xmax}, {ymax}")
