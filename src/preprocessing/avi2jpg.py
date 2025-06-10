import cv2
import os

# === 설정 ===
video_path = 'data/raw_image/dormitory2d2.mp4'      # 입력 동영상 경로
output_dir = 'data/raw_image'      # 출력 디렉토리
interval_seconds = 5                # 몇 초마다 한 장씩 저장할지

# === 출력 폴더 생성 ===
os.makedirs(output_dir, exist_ok=True)

# === 동영상 읽기 ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)     # 프레임률 (frames per second)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"FPS: {fps}, Total Frames: {total_frames}, Duration: {duration:.2f}s")

# === 프레임 저장 ===
frame_interval = int(fps * interval_seconds)
frame_num = 0
saved_count = 0

while cap.isOpened():
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        break

    # 이미지 저장
    output_path = os.path.join(output_dir, f"frame_{int(frame_num):06d}.jpg")
    cv2.imwrite(output_path, frame)
    print(f"Saved {output_path}")
    saved_count += 1

    frame_num += frame_interval

cap.release()
print(f"\n총 {saved_count}장의 이미지가 저장되었습니다.")
