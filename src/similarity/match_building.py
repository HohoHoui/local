import numpy as np

def match_building_by_class_id(detected_class_id, building_db):
    for entry in building_db:
        if entry["class_id"] == detected_class_id:
            return entry["building_id"]
    return None

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def match_building_by_color_vector(input_color_vector, building_db):
    similarities = []
    for entry in building_db:
        db_vector = np.array(entry["color_vector"])
        sim = cosine_similarity(input_color_vector, db_vector)
        similarities.append((entry["building_id"], sim))
    # 가장 높은 유사도 반환
    similarities.sort(key=lambda x: -x[1])  # cosine similarity는 클수록 유사
    return similarities[0][0] if similarities else None

def match_building(detected_class_id, input_color_vector, building_db):
    # 1. DETR class_id 우선 활용
    match = match_building_by_class_id(detected_class_id, building_db)
    if match:
        return match

    # 2. fallback → 색상 기반 유사도 분석
    return match_building_by_color_vector(input_color_vector, building_db)
