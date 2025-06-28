import mediapipe as mp
import cv2 as cv
import time
import numpy as np
import random  # 追加

mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=False
)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv.VideoCapture(0)
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
fps_history = []
history_size = 10

EYE_NOSE_MOUTH_IDX = [1, 2, 98, 327, 168, 197, 195, 5]
ALL_IDX = set(range(468))
OTHER_IDX = list(ALL_IDX - set(EYE_NOSE_MOUTH_IDX))

# ランダムカラーを1秒ごとに更新するための変数
face_line_colors = {}
last_color_update = time.time()

while cap.isOpened():
    tick = cv.getTickCount()
    start_time = time.time()

    # 1秒ごとに顔ラインの色を更新
    if time.time() - last_color_update > 1.0:
        face_line_colors = {}
        last_color_update = time.time()

    success, image = cap.read()
    if not success:
        continue

    image_rgb = cv.cvtColor(cv.flip(image, 1), cv.COLOR_BGR2RGB)
    black_cont = image_rgb.copy(); black_cont[:] = 0

    image_rgb.flags.writeable = False
    face_results = face_mesh.process(image_rgb)
    hand_results = hands.process(image_rgb)
    image_rgb.flags.writeable = True

    # 顔メッシュ描画
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            for start_idx, end_idx in mp_face_mesh.FACEMESH_TESSELATION:
                if start_idx in OTHER_IDX and end_idx in OTHER_IDX:
                    # 線ごとに色を保持・1秒ごとに更新
                    key = (start_idx, end_idx)
                    if key not in face_line_colors:
                        # 赤～紫の範囲（BGR: 青0-128, 緑0, 赤128-255）
                        r = random.randint(128, 255)
                        g = 0
                        b = random.randint(0, 128)
                        face_line_colors[key] = (b, g, r)
                    color = face_line_colors[key]
                    lm_start = face_landmarks.landmark[start_idx]
                    lm_end = face_landmarks.landmark[end_idx]
                    x1, y1 = int(lm_start.x * w), int(lm_start.y * h)
                    x2, y2 = int(lm_end.x * w), int(lm_end.y * h)
                    cv.line(black_cont, (x1, y1), (x2, y2), color, 1)

    # 手メッシュ描画（三角形で）
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # 手のランドマークは21点
            # 三角形の組み合わせ例（親指-人差し指-中指の付け根など）
            triangles = [
                (0, 1, 2), (0, 2, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17), (0, 17, 1),
                (1, 2, 3), (2, 3, 4),
                (5, 6, 7), (6, 7, 8),
                (9, 10, 11), (10, 11, 12),
                (13, 14, 15), (14, 15, 16),
                (17, 18, 19), (18, 19, 20)
            ]
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            for t in triangles:
                pts = np.array([points[i] for i in t], dtype=np.int32)
                cv.polylines(black_cont, [pts], isClosed=True, color=(255, 0, 0), thickness=1)
                cv.line(black_cont, pts[0], pts[1], (255, 0, 0), 1)
                cv.line(black_cont, pts[1], pts[2], (255, 0, 0), 1)
                cv.line(black_cont, pts[2], pts[0], (255, 0, 0), 1)

    black_cont_bgr = cv.cvtColor(black_cont, cv.COLOR_RGB2BGR)

    cv.imshow('Contours', black_cont_bgr)
    if cv.waitKey(5) & 0xFF == 27:
        break

    # 30fpsに制限
    elapsed = time.time() - start_time
    wait = max(0, (1/30) - elapsed)
    if wait > 0:
        time.sleep(wait)

face_mesh.close()
hands.close()
cap.release()
