import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from scipy.spatial import Delaunay
import glob
import sys

# 画像とカメラの準備

# 画像ファイルを拡張子で検索して取得
img_files = glob.glob('img/*.jpg') + glob.glob('img/*.png') + glob.glob('img/*.jpeg')
if not img_files:
    print("画像ファイルが見つかりません")
    exit()
src_img = cv.imread(img_files[0])  # 画像ファイルを指定
if src_img is None:
    print("画像が読み込めません")
    exit()
src_img = cv.cvtColor(src_img, cv.COLOR_BGR2RGB)
src_h, src_w = src_img.shape[:2]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=False)
face_mesh_video = mp_face_mesh.FaceMesh(static_image_mode=False, refine_landmarks=False)
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# コマンドライン引数で--debugや--nofaceがあればフラグを立てる
show_debug = '--debug' in sys.argv
no_face_mode = '--noface' in sys.argv

# 顔の特徴点を取得する関数
def get_face_landmarks(image, face_mesh):
    results = face_mesh.process(image)
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    h, w = image.shape[:2]
    points = np.array([[int(l.x * w), int(l.y * h)] for l in landmarks], np.int32)
    return points

# 唇の内側ランドマークインデックス（MediaPipe 468点モデル）
lips_inner_idx = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 42, 183]

# 画像から特徴点取得
def draw_mediapipe_landmarks(img, points, color=(0,255,0)):
    # points: (N,2) ndarray, img: RGB
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    h, w = img.shape[:2]
    for pt in points:
        lmk = landmark_pb2.NormalizedLandmark()
        lmk.x = pt[0] / w
        lmk.y = pt[1] / h
        lmk.z = 0
        landmark_list.landmark.append(lmk)
    mp.solutions.drawing_utils.draw_landmarks(
        img,
        landmark_list,
        connections=None,
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=color, thickness=2, circle_radius=2),
        connection_drawing_spec=None
    )

src_points = get_face_landmarks(src_img, face_mesh)
debug_img = src_img.copy()
if src_points is None:
    print("画像から顔の特徴点が検出できません")
    # samplemarker.npyが存在すれば読み込み、画像中央に平行移動して仮特徴点とする
    import os
    if os.path.exists('samplemarker.npy'):
        marker = np.load('samplemarker.npy')
        # markerの重心を画像中心に移動
        center_x, center_y = src_img.shape[1] // 2, src_img.shape[0] // 2
        marker_center = marker.mean(axis=0).astype(int)
        shift = np.array([center_x, center_y]) - marker_center
        src_points = marker + shift
        print('samplemarker.npyを中央に配置して仮特徴点としました')
        draw_mediapipe_landmarks(debug_img, src_points, color=(0,255,0))
    else:
        print('samplemarker.npyがありません。仮特徴点は表示しません。')
else:
    # 検出した特徴点をMediaPipe形式でdebug_imgに描画
    draw_mediapipe_landmarks(debug_img, src_points, color=(0,0,255))
# debugウインドウで特徴点付き画像を表示（--debug時のみ）
if show_debug:
    cv.imshow('debug', cv.cvtColor(debug_img, cv.COLOR_RGB2BGR))
    cv.waitKey(500)

# 唇の内側を灰色で塗りつぶす
if not no_face_mode:
    lips_poly = src_points[lips_inner_idx]
    gray_color = (128, 128, 128)
    cv.fillPoly(src_img, [lips_poly], gray_color)

# Delaunay三角分割用の三角形インデックスを作成
tri = Delaunay(src_points)
tri_indices = tri.simplices

# 三角形ごとにワーピングする関数
def warp_triangle(src, dst, t_src, t_dst):
    # バウンディングボックス
    r1 = cv.boundingRect(np.float32([t_src]))
    r2 = cv.boundingRect(np.float32([t_dst]))
    t1_rect = []
    t2_rect = []
    for i in range(3):
        t1_rect.append(((t_src[i][0] - r1[0]), (t_src[i][1] - r1[1])))
        t2_rect.append(((t_dst[i][0] - r2[0]), (t_dst[i][1] - r2[1])))
    # マスク作成
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv.fillConvexPoly(mask, np.int32(t2_rect), (1.0, 1.0, 1.0), 16, 0)
    # ソース三角形をアフィン変換
    img1_rect = src[r1[1]:r1[1]+r1[3], r1[0]:r1[0]+r1[2]]
    size = (r2[2], r2[3])
    mat = cv.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
    warped = cv.warpAffine(img1_rect, mat, size, None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    warped = warped * mask
    # 結果を合成
    dst_rect = dst[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]]
    h, w = mask.shape[:2]
    dst_h, dst_w = dst_rect.shape[:2]
    min_h = min(h, dst_h)
    min_w = min(w, dst_w)
    dst_rect = dst_rect[:min_h, :min_w]
    mask_c = mask[:min_h, :min_w]
    warped_c = warped[:min_h, :min_w]
    dst_rect = dst_rect * (1 - mask_c)
    dst_rect = dst_rect + warped_c
    dst[r2[1]:r2[1]+min_h, r2[0]:r2[0]+min_w] = dst_rect

# カメラ起動
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("カメラが開けません")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv.flip(frame, 1)  # 左右反転
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    dst_points = get_face_landmarks(frame_rgb, face_mesh_video)
    output = np.zeros_like(frame_rgb)
    # 顔ワーピング
    if dst_points is not None:
        for tri_idx in tri_indices:
            t_src = src_points[tri_idx]
            t_dst = dst_points[tri_idx]
            warp_triangle(src_img, output, t_src, t_dst)
    # 腕（Pose）検出と描画
    pose_results = pose.process(frame_rgb)
    if pose_results.pose_landmarks:
        h, w = frame_rgb.shape[:2]
        arm_pairs = [
            (11, 13), (13, 15),  # 左腕
            (12, 14), (14, 16)   # 右腕
        ]
        for idx1, idx2 in arm_pairs:
            # 前腕（肘-手首: 13-15, 14-16）は描画しない
            if (idx1, idx2) in [(13, 15), (14, 16)]:
                continue
            lm1 = pose_results.pose_landmarks.landmark[idx1]
            lm2 = pose_results.pose_landmarks.landmark[idx2]
            x1, y1 = int(lm1.x * w), int(lm1.y * h)
            x2, y2 = int(lm2.x * w), int(lm2.y * h)
            cv.line(output, (x1, y1), (x2, y2), (255, 0, 0), 8)  # 赤色
            # 肘（13,14）は残すが、手首（15,16）は点を描画しない
            if idx2 in [13, 14]:
                cv.circle(output, (x2, y2), 12, (255, 0, 0), -1)  # 赤色
            if idx1 in [13, 14]:
                cv.circle(output, (x1, y1), 12, (255, 0, 0), -1)  # 赤色
    # 手（Hands）検出と描画
    hand_results = hands.process(frame_rgb)
    hand_wrist_points = []
    if hand_results.multi_hand_landmarks:
        h, w = frame_rgb.shape[:2]
        hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # 親指
            (0, 5), (5, 6), (6, 7), (7, 8),      # 人差し指
            (0, 9), (9, 10), (10, 11), (11, 12), # 中指
            (0, 13), (13, 14), (14, 15), (15, 16), # 薬指
            (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        ]
        for hand_landmarks in hand_results.multi_hand_landmarks:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
            # 指定された三角形のみ描画
            triangles = [
                (0, 1, 2), (0, 2, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17), (0, 17, 1),
                (1, 2, 3), (2, 3, 4),
                (5, 6, 7), (6, 7, 8),
                (9, 10, 11), (10, 11, 12),
                (13, 14, 15), (14, 15, 16),
                (17, 18, 19), (18, 19, 20)
            ]
            for t in triangles:
                pts = np.array([points[i] for i in t], dtype=np.int32)
                cv.polylines(output, [pts], isClosed=True, color=(255, 0, 0), thickness=2)  # 赤色
            # 三角形の頂点も描画
            for idx in set([i for tri in triangles for i in tri]):
                x, y = points[idx]
                cv.circle(output, (x, y), 6, (255, 0, 0), -1)  # 赤色
            # 手首点(0番)をリストに保存
            hand_wrist_points.append(points[0])
    # 腕と手を線でつなぐ（手首0番と肘を線でつなぐ）
    if pose_results.pose_landmarks and hand_wrist_points:
        h, w = frame_rgb.shape[:2]
        # 右肘のインデックス: 13, 左肘のインデックス: 14
        right_elbow = pose_results.pose_landmarks.landmark[13]
        right_elbow_point = (int(right_elbow.x * w), int(right_elbow.y * h))
        left_elbow = pose_results.pose_landmarks.landmark[14]
        left_elbow_point = (int(left_elbow.x * w), int(left_elbow.y * h))
        # 右手・左手の0番点を探す
        if hand_results.multi_handedness:
            for i, handedness in enumerate(hand_results.multi_handedness):
                label = handedness.classification[0].label
                if label == 'Right':
                    right_hand_wrist = hand_wrist_points[i]
                    cv.line(output, right_hand_wrist, right_elbow_point, (255, 0, 0), 3)
                elif label == 'Left':
                    left_hand_wrist = hand_wrist_points[i]
                    cv.line(output, left_hand_wrist, left_elbow_point, (255, 0, 0), 3)
    
    output_bgr = cv.cvtColor(output, cv.COLOR_RGB2BGR)
    cv.imshow('Warped Face', output_bgr)
    if cv.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv.destroyAllWindows()
