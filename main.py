#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import math
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=1280)
    parser.add_argument("--height", help='cap height', type=int, default=720)

    parser.add_argument('--static_image_mode', action='store_true')
    parser.add_argument("--model_complexity",
                        help='model_complexity(0,1(default),2)',
                        type=int,
                        default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--rev_color', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    static_image_mode = args.static_image_mode
    model_complexity = args.model_complexity
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    rev_color = args.rev_color

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 色指定
    if rev_color:
        color = (255, 255, 255)
        bg_color = (100, 33, 3)
    else:
        color = (100, 33, 3)
        bg_color = (255, 255, 255)

    while True:
        display_fps = cvFpsCalc.get()

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image01 = copy.deepcopy(image)
        debug_image02 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, (0, 0), (image.shape[1], image.shape[0]),
                    bg_color,
                    thickness=-1)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        # 描画 ################################################################
        if results.pose_landmarks is not None:
            # 描画
            debug_image01 = draw_landmarks(
                debug_image01,
                results.pose_landmarks,
            )
            debug_image02 = draw_stick_figure(
                debug_image02,
                results.pose_landmarks,
                color=color,
                bg_color=bg_color,
            )
            
            # Thêm phân tích tư thế
            debug_image01 = analyze_posture(debug_image01, results.pose_landmarks)

        cv.putText(debug_image01, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(debug_image02, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA)

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #############################################################
        cv.imshow('Tokyo2020 Debug', debug_image01)
        cv.imshow('Tokyo2020 Pictogram', debug_image02)

    cap.release()
    cv.destroyAllWindows()


def draw_stick_figure(
        image,
        landmarks,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
        visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    # 各ランドマーク算出
    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append(
            [index, landmark.visibility, (landmark_x, landmark_y), landmark_z])

    # 脚の付け根の位置を腰の中点に修正
    right_leg = landmark_point[23]
    left_leg = landmark_point[24]
    leg_x = int((right_leg[2][0] + left_leg[2][0]) / 2)
    leg_y = int((right_leg[2][1] + left_leg[2][1]) / 2)

    landmark_point[23][2] = (leg_x, leg_y)
    landmark_point[24][2] = (leg_x, leg_y)

    # 距離順にソート
    sorted_landmark_point = sorted(landmark_point,
                                   reverse=True,
                                   key=lambda x: x[3])

    # 各サイズ算出
    (face_x, face_y), face_radius = min_enclosing_face_circle(landmark_point)

    face_x = int(face_x)
    face_y = int(face_y)
    face_radius = int(face_radius * 1.5)

    stick_radius01 = int(face_radius * (4 / 5))
    stick_radius02 = int(stick_radius01 * (3 / 4))
    stick_radius03 = int(stick_radius02 * (3 / 4))

    # 描画対象リスト
    draw_list = [
        11,  # 右腕
        12,  # 左腕
        23,  # 右脚
        24,  # 左脚
    ]

    # 背景色
    cv.rectangle(image, (0, 0), (image_width, image_height),
                 bg_color,
                 thickness=-1)

    # 顔 描画
    cv.circle(image, (face_x, face_y), face_radius, color, -1)

    # 腕/脚 描画
    for landmark_info in sorted_landmark_point:
        index = landmark_info[0]

        if index in draw_list:
            point01 = [p for p in landmark_point if p[0] == index][0]
            point02 = [p for p in landmark_point if p[0] == (index + 2)][0]
            point03 = [p for p in landmark_point if p[0] == (index + 4)][0]

            if point01[1] > visibility_th and point02[1] > visibility_th:
                image = draw_stick(
                    image,
                    point01[2],
                    stick_radius01,
                    point02[2],
                    stick_radius02,
                    color=color,
                    bg_color=bg_color,
                )
            if point02[1] > visibility_th and point03[1] > visibility_th:
                image = draw_stick(
                    image,
                    point02[2],
                    stick_radius02,
                    point03[2],
                    stick_radius03,
                    color=color,
                    bg_color=bg_color,
                )

    return image


def min_enclosing_face_circle(landmark_point):
    landmark_array = np.empty((0, 2), int)

    index_list = [1, 4, 7, 8, 9, 10]
    for index in index_list:
        np_landmark_point = [
            np.array(
                (landmark_point[index][2][0], landmark_point[index][2][1]))
        ]
        landmark_array = np.append(landmark_array, np_landmark_point, axis=0)

    center, radius = cv.minEnclosingCircle(points=landmark_array)

    return center, radius


def draw_stick(
        image,
        point01,
        point01_radius,
        point02,
        point02_radius,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
):
    cv.circle(image, point01, point01_radius, color, -1)
    cv.circle(image, point02, point02_radius, color, -1)

    draw_list = []
    for index in range(2):
        rad = math.atan2(point02[1] - point01[1], point02[0] - point01[0])

        rad = rad + (math.pi / 2) + (math.pi * index)
        point_x = int(point01_radius * math.cos(rad)) + point01[0]
        point_y = int(point01_radius * math.sin(rad)) + point01[1]

        draw_list.append([point_x, point_y])

        point_x = int(point02_radius * math.cos(rad)) + point02[0]
        point_y = int(point02_radius * math.sin(rad)) + point02[1]

        draw_list.append([point_x, point_y])

    points = np.array((draw_list[0], draw_list[1], draw_list[3], draw_list[2]))
    cv.fillConvexPoly(image, points=points, color=color)

    return image


def draw_landmarks(
    image,
    landmarks,
    # upper_body_only,
    visibility_th=0.5,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue

        if index == 0:  # 鼻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 右目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 右目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 右目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 左目：目頭
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  # 左目：瞳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 左目：目尻
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 右耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 左耳
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 口：左端
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 右肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 左肩
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 13:  # 右肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 左肘
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 右手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 左手首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 17:  # 右手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 左手1(外側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 右手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 左手2(先端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:  # 右手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:  # 左手3(内側端)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23:  # 腰(右側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 24:  # 腰(左側)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 25:  # 右ひざ
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 26:  # 左ひざ
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 27:  # 右足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 28:  # 左足首
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 29:  # 右かかと
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30:  # 左かかと
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:  # 右つま先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:  # 左つま先
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        # if not upper_body_only:
        if True:
            cv.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                       cv.LINE_AA)

    # 右目
    if landmark_point[1][0] > visibility_th and landmark_point[2][
            0] > visibility_th:
        cv.line(image, landmark_point[1][1], landmark_point[2][1],
                (0, 255, 0), 2)
    if landmark_point[2][0] > visibility_th and landmark_point[3][
            0] > visibility_th:
        cv.line(image, landmark_point[2][1], landmark_point[3][1],
                (0, 255, 0), 2)

    # 左目
    if landmark_point[4][0] > visibility_th and landmark_point[5][
            0] > visibility_th:
        cv.line(image, landmark_point[4][1], landmark_point[5][1],
                (0, 255, 0), 2)
    if landmark_point[5][0] > visibility_th and landmark_point[6][
            0] > visibility_th:
        cv.line(image, landmark_point[5][1], landmark_point[6][1],
                (0, 255, 0), 2)

    # 口
    if landmark_point[9][0] > visibility_th and landmark_point[10][
            0] > visibility_th:
        cv.line(image, landmark_point[9][1], landmark_point[10][1],
                (0, 255, 0), 2)

    # 肩
    if landmark_point[11][0] > visibility_th and landmark_point[12][
            0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[12][1],
                (0, 255, 0), 2)

    # 右腕
    if landmark_point[11][0] > visibility_th and landmark_point[13][
            0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[13][1],
                (0, 255, 0), 2)
    if landmark_point[13][0] > visibility_th and landmark_point[15][
            0] > visibility_th:
        cv.line(image, landmark_point[13][1], landmark_point[15][1],
                (0, 255, 0), 2)

    # 左腕
    if landmark_point[12][0] > visibility_th and landmark_point[14][
            0] > visibility_th:
        cv.line(image, landmark_point[12][1], landmark_point[14][1],
                (0, 255, 0), 2)
    if landmark_point[14][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
        cv.line(image, landmark_point[14][1], landmark_point[16][1],
                (0, 255, 0), 2)

    # 右手
    if landmark_point[15][0] > visibility_th and landmark_point[17][
            0] > visibility_th:
        cv.line(image, landmark_point[15][1], landmark_point[17][1],
                (0, 255, 0), 2)
    if landmark_point[17][0] > visibility_th and landmark_point[19][
            0] > visibility_th:
        cv.line(image, landmark_point[17][1], landmark_point[19][1],
                (0, 255, 0), 2)
    if landmark_point[19][0] > visibility_th and landmark_point[21][
            0] > visibility_th:
        cv.line(image, landmark_point[19][1], landmark_point[21][1],
                (0, 255, 0), 2)
    if landmark_point[21][0] > visibility_th and landmark_point[15][
            0] > visibility_th:
        cv.line(image, landmark_point[21][1], landmark_point[15][1],
                (0, 255, 0), 2)

    # 左手
    if landmark_point[16][0] > visibility_th and landmark_point[18][
            0] > visibility_th:
        cv.line(image, landmark_point[16][1], landmark_point[18][1],
                (0, 255, 0), 2)
    if landmark_point[18][0] > visibility_th and landmark_point[20][
            0] > visibility_th:
        cv.line(image, landmark_point[18][1], landmark_point[20][1],
                (0, 255, 0), 2)
    if landmark_point[20][0] > visibility_th and landmark_point[22][
            0] > visibility_th:
        cv.line(image, landmark_point[20][1], landmark_point[22][1],
                (0, 255, 0), 2)
    if landmark_point[22][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
        cv.line(image, landmark_point[22][1], landmark_point[16][1],
                (0, 255, 0), 2)

    # 胴体
    if landmark_point[11][0] > visibility_th and landmark_point[23][
            0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[23][1],
                (0, 255, 0), 2)
    if landmark_point[12][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
        cv.line(image, landmark_point[12][1], landmark_point[24][1],
                (0, 255, 0), 2)
    if landmark_point[23][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
        cv.line(image, landmark_point[23][1], landmark_point[24][1],
                (0, 255, 0), 2)

    if len(landmark_point) > 25:
        # 右足
        if landmark_point[23][0] > visibility_th and landmark_point[25][
                0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[25][1],
                    (0, 255, 0), 2)
        if landmark_point[25][0] > visibility_th and landmark_point[27][
                0] > visibility_th:
            cv.line(image, landmark_point[25][1], landmark_point[27][1],
                    (0, 255, 0), 2)
        if landmark_point[27][0] > visibility_th and landmark_point[29][
                0] > visibility_th:
            cv.line(image, landmark_point[27][1], landmark_point[29][1],
                    (0, 255, 0), 2)
        if landmark_point[29][0] > visibility_th and landmark_point[31][
                0] > visibility_th:
            cv.line(image, landmark_point[29][1], landmark_point[31][1],
                    (0, 255, 0), 2)

        # 左足
        if landmark_point[24][0] > visibility_th and landmark_point[26][
                0] > visibility_th:
            cv.line(image, landmark_point[24][1], landmark_point[26][1],
                    (0, 255, 0), 2)
        if landmark_point[26][0] > visibility_th and landmark_point[28][
                0] > visibility_th:
            cv.line(image, landmark_point[26][1], landmark_point[28][1],
                    (0, 255, 0), 2)
        if landmark_point[28][0] > visibility_th and landmark_point[30][
                0] > visibility_th:
            cv.line(image, landmark_point[28][1], landmark_point[30][1],
                    (0, 255, 0), 2)
        if landmark_point[30][0] > visibility_th and landmark_point[32][
                0] > visibility_th:
            cv.line(image, landmark_point[30][1], landmark_point[32][1],
                    (0, 255, 0), 2)
    return image

def check_neck_angle(landmarks):
    """Kiểm tra góc cổ"""
    # Lấy các điểm mốc cần thiết
    nose = landmarks.landmark[0]
    left_ear = landmarks.landmark[7]
    right_ear = landmarks.landmark[8]
    left_shoulder = landmarks.landmark[11]
    right_shoulder = landmarks.landmark[12]
    
    # Tính điểm trung bình của tai và vai
    ear_x = (left_ear.x + right_ear.x) / 2
    ear_y = (left_ear.y + right_ear.y) / 2
    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    
    # Tính góc giữa cổ và trục dọc
    neck_angle = math.degrees(math.atan2(ear_x - shoulder_x, ear_y - shoulder_y))
    
    return abs(neck_angle)

def check_back_posture(landmarks):
    """Kiểm tra tư thế lưng chi tiết hơn"""
    # Lấy các điểm mốc quan trọng
    nose = landmarks.landmark[0]  # Thêm mũi để tính góc đầu
    left_shoulder = landmarks.landmark[11]
    right_shoulder = landmarks.landmark[12]
    left_hip = landmarks.landmark[23]
    right_hip = landmarks.landmark[24]
    left_knee = landmarks.landmark[25]
    right_knee = landmarks.landmark[26]
    
    # Tính điểm trung bình
    nose_x = nose.x
    nose_y = nose.y
    nose_z = nose.z
    
    shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    shoulder_z = (left_shoulder.z + right_shoulder.z) / 2
    
    hip_x = (left_hip.x + right_hip.x) / 2
    hip_y = (left_hip.y + right_hip.y) / 2
    hip_z = (left_hip.z + right_hip.z) / 2
    
    knee_x = (left_knee.x + right_knee.x) / 2
    knee_y = (left_knee.y + right_knee.y) / 2
    knee_z = (left_knee.z + right_knee.z) / 2
    
    # Tính các góc
    # 1. Góc nghiêng đầu (head tilt)
    head_tilt = math.degrees(math.atan2(nose_x - shoulder_x, nose_y - shoulder_y))
    
    # 2. Góc nghiêng trước/sau của lưng
    forward_tilt = math.degrees(math.atan2(shoulder_z - hip_z, shoulder_y - hip_y))
    
    # 3. Góc nghiêng sang trái/phải
    side_tilt = math.degrees(math.atan2(shoulder_x - hip_x, shoulder_y - hip_y))
    
    # 4. Góc giữa lưng và đùi (góc ngồi)
    hip_angle = math.degrees(math.atan2(
        math.sqrt((hip_x - knee_x)**2 + (hip_z - knee_z)**2),
        hip_y - knee_y
    ))
    
    # Phân tích tư thế
    posture_status = {
        'straight': True,
        'messages': [],
        'status': 'Tot'
    }
    
    # Điều chỉnh các ngưỡng chuẩn cho tư thế ngồi
    THRESHOLDS = {
        'head_tilt': {'min': -180, 'max': -170},  # Khoảng -177° là chuẩn
        'forward_tilt': {'min': -170, 'max': -155},  # Khoảng -163° là chuẩn
        'side_tilt': {'min': 170, 'max': 180},  # Khoảng 176° là chuẩn
        'hip_angle': {'min': 170, 'max': 180}  # Khoảng 176° là chuẩn
    }
    
    # Kiểm tra góc nghiêng đầu
    if head_tilt < THRESHOLDS['head_tilt']['min'] or head_tilt > THRESHOLDS['head_tilt']['max']:
        posture_status['straight'] = False
        if head_tilt < THRESHOLDS['head_tilt']['min']:
            posture_status['messages'].append(f"Dau nga ve truoc ({head_tilt:.1f}°)")
        else:
            posture_status['messages'].append(f"Dau nga ve sau ({head_tilt:.1f}°)")
    
    # Kiểm tra góc nghiêng trước/sau của lưng
    if forward_tilt < THRESHOLDS['forward_tilt']['min']:
        posture_status['straight'] = False
        posture_status['messages'].append(f"Lung nga ve truoc ({forward_tilt:.1f}°)")
    elif forward_tilt > THRESHOLDS['forward_tilt']['max']:
        posture_status['straight'] = False
        posture_status['messages'].append(f"Lung nga ve sau ({forward_tilt:.1f}°)")
    
    # Kiểm tra góc nghiêng sang trái/phải
    if side_tilt < THRESHOLDS['side_tilt']['min']:
        posture_status['straight'] = False
        posture_status['messages'].append(f"Lung nghieng trai ({side_tilt:.1f}°)")
    elif side_tilt > THRESHOLDS['side_tilt']['max']:
        posture_status['straight'] = False
        posture_status['messages'].append(f"Lung nghieng phai ({side_tilt:.1f}°)")
    
    # Kiểm tra góc ngồi
    if hip_angle < THRESHOLDS['hip_angle']['min']:
        posture_status['straight'] = False
        posture_status['messages'].append(f"Ngoi qua thang ({hip_angle:.1f}°)")
    elif hip_angle > THRESHOLDS['hip_angle']['max']:
        posture_status['straight'] = False
        posture_status['messages'].append(f"Ngoi qua nga ({hip_angle:.1f}°)")
            
    posture_status['angles'] = {
        'head_tilt': head_tilt,
        'forward_tilt': forward_tilt,
        'side_tilt': side_tilt,
        'hip_angle': hip_angle
    }
    
    # Đánh giá tổng thể
    if len(posture_status['messages']) == 0:
        posture_status['status'] = 'Tot'
    elif len(posture_status['messages']) <= 2:
        posture_status['status'] = 'Trung binh'
    else:
        posture_status['status'] = 'Xau'
    
    return posture_status

def check_eye_screen_distance(landmarks, image_width):
    """Ước tính khoảng cách từ mắt đến màn hình"""
    left_eye = landmarks.landmark[2]
    right_eye = landmarks.landmark[5]
    
    # Tính khoảng cách giữa 2 mắt theo pixel
    eye_distance = abs(left_eye.x - right_eye.x) * image_width
    
    # Ước tính khoảng cách (công thức đơn giản hóa)
    # Giả sử khoảng cách thực giữa 2 mắt là 6.3cm
    distance_cm = (6.3 * image_width) / (eye_distance * 10)
    
    return distance_cm

def analyze_posture(image, landmarks):
    """Phân tích tư thế và hiển thị cảnh báo"""
    image_height, image_width = image.shape[:2]
    
    # Kiểm tra các thông số
    back_posture = check_back_posture(landmarks)
    eye_distance = check_eye_screen_distance(landmarks, image_width)
    
    # Chuẩn bị thông báo
    messages = []
    
    # Thêm các cảnh báo về tư thế
    messages.extend(back_posture['messages'])
    
    # Kiểm tra khoảng cách màn hình
    if eye_distance < 4:
        messages.append(f"Khoang cach man hinh qua gan ({eye_distance:.1f}cm)")
    elif eye_distance > 12:
        messages.append(f"Khoang cach man hinh qua xa ({eye_distance:.1f}cm)")
        
    # Hiển thị thông số
    y_pos = 60  # Đổi từ 30 thành 60 để tránh overlap với FPS
    
    # Hiển thị trạng thái tổng thể
    status_color = (0, 255, 0) if back_posture['status'] == 'Tot' else \
                  (0, 255, 255) if back_posture['status'] == 'Trung binh' else \
                  (0, 0, 255)
    cv.putText(image, f"Trang thai: {back_posture['status']}", (10, y_pos),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    y_pos += 30
    
    # Cập nhật hiển thị các góc với ngưỡng mới
    angles = back_posture['angles']
    cv.putText(image, f"Goc dau: {angles['head_tilt']:.1f}° (chuan: -180° ~ -170°)", 
               (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_pos += 25
    cv.putText(image, f"Goc lung truoc/sau: {angles['forward_tilt']:.1f}° (chuan: -170° ~ -155°)", 
               (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_pos += 25
    cv.putText(image, f"Goc lung trai/phai: {angles['side_tilt']:.1f}° (chuan: 170° ~ 180°)", 
               (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    y_pos += 25
    cv.putText(image, f"Goc ngoi: {angles['hip_angle']:.1f}° (chuan: 170° ~ 180°)", 
               (10, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    # Hiển thị cảnh báo
    y_pos += 30
    for msg in messages:
        cv.putText(image, msg, (10, y_pos),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_pos += 25
    
    return image

if __name__ == '__main__':
    main()
