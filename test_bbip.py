import cv2
import numpy as np
import math
import re
import os
import argparse
from ultralytics import YOLO
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 设置默认量程
set_start_value = 0
set_end_value = 6

# 变量初始化
prev_keypoints_array = None
prev_saved_boxes = None
prev_start_value = set_start_value
prev_end_value = set_end_value
temp1, temp2 = 20, 15

# 初始化 YOLO 检测模型
model = YOLO(r"model\pose\train\weights\best.pt")#前模型
#model = YOLO(r"runs\detect\train\weights\best.pt")#现模型

# 初始化 OCR 模型
ocr_recognition = pipeline(Tasks.ocr_recognition, model='cv_convnextTiny_ocr-recognition-document_damo')

# 解析命令行参数
parser = argparse.ArgumentParser(description="视频处理脚本")
parser.add_argument("video_source", help="待处理的视频源，可以是视频文件路径、摄像头编号或摄像头RTSP流地址")
args = parser.parse_args()

video_source = args.video_source
video_path = int(video_source) if video_source.isdigit() else video_source

output_path = "processed_output.mp4"

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("无法打开视频源")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def find_center_and_sort_keypoints(keypoints_array, img_height):
    if len(keypoints_array) != 4:
        return None

    transformed_keypoints = [(x, img_height - y) for x, y in keypoints_array]
    cx = sum(p[0] for p in transformed_keypoints) / 4
    cy = sum(p[1] for p in transformed_keypoints) / 4
    center_point = min(transformed_keypoints, key=lambda p: np.linalg.norm(np.array(p) - np.array([cx, cy])))
    remaining_points = [p for p in transformed_keypoints if p != center_point]

    if len(remaining_points) != 3:
        return None

    def angle_counter_clockwise(p):
        dx, dy = p[0] - center_point[0], p[1] - center_point[1]
        angle = math.degrees(math.atan2(dx, -dy))
        return (angle + 360) % 360

    sorted_points = sorted(remaining_points, key=angle_counter_clockwise)
    end_point, needle_point, start_point = sorted_points

    end_angle = angle_counter_clockwise(end_point)
    start_angle = angle_counter_clockwise(start_point)
    if (start_angle - end_angle) % 360 < 180:
        start_point, end_point = end_point, start_point

    def revert_y(p):
        return (p[0], img_height - p[1])

    return revert_y(center_point), revert_y(start_point), revert_y(needle_point), revert_y(end_point)

def ocr_read_number(frame, box):
    try:
        l, t, r, b, conf, cls_id = map(int, box)
        if cls_id == 1:
            b -= temp2
            l += temp1
        if cls_id == 2:
            b -= temp2
            r -= temp1
        roi = frame[t:b, l:r]
        ocr_results = ocr_recognition(roi)
        text_list = ocr_results.get('text', [])
        numbers = [re.findall(r'\d+\.\d+|\d+', t) for t in text_list]
        numbers = [num for sublist in numbers for num in sublist]
        return float(numbers[0]) if numbers else None
    except Exception as e:
        print(f"OCR 失败: {e}")
        return None

def transform_coordinates(point, img_height):
    return (point[0], img_height - point[1])

def clockwise_angle(A, B, O):
    ax, ay = A[0] - O[0], A[1] - O[1]
    bx, by = B[0] - O[0], B[1] - O[1]
    angle_A = math.degrees(math.atan2(ay, ax))
    angle_B = math.degrees(math.atan2(by, bx))
    return (angle_A - angle_B) % 360

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    orig_h, orig_w = frame.shape[:2]

    results = model.predict(frame, imgsz=640, conf=0.5)
    result = results[0]

    saved_boxes = []
    for detection in result.boxes.data.cpu().numpy():
        l, t, r, b, conf, cls_id = detection
        cls_id = int(cls_id)
        if cls_id in [1, 2]:
            saved_boxes.append([l, t, r, b, conf, cls_id])
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 0, 255), 2)

    if len(saved_boxes) < 2 and prev_saved_boxes is not None:
        saved_boxes = prev_saved_boxes.copy()
    prev_saved_boxes = saved_boxes.copy()

    try:
        start_value = ocr_read_number(frame, saved_boxes[0]) if len(saved_boxes) > 0 else None
        if start_value is None:
            start_value = prev_start_value
    except:
        start_value = prev_start_value

    try:
        end_value = ocr_read_number(frame, saved_boxes[1]) if len(saved_boxes) > 1 else None
        if end_value is None:
            end_value = prev_end_value
    except:
        end_value = prev_end_value

    prev_start_value, prev_end_value = start_value, end_value

    keypoints_array = []
    keypoints = result.keypoints.cpu().numpy().data if result.keypoints is not None else []
    for keypoint_set in keypoints:
        for keypoint in keypoint_set:
            x, y, conf = keypoint
            if conf > 0.5:
                keypoints_array.append((x, y))
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    if len(keypoints_array) < 4 and prev_keypoints_array is not None:
        keypoints_array = prev_keypoints_array.copy()
    prev_keypoints_array = keypoints_array.copy()

    sorted_keypoints = find_center_and_sort_keypoints(keypoints_array, orig_h)
    if sorted_keypoints is None:
        print(f"帧 {frame_idx} 关键点排序失败，跳过")
        out.write(frame)
        frame_idx += 1
        continue

    center_point, start_point, needle_point, end_point = sorted_keypoints

    if len(keypoints_array) < 4:
        print(f"帧 {frame_idx} 关键点检测失败，跳过")
        out.write(frame)
        frame_idx += 1
        continue

    cv2.circle(frame, (int(center_point[0]), int(center_point[1])), 5, (255, 255, 0), -1)
    cv2.circle(frame, (int(start_point[0]), int(start_point[1])), 5, (255, 0, 255), -1)
    cv2.circle(frame, (int(needle_point[0]), int(needle_point[1])), 5, (0, 255, 255), -1)
    cv2.circle(frame, (int(end_point[0]), int(end_point[1])), 5, (255, 0, 0), -1)

    center_point = transform_coordinates(center_point, orig_h)
    needle_point = transform_coordinates(needle_point, orig_h)
    start_point = transform_coordinates(start_point, orig_h)
    end_point = transform_coordinates(end_point, orig_h)

    angle_total = clockwise_angle(start_point, end_point, center_point)
    angle_pointer = clockwise_angle(start_point, needle_point, center_point)

    current_reading = start_value + (angle_pointer / angle_total) * (end_value - start_value)

    reading_text = f"Reading: {current_reading:.2f}"
    cv2.putText(frame, reading_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    out.write(frame)

    # 新增：显示实时画面
    cv2.imshow('Processed Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("检测到按下 q，提前结束处理")
        break

    frame_idx += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print("视频处理完成，保存为:", output_path)
