import cv2
import numpy as np
import math
import re
import os
from ultralytics import YOLO
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 设置默认量程
set_start_value = 0  
set_end_value = 6

# 变量初始化
prev_keypoints_array = None  # 存储上一帧关键点
prev_saved_boxes = None  # 存储上一帧检测框
prev_start_value = set_start_value  # 记录上一帧OCR起点
prev_end_value = set_end_value  # 记录上一帧OCR终点
temp1, temp2 = 20, 15  # OCR 识别框微调参数

# 初始化 YOLO 检测模型
model = YOLO("runs/pose/train/weights/best.pt")

# 初始化 OCR 模型
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-document_damo')

# 读取视频
video_path = "MP41.mp4"
output_path = "processed_output.mp4"
cap = cv2.VideoCapture(video_path)

# 获取视频基本信息
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 定义视频写入器
fourcc = cv2.VideoWriter_fourcc(*"avc1")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))


def find_center_and_sort_keypoints(keypoints_array, img_height):
    """
    识别中心点，并按 **负 Y 轴为起点，逆时针排序**，返回 (center, start, needle, end)
    """
    if len(keypoints_array) != 4:
        return None  # 关键点不足 4 个，无法识别

    # **1. 转换 Y 轴 (图像 -> 数学坐标系)**
    transformed_keypoints = [(x, img_height - y) for x, y in keypoints_array]

    # **2. 计算几何中心**
    cx = sum(p[0] for p in transformed_keypoints) / 4
    cy = sum(p[1] for p in transformed_keypoints) / 4

    # **3. 识别中心点 (几何中心最近的点)**
    center_point = min(transformed_keypoints, key=lambda p: np.linalg.norm(np.array(p) - np.array([cx, cy])))

    # **4. 获取剩余 3 点**
    remaining_points = [p for p in transformed_keypoints if p != center_point]

    if len(remaining_points) != 3:
        return None  # 防止错误

    # **5. 计算每个点的逆时针角度 (以负 Y 轴为起点)**
    def angle_counter_clockwise(p):
        dx, dy = p[0] - center_point[0], p[1] - center_point[1]
        angle = math.degrees(math.atan2(dx, -dy))  # 以负 Y 轴 (0°) 为基准
        return (angle + 360) % 360  # 归一化到 [0, 360)

    # **6. 按逆时针角度排序**
    sorted_points = sorted(remaining_points, key=angle_counter_clockwise)

    # **7. 初步分配：end → needle → start**
    end_point, needle_point, start_point = sorted_points

    # **8. 确保 start 在 end 顺时针方向**
    end_angle = angle_counter_clockwise(end_point)
    start_angle = angle_counter_clockwise(start_point)

    if (start_angle - end_angle) % 360 < 180:  # 如果 start 比 end 逆时针更近，交换它们
        start_point, end_point = end_point, start_point

    # **9. 转换回图像坐标**
    def revert_y(p):
        return (p[0], img_height - p[1])

    return revert_y(center_point), revert_y(start_point), revert_y(needle_point), revert_y(end_point)



# OCR 识别数字的函数
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

# 计算角度
def transform_coordinates(point, img_height):
    return (point[0], img_height - point[1])

def clockwise_angle(A, B, O):
    ax, ay = A[0] - O[0], A[1] - O[1]
    bx, by = B[0] - O[0], B[1] - O[1]
    angle_A = math.degrees(math.atan2(ay, ax))
    angle_B = math.degrees(math.atan2(by, bx))
    angle = (angle_A - angle_B) % 360
    return angle

# 逐帧处理视频
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    orig_h, orig_w = frame.shape[:2]

    # 运行目标检测
    results = model.predict(frame, imgsz=640, conf=0.5)
    result = results[0]

    # 存储检测框
    saved_boxes = []
    for detection in result.boxes.data.cpu().numpy():
        l, t, r, b, conf, cls_id = detection
        cls_id = int(cls_id)
        if cls_id in [1, 2]:
            saved_boxes.append([l, t, r, b, conf, cls_id])
        cv2.rectangle(frame, (int(l), int(t)), (int(r), int(b)), (0, 0, 255), 2)

    # **如果当前帧检测失败，使用上一帧的检测框**
    if len(saved_boxes) < 2 and prev_saved_boxes is not None:
        saved_boxes = prev_saved_boxes.copy()
    prev_saved_boxes = saved_boxes.copy()

    # **OCR 读取量程数值**
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

    # **获取关键点**
    keypoints_array = []
    keypoints = result.keypoints.cpu().numpy().data if result.keypoints is not None else []
    for keypoint_set in keypoints:
        for keypoint in keypoint_set:
            x, y, conf = keypoint
            if conf > 0.5:
                keypoints_array.append((x, y))
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    # **如果当前帧的关键点不足 4 个，使用上一帧的关键点**
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

    # **如果关键点仍不足 4 个，跳过当前帧**
    if len(keypoints_array) < 4:
        print(f"帧 {frame_idx} 关键点检测失败，跳过")
        out.write(frame)
        frame_idx += 1
        continue

    cv2.circle(frame, (int(center_point[0]), int(center_point[1])), 5, (255, 255, 0), -1)
    cv2.circle(frame, (int(start_point[0]), int(start_point[1])), 5, (255, 0, 255), -1)
    cv2.circle(frame, (int(needle_point[0]), int(needle_point[1])), 5, (0, 255, 255), -1)
    cv2.circle(frame, (int(end_point[0]), int(end_point[1])), 5, (255, 0, 0), -1)
    # **计算指针角度**
    center_point = transform_coordinates(center_point, orig_h)
    needle_point = transform_coordinates(needle_point, orig_h)
    start_point = transform_coordinates(start_point, orig_h)
    end_point = transform_coordinates(end_point, orig_h)

    angle_total = clockwise_angle(start_point, end_point, center_point)
    angle_pointer = clockwise_angle(start_point, needle_point, center_point)

    # **计算当前读数**
    current_reading = start_value + (angle_pointer / angle_total) * (end_value - start_value)

    # **绘制结果**
    reading_text = f"Reading: {current_reading:.1f}"
    cv2.putText(frame, reading_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # **写入视频**
    out.write(frame)
    frame_idx += 1

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()
print("视频处理完成，保存为:", output_path)
