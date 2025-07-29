import cv2
import numpy as np
import paho.mqtt.client as mqtt
import json
import argparse
import os
import time
import logging
from paho.mqtt.client import CallbackAPIVersion

# 初始化日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# 默认MQTT配置
DEFAULT_MQTT_BROKER = "192.168.18.164"
DEFAULT_MQTT_PORT = 1883
DEFAULT_MQTT_TOPIC = "DCInspection/device/v6"
DEFAULT_MQTT_USER = "user1"
DEFAULT_MQTT_PASS = "axjIWLAMMXA.SAKFN5321ZASHV"

# 添加连接状态标识
mqtt_connected = False  # 连接成功标志

# 默认HSV颜色范围（可根据摄像头环境调整）
DEFAULT_COLOR_RANGES = {
    'red':    ((0, 120, 120), (10, 255, 255)),
    'yellow': ((20, 120, 120), (30, 255, 255)),
    'green':  ((50, 120, 120), (70, 255, 255)),
    'blue':   ((100, 120, 120), (130, 255, 255)),
}
DRAW_COLORS = {
    'red':    (0, 0, 255),
    'yellow': (0, 255, 255),
    'green':  (0, 255, 0),
    'blue':   (255, 0, 0),
}

# MQTT回调函数：连接成功时触发
def on_connect(client, userdata, flags, rc, properties):
    global mqtt_connected
    if rc == 0:
        mqtt_connected = True
        logging.info(f"✅ MQTT连接成功！返回码: {rc}")
        # 可选：订阅主题（如果需要接收消息）
        # client.subscribe(MQTT_TOPIC)
    else:
        mqtt_connected = False
        logging.error(f"❌ MQTT连接失败！返回码: {rc}，错误原因: {mqtt.connack_string(rc)}")

# MQTT回调函数：消息发布成功时触发
def on_publish(client, userdata, mid, reason_code, properties):
    logging.info(f"📤 消息发布成功 (消息ID: {mid})")

# MQTT回调函数：连接断开时触发（V2版本正确参数）
def on_disconnect(client, userdata, rc, properties):
    global mqtt_connected
    mqtt_connected = False
    if rc != 0:
        logging.warning(f"⚠️ MQTT意外断开连接！返回码: {rc}")
    else:
        logging.info("🔌 MQTT已正常断开连接")

def detect_and_draw(frame, color_ranges, min_area):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detections = []  # 存储 (color, bbox)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

    for color, (low, high) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(low), np.array(high))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > min_area:
                x,y,w,h = cv2.boundingRect(c)
                detections.append((color, (x,y,w,h)))
                cv2.rectangle(frame, (x,y), (x+w, y+h), DRAW_COLORS[color], 2)
                cv2.putText(frame, color, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, DRAW_COLORS[color], 2)
    return detections

def main():
    parser = argparse.ArgumentParser(description="LED Detection with MQTT")
    parser.add_argument('--source', default=0, help="视频源（文件路径、RTSP地址或摄像头ID，默认0）")
    parser.add_argument('--min-area', type=int, default=100, help="最小轮廓面积阈值（默认100）")
    parser.add_argument('--mqtt-broker', default=DEFAULT_MQTT_BROKER, help=f"MQTT服务器地址（默认{DEFAULT_MQTT_BROKER}）")
    parser.add_argument('--mqtt-port', type=int, default=DEFAULT_MQTT_PORT, help=f"MQTT端口（默认{DEFAULT_MQTT_PORT}）")
    parser.add_argument('--mqtt-topic', default=DEFAULT_MQTT_TOPIC, help=f"MQTT主题（默认{DEFAULT_MQTT_TOPIC}）")
    parser.add_argument('--hsv-red', nargs=6, type=int, default=[0,120,120,10,255,255], 
                       help="红色HSV范围（low_h low_s low_v high_h high_s high_v）")
    parser.add_argument('--hsv-yellow', nargs=6, type=int, default=[20,120,120,30,255,255], 
                       help="黄色HSV范围（low_h low_s low_v high_h high_s high_v）")
    parser.add_argument('--hsv-green', nargs=6, type=int, default=[50,120,120,70,255,255], 
                       help="绿色HSV范围（low_h low_s low_v high_h high_s high_v）")
    parser.add_argument('--hsv-blue', nargs=6, type=int, default=[100,120,120,130,255,255], 
                       help="蓝色HSV范围（low_h low_s low_v high_h high_s high_v）")
    args = parser.parse_args()

    # 配置颜色范围
    color_ranges = {
        'red': ((args.hsv_red[0], args.hsv_red[1], args.hsv_red[2]),
                (args.hsv_red[3], args.hsv_red[4], args.hsv_red[5])),
        'yellow': ((args.hsv_yellow[0], args.hsv_yellow[1], args.hsv_yellow[2]),
                  (args.hsv_yellow[3], args.hsv_yellow[4], args.hsv_yellow[5])),
        'green': ((args.hsv_green[0], args.hsv_green[1], args.hsv_green[2]),
                 (args.hsv_green[3], args.hsv_green[4], args.hsv_green[5])),
        'blue': ((args.hsv_blue[0], args.hsv_blue[1], args.hsv_blue[2]),
                (args.hsv_blue[3], args.hsv_blue[4], args.hsv_blue[5])),
    }

    # 建立MQTT连接
    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2)
    
    # 绑定回调函数
    client.on_connect = on_connect
    client.on_publish = on_publish
    client.on_disconnect = on_disconnect
    
    # 设置用户名密码
    client.username_pw_set(DEFAULT_MQTT_USER, DEFAULT_MQTT_PASS)
    
    # 连接服务器
    logging.info(f"🔍 正在连接MQTT服务器: {args.mqtt_broker}:{args.mqtt_port}...")
    try:
        client.connect(args.mqtt_broker, args.mqtt_port, keepalive=60)
    except Exception as e:
        logging.error(f"❌ MQTT连接失败: {str(e)}")
        return
    
    # 启动网络循环线程
    client.loop_start()
    
    # 等待连接结果（最多等待5秒）
    start_time = time.time()
    while not mqtt_connected and time.time() - start_time < 5:
        time.sleep(0.1)
    
    # 检查连接状态
    if not mqtt_connected:
        logging.error("❌ 无法建立MQTT连接，程序退出")
        client.loop_stop()
        return

    # 处理视频源
    source = args.source
    # 判断是文件还是流地址
    if isinstance(source, str) and os.path.isfile(source):
        logging.info(f"📂 打开本地视频文件: {source}")
    else:
        # 尝试将字符串转换为整数（摄像头ID）
        try:
            source = int(source)
            logging.info(f"📹 打开本地摄像头: {source}")
        except ValueError:
            logging.info(f"🌐 打开网络视频流: {source}")

    # 打开视频源
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("❌ 无法打开视频，请检查路径或流地址是否正确")
        client.loop_stop()
        client.disconnect()
        return

    # 窗口初始化（只执行一次）
    cv2.namedWindow("LED Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("LED Detection", 1280, 720)

    # 主循环
    reconnect_attempts = 0
    max_reconnect_attempts = 3  # 最大重连次数
    reconnect_delay = 2  # 重连间隔（秒）

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                logging.warning("⚠️ 视频读取失败，尝试重新读取...")
                time.sleep(0.5)
                continue
        except Exception as e:
            logging.error(f"❌ 视频读取异常: {str(e)}")
            time.sleep(1)
            continue

        # 检查MQTT连接状态并尝试重连
        if not mqtt_connected:
            logging.warning(f"⚠️ MQTT连接已断开，尝试第 {reconnect_attempts+1}/{max_reconnect_attempts} 次重连...")
            try:
                client.reconnect()
                start_time = time.time()
                # 等待重连
                while not mqtt_connected and time.time() - start_time < 5:
                    time.sleep(0.1)
                if mqtt_connected:
                    reconnect_attempts = 0  # 重置计数
                    logging.info("✅ MQTT重连成功")
                else:
                    reconnect_attempts += 1
                    if reconnect_attempts >= max_reconnect_attempts:
                        logging.error("❌ 达到最大重连次数，程序退出")
                        break
                    time.sleep(reconnect_delay)  # 间隔后再试
            except Exception as e:
                logging.error(f"❌ 重连异常: {str(e)}")
                reconnect_attempts += 1
                if reconnect_attempts >= max_reconnect_attempts:
                    logging.error("❌ 达到最大重连次数，程序退出")
                    break
                time.sleep(reconnect_delay)

        # 检测并绘制
        detections = detect_and_draw(frame, color_ranges, args.min_area)

        # 准备发送数据
        counts = {color: 0 for color in color_ranges}
        for color, _ in detections:
            counts[color] += 1

        payload = json.dumps(counts)
        
        # 发布消息（带重试机制）
        if mqtt_connected:
            max_publish_attempts = 2
            publish_attempts = 0
            published = False

            while publish_attempts < max_publish_attempts and not published:
                result = client.publish(args.mqtt_topic, payload)
                try:
                    result.wait_for_publish(timeout=1.0)
                    if result.is_published():
                        logging.info(f"📊 发送数据: {payload}")
                        published = True
                    else:
                        publish_attempts += 1
                        logging.warning(f"⚠️ 数据发送失败，重试第 {publish_attempts} 次: {payload}")
                except Exception as e:
                    publish_attempts += 1
                    logging.error(f"⚠️ 发布异常，重试第 {publish_attempts} 次: {str(e)}")

            if not published:
                logging.error(f"❌ 达到最大发布重试次数: {payload}")

        # 显示帧
        cv2.imshow("LED Detection", frame)
        
        # 检查退出
        if cv2.waitKey(1) & 0xFF == 27:  # ESC键退出
            logging.info("🔍 用户请求退出程序")
            break

    # 资源清理
    cap.release()
    cv2.destroyAllWindows()
    client.loop_stop()  # 停止网络循环
    client.disconnect()
    logging.info("📌 程序已退出")

if __name__ == '__main__':
    main()
