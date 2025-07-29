使用yolov8和opencv进行视觉识别
打算将多个模块融合在一起，并最终部署在板卡上面
进入code文件夹，打开终端，启动conda虚拟环境，后安装提示需要的库
仪表指针识别角度和位置：
启动程序python test_bbip.py MP42.mp4
输出视频为processed_output.mp4,视频弹窗可按Q退出
指示灯识别：
本地摄像头识别：
虚拟环境下运行test4LED.py
识别本地视频：
python test4LED.py --source C:/Users/UserX/Desktop/code/MP44.mp4
