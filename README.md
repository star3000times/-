### 使用yolov8和opencv对指示灯和仪器仪表进行视觉识别  
预期目标是将多个模块最后融合在一起，并最终部署在板卡上面。
![识别效果](jetson板卡.png)
环境：-板卡型号:AGX Xavier。
-系统架构：Arm64架构
-JetPack版本:5.1.5
-CUDA 版本：11.4
-cuDNN 版本：8.6.0 
-TensorRT 版本：8.5.0.2
-Pytorch版本：1.14.0
现在已经实现了服务器修改识别的参数，并修改本地的json文件。实现了由服务器使用MQTT发送参数数据，本地摄像头（或者云端有相关IP的摄像头）根据参数调节识别图像，再使用MQTT协议上传识别数据。

1. 进入code文件夹，打开终端，执行以下操作：
   - 启动conda虚拟环境：  
     ```bash
     source /home/jetson/miniconda3/bin/activate labelimg_env
     ```
   - 安装提示需要的库


2. 程序启动与参数配置：
   - 运行`main.py`启动程序
   - 向服务器发送请求参数请求
   - 服务器收到消息并且发送json文件后，直接修改本地的`config.json`文件
   - 注意：json文件的参数不能缺失，否则会无法接收
   - `config.json`可调整的参数包括：
     - HSV和RGB的阈值
     - 识别的区域范围（根据摄像头的精度参数调节）
     - 摄像头的地址（如：
       - rtsp地址：`rtsp://admin:password@192.168.1.100:554/stream1`
       - 本地视频地址：`/media/jetson/rootfs/video_detect/MP44.mp4`
     ）


3. 仪表指针识别角度和位置：
   - 启动程序命令：  
     ```labelimg_env
     python test_bbip.py MP41.mp4
     ```
识别效果
![识别效果](指示灯.png)
