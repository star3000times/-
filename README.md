### 使用yolov8和opencv对指示灯和仪器仪表进行视觉识别  
打算将多个模块最后融合在一起，并最终部署在板卡上面。


#### 进入code文件夹，打开终端，启动conda虚拟环境，后安装提示需要的库


#### 仪表指针识别角度和位置：
##### 启动程序：  
`python test_bbip.py MP42.mp4`  

##### 输出说明：  
输出视频为：`processed_output.mp4`，视频弹窗可按Q退出。


#### 指示灯识别：
##### 本地摄像头识别：  
虚拟环境下运行 `test4LED.py`  

##### 识别本地视频：  
`python test4LED.py --source C:/Users/UserX/Desktop/code/MP44.mp4`  
`python main.py --source C:/Users/UserX/Desktop/code/MP44.mp4`
### 板卡启动文档：
- `code --no-sandbox` # 启动VSCODE  
- `cd code` # 进入代码文件夹  
- `source /home/jetson/miniconda3/bin/activate labelimg_env` # 启动虚拟环境  
- `python test_bbip.py MP41.mp4` # 启动仪表识别
