from ultralytics import YOLO

# 加载已训练好的模型（替换为你的best.pt路径）
model = YOLO("model\pose\train\weights\best.pt")  # 代码中已训练的模型路径

# 在自定义数据集上继续训练（微调）
results = model.train(
    data="finetune.yaml",  # 你的自定义数据集配置文件路径
    epochs=50,                                # 训练轮数（可根据需求调整，如30-100）
    imgsz=640,                                # 输入图像尺寸（常见320, 480, 640, 800）
    batch=16,                                 # 批次大小（根据GPU显存调整）
    lr0=0.0001,                               # 初始学习率（微调时建议较小，如0.0001-0.001）
    momentum=0.937,                           # 动量因子
    weight_decay=0.0005,                      # 权重衰减（防止过拟合）
    warmup_epochs=3,                          # 热身训练轮数
    device=0,                                 # 使用的GPU设备（0表示第一张GPU，cpu表示使用CPU）
    workers=4,                                # 数据加载线程数
    project="runs/pose",                      # 训练结果保存的主目录
    name="finetune",                          # 本次训练的子目录名称（用于区分不同训练）
    pretrained=True,                          # 确保使用预训练权重（默认True）
    freeze=10,                                # 冻结前10层（可选，加速训练并防止过拟合）
    patience=10,                              # 早停耐心值（10轮无提升则停止）
    save_period=10,                           # 每10轮保存一次模型
    verbose=True                               # 显示训练详细日志
)

# 训练完成后可进行验证
metrics = model.val()  # 验证模型在验证集上的性能
