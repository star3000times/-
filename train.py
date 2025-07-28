from ultralytics import YOLO
import os
import torch

# 在Windows系统中启用多进程支持
if __name__ == '__main__':
    # 检查CUDA是否可用，自动选择设备
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU' if device == 0 else 'CPU'}")

    # 加载已训练好的模型
    model = YOLO(r"model\pose\train\weights\best.pt")

    # 检查数据集完整性（修复编码问题）
    def check_dataset(data_yaml_path):
        import yaml
        # 明确指定以UTF-8编码打开文件，解决中文/特殊字符解码问题
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 检查训练图像和标签路径是否存在
        for split in ['train', 'val']:
            if split not in data:
                print(f"警告: YAML文件中缺少 {split} 数据集路径")
                continue
                
            img_dir = data[split]
            if not os.path.exists(img_dir):
                print(f"错误: {split} 图像目录不存在 - {img_dir}")
                continue
                
            # 检查标签目录
            label_dir = os.path.join(os.path.dirname(img_dir), 'labels')
            if not os.path.exists(label_dir):
                print(f"警告: 标签目录不存在 - {label_dir}")
                continue
                
            # 简单检查部分图像的标签是否存在
            img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not img_files:
                print(f"警告: {split} 目录中没有图像文件 - {img_dir}")
                continue
                
            # 随机检查10个图像文件（避免大量文件时检查过慢）
            for img_file in img_files[:10]:
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(label_dir, label_file)
                if not os.path.exists(label_path):
                    print(f"警告: {split} 图像 {img_file} 缺少标签文件 {label_file}")
                elif os.path.getsize(label_path) == 0:
                    print(f"警告: {split} 图像 {img_file} 的标签文件为空")
        
        print("数据集检查完成")

    # 检查数据集
    check_dataset("finetune.yaml")

    # 在自定义数据集上继续训练（微调）
    results = model.train(
        data="finetune.yaml",
        epochs=50,
        imgsz=640,
        batch=4,  # 适应6GB GPU显存的批次大小
        lr0=0.0001,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        device=device,
        workers=2,  # 减少工作进程数
        project="runs/pose",
        name="finetune",
        pretrained=True,
        freeze=10,
        patience=10,
        save_period=10,
        verbose=True,
        cache=True,  # 缓存数据减少内存占用
        half=device != 'cpu',  # GPU上使用半精度训练
    )

    # 训练完成后进行验证
    metrics = model.val()
    