import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


# ==========================================
# 1. 构建 ResNet-18 迁移学习二分类模型
# ==========================================
def get_resnet18_model(device):
    """
    自动下载官方 ImageNet 预训练权重，并将其微调为 2 分类模型
    """
    print("--> 正在加载 ResNet-18 官方预训练权重...")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 获取最后一层全连接层原来的输入维度 (ResNet-18 是 512)
    in_features = model.fc.in_features

    # 将 ImageNet 的 1000 分类替换为我们的 2 分类 (0: crack, 1: intact)
    model.fc = nn.Linear(in_features, 2)

    return model.to(device)


# ==========================================
# 2. 彩色图数据增强与加载 (保留 RGB 3通道)
# ==========================================
def get_dataloader(data_dir, batch_size=16):
    """
    针对彩色图像的增强管道：
    去除了 Grayscale，直接利用 3 通道及 ImageNet 标准统计量归一化
    """
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 统一切块大小
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomVerticalFlip(p=0.5),  # 随机垂直翻转
        transforms.RandomRotation(90),  # 随机旋转（对裂纹方向极其重要）
        transforms.ToTensor(),  # 自动转为 [3, H, W] 浮点张量
        # 使用官方预训练模型对应的 ImageNet 3 通道均值与方差
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"❌ 找不到数据目录: {data_dir}，请确保路径正确！")

    dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader, dataset.class_to_idx


# ==========================================
# 3. 训练主流程
# ==========================================
def train_model():
    # 检测计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==============================================")
    print(f"   启动 ResNet-18 彩色微裂纹分类训练")
    print(f"   计算设备: {device}")
    print(f"==============================================")

    # 加载数据集 (确认你保存小图的根目录为 ./data/train)
    train_dir = "./data/train"
    train_loader, class_idx = get_dataloader(train_dir, batch_size=16)
    print(f"✅ 成功加载数据集！标签对应索引: {class_idx}")

    # 初始化模型、损失函数与优化器
    model = get_resnet18_model(device)
    criterion = nn.CrossEntropyLoss()

    # 迁移学习技巧：学习率稍小一点(0.0005)，避免破坏预训练特征
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

    epochs = 20
    best_acc = 0.0
    save_path = "resnet18_crack_model.pth"

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)

        print(f"Epoch [{epoch + 1:02d}/{epochs}] - Loss: {avg_loss:.4f} - 准确率: {acc:.2f}%")

        # 始终保存训练过程中准确率最高的模型权重
        if acc >= best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)

    print(f"\n🎉 训练全部完成！最高准确率达到: {best_acc:.2f}%")
    print(f"✅ 最佳模型权重已保存至当前目录: {save_path}")


if __name__ == "__main__":
    train_model()