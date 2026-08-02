import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. 重新定义或者导入与训练时相同的 LeNet-5 网络结构
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 针对 32x32 灰度单通道输入 (in_channels=1)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 2)  # 二分类: 0->crack, 1->intact

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        return x


def evaluate_on_test_set(model_path, test_dir):
    # 检查 GPU 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> 当前测试正在使用设备: {device}")

    # 2. 测试集数据预处理（切记：测试集不要做数据增强，只做标准化）
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"--> 成功加载测试集目录: {test_dir}")
    print(f"--> 类别映射关系: {test_dataset.class_to_idx}")

    # 3. 加载训练好的权重
    model = LeNet5().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型文件: {model_path}，请检查路径。")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. 统计结果与混淆矩阵 [ [TP, FP], [FN, TN] ]
    # index 0: crack, index 1: intact
    confusion_matrix = [[0, 0], [0, 0]]
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for t, p in zip(labels.view(-1), predicted.view(-1)):
                confusion_matrix[t.long()][p.long()] += 1

    accuracy = 100 * correct / total

    # 5. 打印专业评测结果
    print("\n==============================================")
    print("            定量评测报告 (Test Evaluation)            ")
    print("==============================================")
    print(f"★ 测试样本总数: {total} 张")
    print(f"★ 总体预测准确率 (Accuracy): {accuracy:.2f}%")
    print("----------------------------------------------")
    print("混淆矩阵 (Confusion Matrix):")
    print(f"  [实际: 开裂 crack ] -> 正确识别: {confusion_matrix[0][0]:3d} | 误报为完好: {confusion_matrix[0][1]:3d}")
    print(f"  [实际: 完好 intact] -> 误报为开裂: {confusion_matrix[1][0]:3d} | 正确识别: {confusion_matrix[1][1]:3d}")
    print("==============================================\n")


if __name__ == "__main__":
    # 执行测试
    evaluate_on_test_set(
        model_path="./models/lenet_qds_best.pth",
        test_dir="./dataset_clean/test"
    )