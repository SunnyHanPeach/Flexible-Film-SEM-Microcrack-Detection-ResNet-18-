import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from PIL import Image

# 1. 消除 Windows 环境变量 DLL 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# =========================================================
# 2. 搭建与训练阶段相吻合的 ResNet-18 模型结构
# =========================================================
def get_resnet18_model(device, num_classes=2):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(device)


# =========================================================
# 3. 核心：稠密概率滑动窗口 + 高斯平滑热力图生成器
# =========================================================
def generate_dense_heatmap(model, image_path, output_path,
                           patch_size=32, stride=8, batch_size=64,
                           gaussian_kernel=31, crack_class_idx=0):
    """
    通过滑动窗口密采各区域开裂概率，并用高斯滤波平滑生成工业级热力图
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # 1. 读取原生大图 (BGR -> RGB)
    original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(f"❌ 无法读取图像: {image_path}")

    h, w, _ = original_bgr.shape
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # 2. 构建数据标准化预处理管道
    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 初始化概率矩阵与计数矩阵（用于处理窗口重叠区域的求均值）
    prob_map = np.zeros((h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    # 4. 收集所有扫描坐标并使用 Batch 批量推理，大幅提升速度
    coords = []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            coords.append((x, y))

    print(f"\n--> 正在进行稠密滑动概率扫描: {os.path.basename(image_path)}")
    print(f"    图像尺寸: {w}x{h} | 采样窗口: {patch_size}x{patch_size} | 步长: {stride}")
    print(f"    总采样微区数: {len(coords)} -> 采用 Batch size = {batch_size} 并行加速...")

    for i in range(0, len(coords), batch_size):
        batch_coords = coords[i:i + batch_size]
        batch_tensors = []

        for (x, y) in batch_coords:
            patch_rgb = original_rgb[y:y + patch_size, x:x + patch_size]
            patch_pil = Image.fromarray(patch_rgb)
            batch_tensors.append(transform(patch_pil))

        # 打包堆叠为 Batch Tensor [B, 3, 32, 32]
        input_batch = torch.stack(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            outputs = model(input_batch)
            probs = torch.softmax(outputs, dim=1)
            # 获取当前 Batch 内所有窗口的开裂缺陷 (crack) 预测概率
            crack_probs = probs[:, crack_class_idx].cpu().numpy()

        # 将概率累加到对应坐标系空间中
        for idx, (x, y) in enumerate(batch_coords):
            prob_map[y:y + patch_size, x:x + patch_size] += crack_probs[idx]
            count_map[y:y + patch_size, x:x + patch_size] += 1.0

    # 5. 求出所有重叠区域的真实概率均值
    count_map = np.maximum(count_map, 1.0)
    avg_prob_map = prob_map / count_map

    # 6. 【核心美化】二维高斯空间平滑 (Gaussian Smoothing)
    # 彻底抹平微区边界方块感，生成连续自然的等高线应力热场
    print("--> 正在执行二维高斯空间平滑处理...")
    smoothed_prob_map = cv2.GaussianBlur(avg_prob_map, (gaussian_kernel, gaussian_kernel), 0)

    # 7. 动态对比度归一化至 [0, 1] 区间以便伪彩渲染
    min_val, max_val = smoothed_prob_map.min(), smoothed_prob_map.max()
    if max_val > min_val:
        norm_map = (smoothed_prob_map - min_val) / (max_val - min_val)
    else:
        norm_map = np.zeros_like(smoothed_prob_map)

    # 8. 上色与多视图融合
    heatmap_uint8 = np.uint8(255 * norm_map)
    colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 按照 55% 原图 + 45% 热场的黄金叠加比合成
    superimposed_img = cv2.addWeighted(original_bgr, 0.55, colored_heatmap, 0.45, 0)

    # 9. 拼接对照大图 (左: 原始 SEM 形貌 | 中: 连续概率热场 | 右: 叠加定位诊断图)
    combined_result = np.hstack([original_bgr, colored_heatmap, superimposed_img])

    # 水印抬头
    title = f"Dense Probability Heatmap (ResNet-18) | Max Crack Prob: {max_val:.2%}"
    cv2.putText(combined_result, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(output_path, combined_result)
    print(f"✅ 高清连续概率热力图已完美生成！保存路径: {output_path}")

    return norm_map


if __name__ == "__main__":
    MODEL_FILE = "./model/resnet18_crack_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化并载入训练好的 ResNet-18 权重
    model = get_resnet18_model(device=device)
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"❌ 找不到模型权重: {MODEL_FILE}，请先执行 step2_train_resnet18.py。")
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))

    print("==========================================================")
    print("      柔性薄膜微观应力损伤 - 稠密概率分布热图生成系统      ")
    print("==========================================================")

    # 任务 1：测试带细微发丝纹的开裂样本 (高概率高温带将精确沿着纹路展开)
    generate_dense_heatmap(
        model=model,
        image_path="./raw_large_images/test_hairline_crack.png",
        output_path="./result_dense_heatmap_crack.png",
        patch_size=32,
        stride=8,  # 步长越小，等高线越精细，推荐 4 或 8
        gaussian_kernel=31  # 高斯核一定要是奇数，推荐 31 或 45
    )

    # 任务 2：测试完好平整对照组 (全图应当沉寂在冷蓝色海洋中)
    generate_dense_heatmap(
        model=model,
        image_path="./raw_large_images/test_intact_clean.png",
        output_path="./result_dense_heatmap_intact.png",
        patch_size=32,
        stride=8,
        gaussian_kernel=31
    )