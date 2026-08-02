import os
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ==========================================
# 1. 消除 Windows DLL 冲突 & 搭建与 step2 一致的 ResNet-18
# ==========================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_resnet18_model(device):
    """
    搭建与训练脚本完全同步的 ResNet-18 二分类网络结构
    """
    model = models.resnet18(weights=None)  # 仅加载网络结构，权重用我们自己训好的
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)  # 2 分类: 0-crack, 1-intact
    return model.to(device)


# ==========================================
# 2. 全图滑动扫描检测主逻辑 (支持 RGB 彩色)
# ==========================================
def detect_cracks_resnet18(model_path, image_path, output_path, patch_size=32, stride=16, conf_threshold=0.70):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 自动校验模型是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ 找不到模型权重: {model_path}，请确认 step2 训练完毕且路径正确！")

    model = get_resnet18_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 校验测试图片是否存在
    if not os.path.exists(image_path):
        print(f"【跳过】找不到待测图: {image_path}，请检查路径。")
        return 0, 0.0

    # 1. 以彩色 3 通道形式读取原始大图 (BGR)
    original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    h, w, _ = original_img.shape

    # 2. 严格对齐 step2 训练时的彩色转换管道 (必须保持一模一样)
    transform = transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    crack_boxes = []
    total_patches = 0

    print(f"\n--> [ResNet-18] 正在高精度扫描: {os.path.basename(image_path)} (尺寸: {w}x{h}) ...")

    # 3. 滑动窗口抓取切块
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            total_patches += 1
            patch_bgr = original_img[y:y + patch_size, x:x + patch_size]

            # OpenCV (BGR) 转 PIL (RGB) 以供 torchvision 能够正确处理颜色
            patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
            pil_patch = Image.fromarray(patch_rgb)
            tensor_patch = transform(pil_patch).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(tensor_patch)
                probs = torch.softmax(outputs, dim=1)

                # 注意：假设 DataLoader 自动映射字典为 {'crack': 0, 'intact': 1}
                # 如果发现全部框错，把 0 改为 1 即可 (probs[0, 1].item())
                crack_prob = probs[0, 0].item()

                if crack_prob > conf_threshold:
                    crack_boxes.append((x, y, x + patch_size, y + patch_size))

    # 4. 在原图上标记红色警示框 (BGR中红色的代号为 0, 0, 255)
    result_img = original_img.copy()
    for (x1, y1, x2, y2) in crack_boxes:
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

    damage_ratio = (len(crack_boxes) / total_patches) * 100 if total_patches > 0 else 0.0

    # 顶部添加标注水印
    info_text = f"ResNet18 | Damage: {damage_ratio:.2f}% | Conf > {conf_threshold}"
    cv2.putText(result_img, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(output_path, result_img)
    print(f"  ✅ 扫描完毕！遍历微区: {total_patches} 个 | 锁定了 {len(crack_boxes)} 个微裂纹框")
    print(f"  ✅ 物理损伤率估算: {damage_ratio:.2f}% -> 效果图已生成: {output_path}")

    return len(crack_boxes), damage_ratio


if __name__ == "__main__":
    # ==========================================================
    # 参数配置区：优先使用你刚刚训练好的 ResNet-18 权重
    # ==========================================================
    MODEL_FILE = "./resnet18_crack_model.pth"

    # 兼容提醒：如果找不到 resnet18 权重，提醒用户去跑 step2
    if not os.path.exists(MODEL_FILE):
        print(f"⚠️ 未检测到 {MODEL_FILE}，请确认你已经运行过 step2_train_resnet18.py！")

    print("==========================================================")
    print("      ResNet-18 柔性薄膜 SEM 显微形貌损伤智能盲测系统        ")
    print("==========================================================")

    # 任务 1：高精度扫描“细微发丝微裂纹”样本 (QDs-OA after bending)
    detect_cracks_resnet18(
        model_path=MODEL_FILE,
        image_path="./raw_large_images/test_hairline_crack.png",
        output_path="./result_resnet18_hairline_crack.png",
        conf_threshold=0.70
    )

    # 任务 2：高精度扫描“平整完好对照组”样本 (QDs-DDTC after bending)
    detect_cracks_resnet18(
        model_path=MODEL_FILE,
        image_path="./raw_large_images/test_intact_clean.png",
        output_path="./result_resnet18_intact_control.png",
        conf_threshold=0.70
    )

    print("\n🎉 自动化双盲检测全部结束！快打开根目录查看生成的 result_resnet18_*.png 对比图！")