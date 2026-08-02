import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import glob


def slice_large_image_color(img_path, output_dir, label_name, patch_size=32, stride=16):
    """
    将单张大型彩色图切为 patch_size x patch_size 的彩色小块
    保留原图完整的 3 通道 RGB/BGR 色彩信息
    """
    # 1. 默认读取为标准 3 通道彩色图 (B, G, R)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"  ❌ 读取失败，跳过文件: {img_path}")
        return 0

    # 获取高、宽、通道数
    h, w, c = img.shape
    print(f"  --> 正在切片: {os.path.basename(img_path)} | 尺寸: {w}x{h} | 通道数: {c}")

    # 自动创建对应分类的目标文件夹
    save_folder = os.path.join(output_dir, label_name)
    os.makedirs(save_folder, exist_ok=True)

    count = 0
    img_name_prefix = os.path.splitext(os.path.basename(img_path))[0]

    # 2. 滑动窗口切块 (Sliding Window Patching)
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            # 切下 patch_size x patch_size x 3 的彩色彩素块
            patch = img[y:y + patch_size, x:x + patch_size]

            # 命名格式: 分类名_大图名字_y坐标_x坐标.png
            patch_filename = f"{label_name}_{img_name_prefix}_y{y}_x{x}.png"
            patch_filepath = os.path.join(save_folder, patch_filename)

            # 3. 保存为彩色 PNG 图
            cv2.imwrite(patch_filepath, patch)
            count += 1

    print(f"      ✅ 完成！从小图中切出 {count} 张彩色块至 -> {save_folder}")
    return count


def batch_process_color_dataset(source_root, target_root, patch_size=32, stride=16):
    """
    批量处理整个 source_root 目录下的各分类分类大图
    默认目录结构预期：
      source_root/
         ├── crack/     <-- 里面放带裂纹的高清彩色大图
         └── intact/    <-- 里面放完好无损的高清彩色大图
    """
    print("==============================================")
    print("        彩色高清微观图像切块系统 (RGB Patching)")
    print("==============================================")

    # 支持的图片后缀
    valid_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.PNG", "*.JPG")

    categories = ["crack", "intact"]
    total_generated = 0

    for category in categories:
        cat_dir = os.path.join(source_root, category)
        if not os.path.exists(cat_dir):
            print(f"【跳过】未见类别目录: {cat_dir}，如果存在单个文件夹请核对名字。")
            continue

        print(f"\n📁 正在处理类别分类: [{category}] ...")

        # 匹配该分类下所有图片
        img_files = []
        for ext in valid_exts:
            img_files.extend(glob.glob(os.path.join(cat_dir, ext)))

        if not img_files:
            print(f"  ⚠️ 分类 [{category}] 下未发现图片！")
            continue

        for img_path in img_files:
            num_patches = slice_large_image_color(
                img_path=img_path,
                output_dir=target_root,
                label_name=category,
                patch_size=patch_size,
                stride=stride
            )
            total_generated += num_patches

    print(f"\n🎉 批量切块全流程结束！一共为你生成了 {total_generated} 张彩色小块。")
    print(f"📁 最终输出路径: {target_root}")


if __name__ == "__main__":
    # ====================================================
    # 路径配置区域
    # ====================================================
    # 1. 存放高清大图的源目录 (请确保里面有 crack 和 intact 两个子文件夹)
    SOURCE_DIR = "./raw_large_images"

    # 2. 生成后保存小切块的目标目录 (默认输出到 ./data/train)
    TARGET_DIR = "./data/train"

    # 3. 运行批量处理
    batch_process_color_dataset(
        source_root=SOURCE_DIR,
        target_root=TARGET_DIR,
        patch_size=32,  # 小块边长：32x32 像素
        stride=16  # 滑动步长：16 像素 (50% 重叠采样，增大样本量)
    )