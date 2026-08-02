import os
import random


def random_prune_folder(folder_path, keep_num=1500):
    """
    对单个文件夹下的图片进行随机删减，保留指定的 keep_num 张
    自动过滤 Windows 后缀大小写导致的重复路径问题
    """
    if not os.path.exists(folder_path):
        print(f"【跳过】路径不存在: {folder_path}")
        return

    # 1. 安全获取目录下所有真实有效的图片路径（通过 set 彻底杜绝 Windows 重复匹配）
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = []
    for file_name in os.listdir(folder_path):
        # 统一转小写进行匹配，确保没有重复项
        if file_name.lower().endswith(valid_extensions):
            image_files.append(os.path.join(folder_path, file_name))

    total_files = len(image_files)
    print(f"\n检查目录: {folder_path}")
    print(f" -> 真实有效图片总数: {total_files} 张")

    # 2. 判断是否需要删除
    if total_files <= keep_num:
        print(f" -> 当前图片数量 ({total_files}) 未超过目标数量 ({keep_num})，无需删除。")
        return

    # 3. 随机选出需要被删除的图片列表
    delete_num = total_files - keep_num
    files_to_delete = random.sample(image_files, delete_num)

    # 4. 执行删除操作
    print(f" -> 正在随机删除 {delete_num} 张多余图片...")
    for file_path in files_to_delete:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"  ❌ 删除失败: {file_path} | 原因: {e}")

    # 5. 二次终检
    remaining_files = len([
        f for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ])
    print(f" ✅ 处理完成！当前文件夹精准剩余: {remaining_files} 张")


def prune_dataset_by_class(root_dir, keep_per_class=1500):
    print("==============================================")
    print("           随机裁剪数据集 (Prune Dataset)      ")
    print("==============================================")
    print(f"目标根目录: {root_dir}")
    print(f"每类保留上限: {keep_per_class} 张")

    subdirs = [
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    if not subdirs:
        random_prune_folder(root_dir, keep_num=keep_per_class)
    else:
        for label in subdirs:
            class_folder = os.path.join(root_dir, label)
            random_prune_folder(class_folder, keep_num=keep_per_class)

    print("\n🎉 全部分类处理完毕！")


if __name__ == "__main__":
    TARGET_DIR = "./data/train"
    KEEP_NUM = 1500

    prune_dataset_by_class(root_dir=TARGET_DIR, keep_per_class=KEEP_NUM)