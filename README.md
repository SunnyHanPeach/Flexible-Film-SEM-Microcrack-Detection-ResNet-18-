# 🔬 Flexible Film SEM Microcrack Detection (ResNet-18)
> **基于 ResNet-18 迁移学习的柔性发光薄膜微观裂纹自动检测与形貌损伤率定量评估系统**

---

## 📖 项目简介 (Project Overview)

在柔性半导体与薄膜材料（如 QLED / 量子点发光器件）的形变疲劳研究中，显微图（SEM/光学显微镜）下的**极细微裂纹（Microcracks）**识别往往依赖人工目视筛查，不仅耗时费力，且难以对弯折后的面损伤率进行定量表征。

本项目构建了一套**端到端、无需手动裁剪的“计算机视觉 + 材料损伤表征”自动化工作流**。通过改进的滑动窗口切样算法，把原始显微图片转化为 RGB 纹理贴片数据集，并基于 **ResNet-18** 迁移学习进行微观形貌缺陷分类，最终实现大图**无死角全扫检测、高亮画框预警以及整体损伤率（Damage Ratio）精准测算**。

---

## ✨ 核心功能亮点 (Key Features)

* **🖼️ 原生 RGB 色彩与纹理保留**：区别于传统灰度脱色，针对彩色光显或伪彩 SEM 图像，全流程保留 3 通道色彩与纹理特征，极大增强对渐变式拉伸细纹的抓取敏感度。
* **🧮 步长重叠扩增技术 (Overlapping Sliding Window)**：利用 `stride < patch_size` 机制在单张大图上执行重叠扫描，无需额外搜集海量电镜图，即可让微小样本量暴增 4~8 倍。
* **🚀 迁移学习与自动调优 (ResNet-18 Transfer Learning)**：引入 ImageNet 官方预训练权重与标准化处理，在仅 1500 张微区贴片上快速收敛至 **98%+** 准确率，有效防范过拟合。
* **📊 盲测大图红框扫描与物理量化 (Full-Image Auditing)**：通过阈值微调（Threshold Tuning），在全图盲测中做到“细微发丝纹精准检出”与“完好对照组 0 误报”，并自动在图首打印总面积损伤百分比。

---

## 📂 项目目录结构 (Repository Structure)

```text
QDs_Crack_Detection_Project/
 │
 ├── raw_large_images/                 # 原始没切块的高清微观显微大图
 │    ├── crack/                       # 带有开裂缺陷的大图样本
 │    ├── intact/                      # 完好无裂痕的大图样本
 │    ├── test_hairline_crack.png      # 【盲测用】细微发丝裂纹样本
 │    └── test_intact_clean.png        # 【盲测用】平整对照组样本
 │
 ├── data/
 │    └── train/                       # 自动化处理生成的分类小图像块 (32x32)
 │         ├── crack/
 │         └── intact/
 │
 ├── models/
 │    └── resnet18_crack_model.pth     # 训练完成的最佳模型权重文件
 │
 ├── result/
      └── 
 ├── step1_patching_batch.py           # [步骤1] 彩色显微大图批量重叠切块脚本
 ├── delete_folder.py                  # [辅助]  数据集按类别1:1精准平衡删减脚本
 ├── step2_train_resnet18.py           # [步骤2] ResNet-18 迁移学习与模型训练
 └── step4_visual_demo_resnet18.py     # [步骤3] 全图滑动窗口盲测、红框可视化与损伤计算
