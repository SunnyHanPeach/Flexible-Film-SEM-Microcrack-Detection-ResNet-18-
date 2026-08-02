# Flexible Thin-Film SEM Stress Crack Automated Detection & XAI Characterization
### 柔性薄膜 SEM 应力裂纹自动化双盲检测与高斯概率热力学表征系统

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776ab.svg)](https://www.python.org/)
[![AI for Science](https://img.shields.io/badge/AI4S-Materials%20Science-0052cc.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **AI for Materials Science (AI4S) 跨学科实践：**  
> 将工程断裂力学、SEM 表面形貌表征与深度学习可解释性（XAI）相结合，为柔性半导体薄膜（如 QDs / 钙钛矿薄膜）在机械弯折后的应力损伤与疲劳失效，提供**全图无损定量检测**与**高分辨率应力分布热场表征**。

---

## 📖 项目简介 (Overview)

柔性薄膜在经历了高强度弯折测试后，极易在表面生成极其隐蔽的发丝状微裂纹（Hairline Micro-cracks）。传统材料学研究多依赖**人工选图、肉眼定性观察**，存在严重的**主观选取偏见**且难以进行定量统计。

本项目设计了一套从**大图色彩采样、数据平衡、ResNet-18 深度特征提取**，到**盲测定量计算**与**高斯平滑稠密概率密度映射（Dense Probability Heatmap）**的端到端自动化表征体系。不仅解决了早期应力微缝难以精准检出的难题，更实现对完好对照组（Negative Control）的**极高统计特异性（零误报早筛）**。

---

## ✨ 核心亮点 (Key Highlights)

* **🔬 严谨的物理与力学机理印证**
  * 将原子力显微镜（AFM）测得的**杨氏模量差异**与 SEM 应力开裂行为挂钩：验证了高模量脆性表面（`QDs-OA`）在拉伸应力下的疲劳开裂网络，以及配体置换后均匀弹性表面（`QDs-DDTC`）的优异抗折特性。
* **⚖️ 高灵敏度 vs. 高特异性双盲对照（Dual-Sample Blind Test）**
  * **开裂样本 (`QDs-OA`)**：微区损伤响应率达 **88.13%**，极佳地捕捉了应力集中区域的网状发丝裂纹。
  * **完好对照组 (`QDs-DDTC`)**：测试误报率低至 **0.86%**（特异性达 **99.14%**），彻底排除了模型对 SEM 颗粒背景的过度拟合。
* **🗺️ 创新性 XAI 稠密概率热图表征 (Dense Probability Heatmap mapping)**
  * 突破传统 CNN/Grad-CAM 最后一个卷积层缩放引起的“空间分辨率坍塌 ($1 \times 1$ 死色块)”技术瓶颈。
  * 采用 **滑动窗口高密度概率探针 + 二维高斯连续空间平滑**，把离散的二分类结果转化为**地理等高线般的连续拉伸应力损伤分布热力场**。

---

## 📂 项目结构 (Repository Structure)

```text
├── models/
│   └── resnet18_crack_model.pth           # 训练收敛的最佳 ResNet-18 权重文件
├── raw_large_images/                      # 原始 SEM 高清整图目录
│   ├── test_hairline_crack.png            # 弯折开裂测试样本 (QDs-OA)
│   └── test_intact_clean.png              # 平整完好对照组 (QDs-DDTC)
├── step1_patching_batch.py                # 自动化 RGB 滑动切块脚本 (32x32, 50% 重叠步长)
├── prune_dataset.py                       # 数据集平衡脚本 (防止类别严重不平衡导致的规则误读)
├── step2_train_resnet18.py                # ResNet-18 迁移学习与标准三通道图像训练
├── step4_visual_demo_resnet18.py          # 全图盲测红框目标定位 & 物理损伤率计算
├── step5_sliding_heatmap.py               # 核心亮点：高分辨高斯平滑稠密概率热场生成器
└── README.md
