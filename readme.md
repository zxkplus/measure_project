# Halcon1DMeasure - OpenCV 1D 测量工具

基于 OpenCV 实现的一维测量工具（简化版 Halcon `measure_pos` / `measure_pairs`），通过仿射变换与亚像素边缘检测实现高精度的几何测量。

## 项目概述

本项目提供了一个 Python 类 `Halcon1DMeasure`，旨在模拟机器视觉库 Halcon 中的 1D 测量逻辑。它能够处理任意角度的矩形 ROI（感兴趣区域），将其转换为正交坐标系进行灰度分析，并精确检测边缘位置和边缘对。

### 核心特性

*   **任意角度测量**：支持任意角度的 ROI，通过仿射变换自动校正。
*   **亚像素精度**：使用二次多项式拟合梯度极值点，实现亚像素级边缘定位。
*   **丰富的边缘筛选**：支持正边缘（暗->亮）、负边缘（亮->暗）及边缘对检测。
*   **可视化调试**：内置详细的调试视图，可实时显示灰度轮廓、梯度曲线及 ROI 标记。
*   **纯 Python/OpenCV**：无需安装 Halcon，仅需 `numpy` 和 `opencv-python`。

## 前置条件

*   Python 3.6+
*   NumPy
*   OpenCV (`opencv-python`)

## 安装与构建

本项目为单文件脚本，无需复杂的构建过程。

1.  **克隆或下载代码**：
    ```bash
    # 确保目录结构如下
    .
    ├── measure.py
    ├── test_1d_measure.py
    └── data/
        └── sample/ (存放测试图片)
    ```

2.  **安装依赖**：
    ```bash
    pip install numpy opencv-python pytest
    ```

## 运行方法

### 1. 直接运行示例

直接运行 `measure.py` 可以查看内置的合成图像测试效果：

```bash
python measure.py
```

### 2. 运行单元测试

项目包含基于 `pytest` 的测试用例，用于验证真实图片上的测量效果。

```bash
# 运行所有测试
pytest test_1d_measure.py

# 运行特定测试（例如测试基本调试功能）
pytest test_1d_measure.py::TestHalcon1DMeasure::test_basic_debug -s
```

### 3. 在代码中使用

```python
import cv2
import numpy as np
from measure import Halcon1DMeasure

# 读取图像
img = cv2.imread('path_to_image.jpg', cv2.IMREAD_GRAYSCALE)

# 初始化测量对象
# row, col: ROI中心; angle: 弧度; length1: 长度; length2: 宽度
measure = Halcon1DMeasure(
    row=300, col=400, angle=0, 
    length1=200, length2=50
)

# 执行边缘检测
row_edges, col_edges, amplitudes, distances = measure.measure_pos(
    img, sigma=1.5, threshold=30, transition='all', debug=False
)

# 执行边缘对检测
r1, c1, amp1, r2, c2, amp2, cr, cc, intra, inter = measure.measure_pairs(
    img, sigma=1.5, threshold=30
)

# 显示结果
measure.display_results(img, row_edges, col_edges, amplitudes)
```

## 关键参数说明

*   **sigma**: 高斯平滑系数，值越大抗噪性越强，但边缘位置可能会模糊。
*   **threshold**: 边缘幅度阈值。
    *   `normalize_threshold=False`: 范围 [0, 255]（默认）。
    *   `normalize_threshold=True`: 范围 [0, 1]。
*   **transition**: 边缘极性选择 (`'positive'`, `'negative'`, `'all'`)。
*   **select**: 结果选择 (`'all'`, `'first'`, `'last'`)。

## 许可证

本项目代码仅供学习和参考使用。

## 其他信息

*   **测试图片**：测试脚本中引用了 `data/sample/` 目录下的图片，请确保该路径下存在对应的测试图（如 `bottleneck_2.jpg`），或者修改代码中的图片路径。
*   **坐标系**：遵循图像处理惯例，原点在左上角，x 轴向右，y 轴向下。角度 0 表示水平向右，$\pi/2$ 表示垂直向下。