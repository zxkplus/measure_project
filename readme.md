Halcon 1D Measure - Python 实现


<div align="center">


<img alt="Python Version" src="https://img.shields.io/badge/python-3.6%2B-blue.svg">
<img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg">
<img alt="OpenCV" src="https://img.shields.io/badge/opencv-4.0%2B-orange.svg">


纯 Python 实现的 Halcon 一维测量工具，无需依赖 Halcon 商业库


</div>


📖 目录


功能特性
安装指南
快速开始
核心概念
API 文档
GUI 应用
示例代码
项目结构
常见问题


✨ 功能特性
🔍 一维边缘检测


measure_pos - 检测测量线上的所有边缘点
measure_pairs - 检测边缘对（如物体边缘）
亚像素精度 - 边缘定位精度可达亚像素级别
归一化阈值 - 支持 0-1 范围的阈值，用户友好
📐 几何测量 (Metrology)


直线测量 - 沿直线生成多个测量矩形，检测边缘点并拟合直线
圆测量 - 沿圆弧生成多个测量矩形，检测边缘点并拟合圆
Metrology 模型 - 统一管理多个测量对象
🎯 模板匹配测量

在参考图上点击选点 → 创建模板 → 在新图上自动匹配定位 → 计算两点距离
可配置预处理器 (Raw, Canny, Sobel, CLAHE, Threshold) 适应不同图像特征
支持模板序列化 (.npz) 跨会话复用
支持旋转/缩放不变匹配与多目标检测
🖥️ GUI 图形界面

Tkinter 原生桌面应用，零额外 GUI 依赖
交互式旋转 ROI 绘制：画框 → 裁图摆正 → 自动创建模板
摆正模板上交互添加测量工具 (边缘点/边缘对/拟合直线/拟合圆/距离/角度)
多目标自动匹配 → 逐个摆正 → 执行全部测量步骤
结果表格展示 + 汇总文本输出 + CSV 导出
🛠️ 调试与可视化


完整的调试模式 - 可视化中间处理过程
结果可视化 - 直观展示测量结果
ROI 绘制 - 在图像上绘制测量区域


📦 安装指南
方式一：使用 pip 安装（推荐）


bash
pip install numpy opencv-python scipy matplotlib

方式二：从源码安装


bash
git clone https://github.com/yourusername/halcon_1d_measure_python.git
cd halcon_1d_measure_python
pip install -r requirements.txt

依赖项


依赖	版本	必需	说明
Python	≥ 3.6	✓	运行环境
NumPy	≥ 1.16	✓	数值计算
OpenCV	≥ 4.0	✓	图像处理
SciPy	≥ 1.2	✓	科学计算
Matplotlib	≥ 3.0	可选	可视化
Pillow	≥ 8.0	GUI	图像格式转换


🚀 快速开始
1. 一维边缘检测


python
import cv2
import numpy as np
from halcon_1d_measure import Halcon1DMeasureSimplified

# 读取图像
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

# 创建测量对象
measure = Halcon1DMeasureSimplified(
    row=250,              # 中心行坐标
    col=250,              # 中心列坐标
    angle=np.radians(45), # 测量角度（弧度）
    length1=100,          # 沿测量方向的半长度
    length2=20            # 垂直测量方向的半宽度
)

# 执行边缘检测
row_edges, col_edges, amplitudes, distances = measure.measure_pos(
    img, 
    sigma=1.0,           # 高斯平滑参数
    threshold=30.0       # 边缘阈值
)

print(f"检测到 {len(row_edges)} 个边缘")

# 可视化结果
vis_img = measure.visualize_results(img, (row_edges, col_edges, amplitudes, distances))
cv2.imwrite('result.png', vis_img)

2. 直线测量


python
from halcon_1d_measure import LineMeasureObject

# 创建直线测量对象
line_measure = LineMeasureObject(
    start=(100, 50),       # 起点 (row, col)
    end=(100, 350),        # 终点 (row, col)
    measure_length1=10,    # 沿直线方向半长度
    measure_length2=20,    # 垂直直线方向半宽度
    num_measures=10        # 测量点数
)

# 执行测量
result = line_measure.measure(img)

if result:
    print(f"直线角度: {np.degrees(result['angle']):.2f}°")

# 可视化
vis_img = line_measure.visualize(img)
cv2.imwrite('line_result.png', vis_img)

3. 圆测量


python
from halcon_1d_measure import CircleMeasureObject

# 创建圆测量对象
circle_measure = CircleMeasureObject(
    center=(250, 250),     # 圆心 (row, col)
    radius=100,            # 半径
    measure_length1=10,    # 径向半长度
    measure_length2=20,    # 切向半宽度
    num_measures=16        # 测量点数
)

# 执行测量
result = circle_measure.measure(img)

if result:
    print(f"圆心: {result['center']}, 半径: {result['radius']:.2f}")


4. 模板匹配测量

```python
from measure_template import TemplatePoint, DistanceMeasure, CannyPreprocessor

# 从参考图创建两个模板点
pt_a = TemplatePoint(ref_img, click_row=200, click_col=150, template_size=80)
pt_b = TemplatePoint(ref_img, click_row=200, click_col=350, template_size=80)

# 也可使用预处理器增强匹配鲁棒性
# pt_a = TemplatePoint(ref_img, ..., preprocessor=CannyPreprocessor(50, 150))

# 保存模板供后续使用
pt_a.save("template_A.npz")
pt_b.save("template_B.npz")

# 在新图上匹配并计算距离
pt_a = TemplatePoint.from_file("template_A.npz")
pt_b = TemplatePoint.from_file("template_B.npz")
dm = DistanceMeasure(pt_a, pt_b)
result = dm.measure(new_img)
print(f"距离: {result["distance"]:.2f} px")

# 可视化
vis = dm.visualize(new_img)
cv2.imwrite("result.jpg", vis)
```


5. GUI 图形界面

```bash
# 启动 GUI 应用
python run_gui.py
```

**教学流程：**

1. 加载参考图 → 在图上**画旋转目标框**（点击中心 → 拖拽大小 → 滚轮调角度 → 双击确认）
2. 软件自动裁图摆正作为模板，在右侧预览
3. 在摆正模板上**交互添加测量工具**（选择工具类型 → 点击拖拽放置 → 弹出对话框调参数）
4. 可通过面板添加组合测量（距离/角度等）
5. 保存项目文件 (.npz)

**检测流程：**

1. 加载检测图 → 点击执行测量
2. 自动多目标匹配 → 每个目标独立摆正 → 执行全部测量
3. 底部面板展示：目标列表 | 测量结果表 | 汇总文本
4. 可导出 CSV

详细说明见下方 [🖥️ GUI 应用](#-gui-应用) 章节。


📚 核心概念
测量矩形 (Measure Rectangle)


测量矩形定义了边缘检测的区域和方向：


plaintext
        length1 (沿测量方向)
    ┌─────────────────────────┐
    │                         │
    │    ───────────────>     │  测量方向 (angle)
    │          center         │
    │                         │
    └─────────────────────────┘
        length2 (垂直测量方向)



参数说明：


参数	说明	范围
row, col	矩形中心的行列坐标	图像范围内
angle	测量方向（弧度）	0 ~ 2π
length1	沿测量方向的半长度	> 0
length2	垂直测量方向的半宽度	> 0


角度定义：


angle = 0：水平向右
angle = π/2：垂直向下
angle = π：水平向左
angle = 3π/2：垂直向上
阈值模式
归一化阈值模式（推荐）


python
# 阈值范围 [0, 1]，更直观
measure.measure_pos(img, threshold=0.2, normalize_threshold=True)
# threshold=0.2 表示检测对比度 > 255×0.2 = 51 的边缘

原始阈值模式（兼容 Halcon）


python
# 阈值范围 [0, 255]，与 Halcon 一致
measure.measure_pos(img, threshold=30, normalize_threshold=False)



推荐阈值范围：


归一化阈值	等效原始值	边缘强度	适用场景
0.05-0.15	13-38	弱边缘	细节检测
0.15-0.25	38-64	中等边缘	常用场景
0.25-0.40	64-102	强边缘	噪声抑制
0.40-0.60	102-153	极强边缘	显著边缘


📖 API 文档
Halcon1DMeasureSimplified 类


一维边缘检测的核心类。
初始化


python
Halcon1DMeasureSimplified(
    row: float,              # 中心行坐标
    col: float,              # 中心列坐标
    angle: float,            # 测量角度（弧度）
    length1: float,          # 沿测量方向的半长度
    length2: float,          # 垂直测量方向的半宽度
    interpolation: str = 'linear'  # 插值方式
)

主要方法
measure_pos() - 检测所有边缘点


python
row_edges, col_edges, amplitudes, distances = measure.measure_pos(
    img: np.ndarray,                    # 输入灰度图像
    sigma: float = 1.0,                 # 高斯平滑参数
    threshold: float = 30.0,            # 边缘阈值
    transition: str = 'all',            # 边缘类型
    select: str = 'all',                # 边缘选择
    normalize_threshold: bool = False,  # 是否归一化阈值
    debug: bool = False                 # 是否显示调试信息
)



参数说明：


参数	类型	默认值	说明
sigma	float	1.0	高斯平滑参数，越大越平滑
threshold	float	30.0	边缘阈值
transition	str	'all'	'all', 'positive', 'negative'
select	str	'all'	'all', 'first', 'last'
normalize_threshold	bool	False	是否使用 0-1 范围阈值
debug	bool	False	是否显示调试可视化


返回值：


返回值	类型	说明
row_edges	List[float]	边缘点的行坐标
col_edges	List[float]	边缘点的列坐标
amplitudes	List[float]	边缘梯度幅度
distances	List[float]	边缘到起点的距离
measure_pairs() - 检测边缘对


python
result = measure.measure_pairs(
    img: np.ndarray,
    sigma: float = 1.0,
    threshold: float = 30.0,
    transition: str = 'all',
    normalize_threshold: bool = False,
    debug: bool = False
)

visualize_results() - 可视化结果


python
vis_img = measure.visualize_results(
    img: np.ndarray,
    result: Tuple,
    line_thickness: int = 2
)



LineMeasureObject 类


直线测量对象，用于拟合直线边缘。
初始化


python
LineMeasureObject(
    start: Tuple[float, float],      # 起点 (row, col)
    end: Tuple[float, float],        # 终点 (row, col)
    measure_length1: float,          # 沿直线方向半长度
    measure_length2: float,          # 垂直直线方向半宽度
    num_measures: int = 10,          # 测量点数
    sigma: float = 1.0,              # 高斯平滑参数
    threshold: float = 30.0,         # 边缘阈值
    transition: str = 'all'          # 边缘类型
)

主要方法


measure(image) - 执行直线测量
visualize(image) - 可视化测量结果


CircleMeasureObject 类


圆测量对象，用于拟合圆形边缘。
初始化


python
CircleMeasureObject(
    center: Tuple[float, float],     # 圆心 (row, col)
    radius: float,                   # 半径
    measure_length1: float,          # 径向半长度
    measure_length2: float,          # 切向半宽度
    num_measures: int = 16,          # 测量点数
    start_angle: float = 0.0,        # 起始角度
    end_angle: float = 2*np.pi,      # 结束角度
    sigma: float = 1.0,              # 高斯平滑参数
    threshold: float = 30.0,         # 边缘阈值
    transition: str = 'all'          # 边缘类型
)

主要方法


measure(image) - 执行圆测量
visualize(image) - 可视化测量结果


MetrologyModel 类


统一管理多个测量对象的模型。


python
model = MetrologyModel()

# 添加测量对象
line_id = model.add_line_measure(...)
circle_id = model.add_circle_measure(...)

# 执行测量
model.apply(image)

# 获取结果
line_result = model.get_result(line_id)
circle_result = model.get_result(circle_id)




### 模板匹配 (Template Matching)

**TemplatePoint 类** — 单点模板匹配

```python
TemplatePoint(
    reference_image: np.ndarray,  # 参考图像 (灰度)
    click_row: float,             # 点击行坐标
    click_col: float,             # 点击列坐标
    template_size: int = 80,      # 模板正方形边长 (px)
    preprocessor: Preprocessor = None,  # 预处理器 (None=原始像素)
    match_score_threshold: float = 0.5, # 匹配置信度阈值
    use_subpixel: bool = True,    # 亚像素精炼
    rotation_invariant: bool = False,   # 旋转不变匹配
    angle_range: Tuple[float,float] = (-30, 30),  # 角度搜索范围 (度)
    scale_invariant: bool = False,       # 缩放不变匹配
    scale_range: Tuple[float,float] = (0.9, 1.1), # 缩放范围
    multi_target: bool = False,          # 多目标检测
    max_matches: int = 0,               # 最大匹配数 (0=无限制)
)

# 主要方法
measure(image, search_region=None) -> Dict  # 在新图上匹配
visualize(image, ...) -> np.ndarray         # 可视化结果
save(filepath) / from_file(filepath)       # 序列化
```

**预处理器 (Preprocessor)** — 可插拔的图像增强

| 类 | 说明 |
|------|------|
| `RawPreprocessor()` | 不做增强，使用原始像素 |
| `CannyPreprocessor(t1, t2)` | Canny 边缘检测 → 二值边缘图 |
| `SobelPreprocessor(ksize)` | Sobel 梯度幅值 |
| `CLAHEPreprocessor(clip_limit)` | 自适应直方图均衡化 |
| `ThresholdPreprocessor(threshold, mode)` | 全局阈值二值化 |

**DistanceMeasure 类** — 两点距离测量

```python
dm = DistanceMeasure(point_a: TemplatePoint, point_b: TemplatePoint)
result = dm.measure(image)  # -> {"point_a": ..., "point_b": ..., "distance": float, "valid": bool}
vis = dm.visualize(image)
```

### 工作流 (Workflow)

**MeasurementWorkflow 类** — 统一可组合测量工作流

```python
from measure_workflow import MeasurementWorkflow, TemplatePointObject, EdgePointObject, FitLineObject, TwoPointsDistanceObject

wf = MeasurementWorkflow()

# 定位模板
wf.add(TemplatePointObject("loc_A", ref_img, 120, 150, template_size=40, is_localization=True))
wf.add(TemplatePointObject("loc_B", ref_img, 330, 210, template_size=40, is_localization=True))

# 边缘探测
wf.add(EdgePointObject("edge", row=150, col=190, angle=0.0, length1=30, length2=10, transition="positive", select="first"))

# 拟合直线
wf.add(FitLineObject("line", start=(120, 180), end=(120, 400), measure_length1=5, measure_length2=25, num_measures=8))

# 组合测量
wf.add(TwoPointsDistanceObject("gap", "loc_A", "loc_B"))

# 执行
results = wf.measure(inspection_img)
vis = wf.visualize(inspection_img)
wf.save("project.npz")
```


🖥️ GUI 应用

启动方式

```bash
python run_gui.py
```

依赖: `pip install Pillow` (Tkinter 为 Python 内置，无需额外安装)

核心工作流

GUI 实现了一个完整的「教学 → 检测」视觉测量流程：

```
[教学模式]
参考图 → 画旋转目标框 → 裁图摆正 → 模板预览
       → 在摆正模板上交互添加测量工具
       → 保存项目 (.npz)

[检测模式]
检测图 → 加载项目 → 多目标模板匹配
       → 每个目标: 独立摆正 → 执行全部测量
       → 结果表格 + 汇总文本 + 可视化
```

教学步骤

1. **加载参考图** — Ctrl+O 或 菜单「文件 → 加载参考图」
2. **画旋转目标框**:
   - 点击确定中心点 → 拖拽定义框的宽高 → 鼠标滚轮调整旋转角度
   - 右侧实时显示摆正后的模板预览
   - 双击框内确认 ROI
3. **添加测量工具** — 在右侧模板预览上:
   - 选择工具类型 (边缘点/边缘对/拟合直线/拟合圆)
   - 点击拖拽放置工具 → 弹出对话框调整算法参数 (sigma, threshold 等)
4. **添加组合测量** — 在左侧面板点击组合测量按钮 (距离/角度/点线距等)
5. **保存项目** — Ctrl+S

检测步骤

1. **加载检测图** — Ctrl+I 或 菜单「文件 → 加载检测图」
2. **执行测量** — Ctrl+E 或 点击「▶ 执行测量」按钮
3. **查看结果**:
   - **目标列表**: 显示所有匹配到的目标 (ID / 分数 / 旋转角度 / 位置 / 状态)
   - **测量结果表**: 选中目标后显示各测量项的详细值
   - **汇总文本框**: 所有目标测量结果的纯文本汇总 (可复制)
4. **导出 CSV** — 结果面板「导出 CSV」按钮

GUI 布局

```
┌──────────────────────────────────────────────────────────┐
│ [菜单] 文件 视图 帮助                                     │
│ [工具条] 📁参考 📁检测 💾保存 📂加载 │ ▶执行 │ 教学模式    │
├──────────┬───────────────────────────┬───────────────────┤
│ ToolPanel│ [参考图] [检测图]         │  摆正模板预览      │
│          │ ┌───────────────────────┐ │  ┌───────────┐    │
│ 文件操作  │ │  主图像 (Canvas)       │ │  │ +边缘点    │    │
│ 模板设置  │ │  + ROI 旋转矩形       │ │  │ +边缘对    │    │
│ 测量工具  │ │  + 匹配结果           │ │  │ +拟合直线  │    │
│ 组合测量  │ │                       │ │  │ +拟合圆    │    │
│ 执行     │ └───────────────────────┘ │  └───────────┘    │
├──────────┴───────────────────────────┴───────────────────┤
│ ResultPanel: 目标列表 │ 测量结果表                        │
│ === 汇总文本 === ...                                     │
└──────────────────────────────────────────────────────────┘
```

支持的测量工具

| 类型 | 交互方式 | 说明 |
|------|---------|------|
| EdgePoint (边缘点) | 点击中心 + 拖拽方向 | 1D 边缘检测，返回单个亚像素边缘点 |
| EdgePair (边缘对) | 点击中心 + 拖拽方向 | 1D 边缘对检测，返回边缘对中心 |
| FitLine (拟合直线) | 点击起点 + 点击终点 | 沿线段放置多组测量矩形，拟合直线 |
| FitCircle (拟合圆) | 点击圆心 + 拖拽半径 | 沿圆弧放置多组测量矩形，拟合圆 |
| TwoPointsDistance | 选择两点 | 两点间欧氏距离 |
| TwoLinesAngle | 选择两线 | 两直线夹角 |
| PointLineDistance | 选择点+线 | 点到直线垂直距离 |
| PointCircleDistance | 选择点+圆 | 点到圆周的最短距离 |

快捷键

| 快捷键 | 功能 |
|--------|------|
| Ctrl+O | 加载参考图 |
| Ctrl+I | 加载检测图 |
| Ctrl+S | 保存项目 |
| Ctrl+E | 执行测量 |
| Escape | 取消当前操作 |
| 鼠标滚轮 | 缩放图像 / 调整 ROI 角度 |
| 双击 ROI | 确认旋转框 |
| 双击图像 | 适应窗口 |
💡 示例代码
示例 1：基本边缘检测


python
import cv2
import numpy as np
from halcon_1d_measure import Halcon1DMeasureSimplified

# 创建测试图像
img = np.ones((500, 500), dtype=np.uint8) * 255
cv2.line(img, (100, 250), (400, 250), 0, 2)

# 创建测量对象（水平测量）
measure = Halcon1DMeasureSimplified(
    row=250, col=250, angle=0,
    length1=150, length2=20
)

# 执行边缘检测
row_edges, col_edges, amplitudes, distances = measure.measure_pos(img)

print(f"检测到 {len(row_edges)} 个边缘")

# 可视化
vis_img = measure.visualize_results(img, (row_edges, col_edges, amplitudes, distances))
cv2.imwrite('example1_result.png', vis_img)

示例 2：调试模式


python
import cv2
import numpy as np
from halcon_1d_measure import Halcon1DMeasureSimplified

# 创建测试图像
img = np.ones((500, 500), dtype=np.uint8) * 255
cv2.rectangle(img, (100, 100), (400, 400), 0, 2)

# 创建测量对象
measure = Halcon1DMeasureSimplified(
    row=250, col=250, angle=0,
    length1=150, length2=50
)

# 执行边缘检测（开启调试模式）
row_edges, col_edges, amplitudes, distances = measure.measure_pos(
    img, 
    sigma=1.5, 
    threshold=0.15, 
    normalize_threshold=True,
    debug=True  # 开启调试可视化
)

示例 3：直线拟合


python
import cv2
import numpy as np
from halcon_1d_measure import LineMeasureObject

# 创建测试图像（带噪声的直线）
img = np.ones((500, 500), dtype=np.uint8) * 255
for i in range(50, 450):
    noise = np.random.randint(-3, 3)
    cv2.circle(img, (i, 250 + noise), 1, 0, -1)

# 创建直线测量对象
line_measure = LineMeasureObject(
    start=(250, 50),
    end=(250, 450),
    measure_length1=5,
    measure_length2=20,
    num_measures=20
)

# 执行测量
result = line_measure.measure(img)

if result:
    print(f"拟合直线角度: {np.degrees(result['angle']):.2f}°")

# 可视化
vis_img = line_measure.visualize(img)
cv2.imwrite('example3_result.png', vis_img)



📁 项目结构


plaintext
measure_project/
├── measure1D.py                  # 一维边缘检测核心
├── measure2D.py                  # 几何测量 (直线/圆拟合)
├── measure_template.py           # 模板匹配 + 预处理器
├── measure_workflow.py           # 统一可组合测量工作流
│
├── measure_gui/                  # GUI 图形界面包 (Tkinter)
│   ├── utils.py                  # 几何变换工具 (裁图摆正/坐标映射)
│   ├── multi_target.py           # 多目标测量编排器
│   ├── app.py                    # 主窗口 (整合所有组件)
│   ├── image_canvas.py           # 图像查看器 + 旋转 ROI 交互绘制
│   ├── template_view.py          # 摆正模板预览 + 测量工具交互添加
│   ├── tool_panel.py             # 左侧测量工具栏
│   ├── result_panel.py           # 底部结果展示面板
│   └── dialogs.py                # 参数编辑对话框
│
├── test_1d_measure.py            # 一维测量测试
├── test_2d_measure.py            # 二维测量测试
├── test_template_measure.py      # 模板匹配测试
├── test_measure_workflow.py      # 工作流测试
│
├── data/sample/                  # 样例图片 (gitignored)
├── run_gui.py                    # GUI 启动脚本
├── README.md                     # 项目文档
└── CLAUDE.md                     # 开发指南



❓ 常见问题
Q1: 为什么边缘检测结果与预期不符？


A: 可能的原因：


阈值设置不当 - 尝试调整 threshold 参数
测量方向错误 - 检查 angle 参数是否正确
图像质量问题 - 增大 sigma 参数以平滑噪声
Q2: 如何选择合适的阈值？


A: 建议：


使用归一化阈值模式（normalize_threshold=True）
从 0.15 开始，根据实际效果调整
边缘对比度高时，增大阈值
边缘对比度低时，减小阈值
Q3: 为什么 45 度角测量时边缘点看起来"垂直"？


A: 这是正常现象。测量线段从起点到终点，覆盖 180° 的角度范围。在 45° 角时，起点方向是 -135°，终点方向是 +45°，看起来像"垂直"，但实际上是正确的。详见 tests/FINAL_BUG_REPORT.py。
Q4: 如何提高测量精度？


A: 建议：


使用高质量的图像
适当增大 sigma 参数平滑噪声
调整测量矩形尺寸以覆盖足够的边缘区域
使用亚像素精炼（默认已启用）
Q5: measure_pos 和 measure_pairs 的区别？


A:


measure_pos: 检测测量线上的所有边缘点
measure_pairs: 检测边缘对，返回每对边缘的中心位置和距离


📄 许可证


本项目采用 MIT 许可证


📮 联系方式


项目主页: GitHub Repository
问题反馈: GitHub Issues


<div align="center">


⭐ 如果这个项目对你有帮助，请给一个 Star！⭐


Made with ❤️ by Halcon Measure Team


</div>
