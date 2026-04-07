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
halcon_1d_measure_python/
├── halcon_1d_measure/           # 核心代码包
│   ├── __init__.py             # 包初始化
│   ├── measure.py              # 一维测量核心类
│   └── metrology.py            # 几何测量类
│
├── examples/                    # 示例代码
│   ├── quick_start.py          # 快速开始
│   ├── demo_measure_pos_debug.py
│   ├── demo_draw_roi.py
│   └── demo_gradient_scaling.py
│
├── tests/                       # 测试代码
│   ├── test_halcon_1d_measure_simplified.py
│   ├── test_metrology.py
│   └── ...
│
├── docs/                        # 文档和说明
│   ├── measure_pos_vs_measure_pairs.py
│   └── explain_gradient_scaling.py
│
├── README.md                    # 项目文档
├── PROJECT_OVERVIEW.md          # 项目概览
└── requirements.txt             # 依赖列表



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
