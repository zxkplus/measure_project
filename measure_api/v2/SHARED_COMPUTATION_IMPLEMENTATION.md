# 共享计算引擎实施规格：接口、数据与迁移契约

状态：待实施的技术规格，2026 年 11 月 18 日核对代码。本文定义的是拟实现接口，
代码块不能视为当前已发布的 Python 接口。本次只编写文档，没有完成下述重构或缺陷修复。

本文承接[总体方案](SHARED_COMPUTATION_DESIGN.md)，将其中的改造清单展开成可实现的契约。
总体方案解释为什么统一、哪些旧行为不能照搬；本文规定具体新增或修改什么、传入什么、
返回什么、如何失败，以及旧调用方如何接入。实现涉及的独立正确性问题沿用总体方案 D01—D10 编号。

## 1. 实施范围与文件清单

这里的“接口”指 Python 计算接口。本次不新增 HTTP 路由，不合并 V1 会话与 V2 无状态服务，
也不要求两种资产目录互相加载。HTTP 适配继续负责读图、Pydantic、资源限制、任务与文件发布。

| 标记 | 含义 | 如何实施 |
|---|---|---|
| A | 已有共享实现 | 复用，必要时修复；不增加同义算法 |
| B | 旧引擎已有计算，缺少纯计算入口 | 提取后由旧类委托，删除旧计算主体 |
| C | API 层已有计算 | 下沉到 `measure/`，API 保留适配 |
| D | 现有计算或契约不完整 | 新增类型、校验、诊断或正确性修复 |

| 文件 | 状态 | 主要变更 |
|---|---|---|
| `measure/computation_types.py` | 新增，D | 通用空间、观察结果、数值异常和几何值；不放业务算法 |
| `measure/measure1D.py` | 已有，B+C | 提取轮廓、边缘和完整边缘对 |
| `measure/measure2D.py` | 已有，B+C+D | 提取完整探针到拟合流程，保留旧类适配 |
| `measure/geometric_fits.py` | 已有，A+D | 保留旧拟合入口，增加有类型结果入口，共享同一求解主体 |
| `measure/template_search.py` | 新增，B+C+D | 模板准备、搜索及局部细化 |
| `measure/preprocessing.py` | 新增，B+C | 内存像素预处理 |
| `measure/rectify.py` | 已有，A+C+D | 保留原裁剪签名，补齐有类型裁剪与通用重采样 |
| `measure/alignment.py` | 已有，B+C+D | 相似变换估计与控制点修正 |
| `measure/geometry_ops.py` | 新增，B+C+D | 几何组合、圆心转换和圆采样指标 |
| `measure/measure_calibration.py` | 已有，B+C+D | 内存圆点检测、相机和双目求解、数值校验 |
| `measure/pixel_calibration.py` | 已有，A+D | 复用棋盘格与比例求解；补齐直接调用校验 |
| `measure/plane_geometry.py` | 新增，B+C | 平面位姿、投影、画布和拼接 |
| `measure/geometry_mapping.py` | 新增，B+C+D | 完整几何变换、畸变映射和采样后计量 |
| `measure/evaluation.py` | 新增，C+D | 质量汇总与产品评定 |
| `measure/viz.py` | 已有，A+B+C | 只返回栅格的绘制与诊断预览 |
| `measure/pick_api.py`、`measure/robot_calibration.py` | 已有，A | 原入口继续复用，本次无新增签名 |

共新增 7 个文件：总体方案中的 6 个计算模块，以及本文补充的 1 个公共类型模块。
各模块独有的参数、结果类型放在所属模块，不把所有类型塞进公共文件。
公共模块不得导入工作流、模板、API、GUI；其他共享模块可以依赖它。

阅读路线：先实现第 2 节公共契约，再按第 15 节任务顺序推进；每个模块以本节的接口表和验收项为完成标准。

### 1.1 接口索引与实现来源

新增入口不等于新增算法。下表同时说明代码从哪里来；后文各节规定该入口的完整契约。

| 入口 | 当前状态及来源 | 类型 | 规格位置 |
|---|---|---|---|
| Space、Observation、Mapping2D及make_mapping | 尚无统一普通Python契约，整合旧结果与V2观察/映射规则 | D | 第2节 |
| sample_profile、detect_edges、detect_edge_pairs | 尚无这些函数；Halcon1DMeasure和V2 extract_profile已有计算 | B+C | 第3节 |
| fit_line_points、fit_circle_points | 已有共享函数，保留旧签名和返回表示 | A | 第4.1节 |
| fit_line_observation、fit_circle_observation | 新增有类型入口，复用上述拟合主体 | A+D | 第4.1节 |
| probe_and_fit_line、probe_and_fit_circle | 尚无这些函数；旧Line/CircleMeasureObject与V2同类测量已有计算 | B+C+D | 第4.2节 |
| preprocess_pixels | 新入口；旧预处理类和V2 apply_preprocessing已有计算 | B+C | 第5节 |
| crop_and_straighten、矩阵/点映射工具 | 已有共享函数；不改历史返回方向 | A | 第6节 |
| crop_region、warp_mapped、legacy_roi_to_box | 新契约入口；rectify、alignment、V2图像与旧转换中已有公式 | A+B+C | 第6节 |
| prepare_support、search_support | 新入口；TemplatePoint与V2 build/match_template已有计算 | B+C+D | 第7节 |
| estimate_similarity、correct_alignment | 前者V2同名、后者新名；旧SimilarityTransform/MultiPointAlignment已有计算 | B+C+D | 第8节 |
| circle_center | 新纯转换，旧CircleResult别名及V2端口需共同适配 | D | 第9节 |
| line_through_points及四种距离/角度组合 | 新共享入口，旧两条工作流和V2 geometry已有公式 | B+C+D | 第9节 |
| sampled_diameter_spread | 新共享函数名，旧fit_circle_points与V2 circle_dimensions已有公式 | A+C+D | 第9节 |
| transform_geometry、map_geometry_for_display | 新共享归属；V2前者同名、后者对应result_mapping.map_geometry | B+C+D | 第10.1节 |
| map_sample_points、map_metric_samples | 新入口；V2 SampleMapper.map_points/convert及显示映射已有步骤 | C | 第10.2节 |
| detect_dot_grid | V2有同名函数；旧CameraCalibration与StereoRigCalibration也有检测 | B+C | 第11.2节 |
| fit_camera_intrinsics、fit_stereo_rig | 新共享入口；旧标定类及V2 compute_camera/compute_stereo已有求解 | B+C+D | 第11.3—11.4节 |
| 棋盘格calibrate/compute_scale | 已有共享类方法继续复用 | A | 第11.5节 |
| validate_scalar_scale、validate_camera_model、validate_pose | 新共享校验入口；规则由旧类和V2数值校验器整理补齐 | B+C+D | 第11.5节 |
| solve_plane_pose、build_plane_view、build_plane_canvas | 新共享入口；旧_solve_board_pose/_compute_canvas与V2 pose/make_plane_view/plane_parameters已有计算 | B+C | 第12节 |
| project_camera_points、stitch_plane_views、rebase_plane | 新共享入口；旧投影/拼接与V2 project_points/stitch_images/adjust_origin已有计算 | B+C | 第12节 |
| evaluate_range、diameter_variation、centerline_deviation | evaluate_range在V2同名，另两项V2名为evaluate_variation/evaluate_deviation | C | 第13.1节 |
| summarize_quality | 新入口；V1 compute_quality与V2 fit_diagnostics已有不完整规则 | C+D | 第13.2节 |
| rasterize_geometry、render_profile、render_probes | 新共享入口；viz辅助函数、V2 rasterize/curve_preview/probe_provenance已有计算 | A+B+C | 第14节 |
| GantryPick与ImageRobotCalibration相关入口 | 已有共享实现，本次保留原签名 | A | 第1节 |

## 2. 所有接口共同遵守的契约

### 2.1 类型、空间与有效性

使用标准库 `dataclasses`、`typing` 和 NumPy，兼容 Python 3.9。
代码中的 `Array` 表示 `np.ndarray`，不是隐式接受任意嵌套对象数组。
几何标量用 Python `float`，点用 `XY = Tuple[float, float]`。
除注明默认值的字段外，下文类型表中的字段均为必填。

```python
# measure/computation_types.py：拟新增的公共定义
from dataclasses import dataclass, field
from typing import Any, Generic, Mapping, Optional, Tuple, TypeVar

import numpy as np

Array = np.ndarray
XY = Tuple[float, float]
T = TypeVar("T")

@dataclass(frozen=True)
class Space:
    frame: str
    unit: str

@dataclass(frozen=True)
class Observation(Generic[T]):
    space: Space
    valid: bool
    value: Optional[T]
    reason: Optional[str] = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class LineValue:
    coefficients: Tuple[float, float, float]
    start: Optional[XY] = None
    end: Optional[XY] = None

@dataclass(frozen=True)
class CircleValue:
    center: XY
    radius: float

@dataclass(frozen=True)
class RotatedBox:
    center: XY
    size_wh: Tuple[float, float]
    angle_rad: float

@dataclass(frozen=True)
class PolylineValue:
    points_xy: Array
    closed: bool = False

class ComputationContractError(ValueError):
    # __init__(code: str, message: str, *, field: Optional[str] = None)
    # 实例属性：code、message、field；不携带 HTTP 状态码。
    pass
```

`Observation` 必须通过构造后校验或共享工厂保证以下不变量，不能只依赖类型注解：

| 条目 | 必须成立的规则 |
|---|---|
| `space` | `frame`、`unit` 非空；空间身份由调用者提供，不自动猜相机或目标 |
| 有效观察 | `valid=True`、`value is not None`、`reason=None`；载荷数值有限 |
| 无效观察 | `valid=False`、`value=None`、`reason` 为稳定非空原因码 |
| 诊断 | 可以保留失败前的真实采样和计数；不允许把诊断当成有效观察载荷继续组合 |
| 嵌套类型 | 点有限；圆半径 `>0`；线系数非退化；每个模块执行自己的完整载荷校验 |
| 诊断字段 | `diagnostics` 的必需字段和类型由对应小节列出；扩展字段只能新增，不能代替必需字段 |

`diagnostics` 可以包含 NumPy 数组，属于内部协议；API 负责转换为列表或 NPZ。
类型中 `frozen=True` 只限制属性赋值，不会自动冻结数组：所有返回数组必须由本次调用拥有，
准备缓存中的数组必须独立复制并设置只读；不得通过返回值意外修改输入、其他结果或共享缓存。
LineValue在构造后完成归一化、确定符号和端点一致性检查；V2 Line.normalize改为构造该值后读取规范系数，
不保留第二套归一化公式。端点残差容差为 `1e-7 * max(1,norm(point))`。
Space、PixelWindow、Mapping2D和make_mapping均放公共类型模块，其余参数/载荷放对应计算模块。

### 2.2 坐标、单位、数组与图像

| 项目 | 明确契约 |
|---|---|
| 二维点 | `(x,y)=(col,row)`；数组形状 `(N,2)`，`float64`，保持输入顺序 |
| 三维点 | `(X,Y,Z)`，数组 `(N,3)`，右手相机坐标：X 向右、Y 向下、Z 向前 |
| 图像 | 非空连续或非连续 `uint8` 灰度 `(H,W)` / BGR `(H,W,3)`；预处理后的搜索像素另有声明；拒绝 BGRA、RGB 猜测和 object 数组 |
| 像素位置 | 整数坐标指像素中心；右下中心为 `(W-1,H-1)` |
| 矩形窗口 | `PixelWindow(x,y,width,height)`，四个整数，x/y≥0，宽高>0；切片为 `[y:y+h,x:x+w]` |
| 方向角 | 弧度，`atan2(dy,dx)`，x 向右、y 向下；正角在图像中视觉顺时针 |
| 直线 | `a*x+b*y+c=0`，`hypot(a,b)=1`；确定符号：a 非零时 a>0，否则 b>0 |
| 长度单位 | `px`、`mm`、`cm`、`m`、`um`；换算必须显式；物理标定仅接受后四种 |
| 角度输出 | `line_angle` 输出 `rad`，API 需要时转 `deg`；角度不乘长度比例 |
| 空间匹配 | 组合先检查 frame、unit，再检查 valid；不匹配抛契约异常，即使观察无效也不跳过空间检查 |
| 端点 | 线的 `start/end` 若出现须同时出现并满足线方程；无限线计算不因有限展示端点而变成线段距离 |

拟合端点沿规范方向 `d=(-b,a)` 排序。旧拟合器两端各扩展投影跨度 10% 的行为保留；
两点成线则以输入的真实端点为区间，按规范方向排序，不扩展。线的无向方向角归一化到 `[0,π)`。

图像工具和模板支持区使用连续几何尺寸，整数输出宽高为 `int(size)`，即正数向下取整。
保持局部中心 `width/2,height/2`，不改成 `(size-1)/2`；连续尺寸与整数输出尺寸同时记录。
1D 探针的局部原点单独遵守第 3 节，不能用新整数中心替换旧半长。

### 2.3 错误与失败

| 情况 | 统一处理 | 典型原因码 |
|---|---|---|
| 错误类型、形状、非有限输入、参数范围错误、空间冲突 | 抛 `ComputationContractError` | `contract.invalid_shape`、`contract.nonfinite`、`contract.invalid_parameter`、`contract.space_mismatch` |
| 格式正确但观测不足、退化、未检测到目标、求解无有效解 | 返回无效 `Observation` | `insufficient_samples`、`coincident_points`、`collinear_points`、`no_edges`、`no_pairs`、`no_matches`、`degenerate_correspondences`、`solver_failed` |
| 给定变换无法精确表示某几何类型 | 返回无效观察，保留原因 | `unsupported_geometry_mapping` |
| 实现错误、意外异常 | 传播给执行层记录节点失败；不能伪装为无边缘或吞掉后复用旧结果 | 执行层错误码，不写入观察值 |

OpenCV 可预期求解失败在所属模块转换为 `solver_failed`；`MemoryError`、取消、程序错误不得吞掉。
共享函数无日志打印、窗口或文件副作用。取消、超时、输入文件权限由适配层处理。
V2 现有错误码由适配层映射；不得直接把所有契约异常统一改成一个 HTTP 500。

### 2.4 映射类型

新增公共类型 `Mapping2D`，字段如下；它只表示可由 3×3 矩阵完整表达的映射。
镜头畸变不得谎称为一个矩阵，使用第 10 节的有序映射步骤。

| 字段 | 类型 | 契约 |
|---|---|---|
| `source`、`destination` | `Space` | 包含源、目标各自单位 |
| `kind` | `Literal['similarity','affine','projective']` | 必须与矩阵实际性质一致 |
| `matrix` | `Array (3,3)` | `p_destination ~ H @ p_source`，列向量 |
| `inverse_matrix` | `Array (3,3)` | 由引擎计算，禁止调用者传两份互相矛盾的矩阵 |

公共构造函数：

```python
def make_mapping(
    matrix: Array, *, source: Space, destination: Space, kind: str
) -> Mapping2D: ...
```

系数有限、可逆；先按最大绝对元素归一化后检查奇异值，最小/最大奇异值必须大于 `1e-12`。
仿射最后一行为 `[0,0,1]`；相似映射还要求 `A.T @ A = s²I`、s>0、det(A)>0，
使用 `rtol=1e-9, atol=1e-12`；本期不接受反射作为相似映射。
求逆与正反往返必须在该容差内成立。投影点使用归一化矩阵，齐次分母绝对值≤`1e-12` 为无效。
曲线、区域映射还拒绝相邻端点分母跨符号，不能只检查有限离散点后连过无穷远。

## 3. 1D 轮廓、边缘、边缘对

位置：已有 `measure/measure1D.py`；来源：`Halcon1DMeasure` 和 V2 `families/measurements.py`。
类型 B+C；保留 `Halcon1DMeasure` 构造及返回元组接口，由它适配以下函数。

```python
def sample_profile(
    image: Array, probe: ProbeSpec, *, space: Space, sigma: float = 1.0
) -> Observation[ProfileData]: ...

def detect_edges(
    image: Array, probe: ProbeSpec, options: EdgeOptions, *, space: Space
) -> Observation[EdgeSet]: ...

def detect_edge_pairs(
    image: Array, probe: ProbeSpec, options: PairOptions, *, space: Space
) -> Observation[EdgePairSet]: ...
```

### 3.1 参数类型

| 类型 | 字段及默认值 | 校验与含义 |
|---|---|---|
| `ProbeSpec` | `center: XY`；`angle_rad: float`；`search_half_length: float`；`average_half_width: float` | 空间必须 px；半搜索长度≥2，半平均宽≥0.5 |
| 同上 | `interpolation='linear'` | `nearest/linear/cubic` |
| 同上 | `border_policy='reject'` | `reject/replicate`；reject 检查实际输出网格的四角，容差 `1e-7 px`；越界返回 `probe_outside_image` |
| `EdgeOptions` | `sigma=1.0`；`threshold=30.0`；`threshold_unit='gray'` | sigma≥0；gray 阈值 `[0,255]`，fraction `[0,1]`，乘 255 得 `threshold_gray` |
| 同上 | `polarity='all'`；`selection='all'` | 极性 `positive/negative/all`；选择 `first/last/all` |
| `PairOptions` | sigma、threshold、threshold_unit 同上；`polarity='all'`；`selection='all'` | 选择另支持 `endpoints`；不使用单边缘的 first/last 提前过滤 |

输出 ROI 为 `(int(2*average_half_width),int(2*search_half_length))`。
局部 `(search_half_length,average_half_width)` 对应输入中心；
`sample_points_xy[i]` 是局部 `(i,average_half_width)` 的逆映射，不能用取整后高度的半高替换。
平均 axis=0，输出 float64；sigma=0 跳过平滑，否则 `gaussian_filter1d(mode='reflect',truncate=4)`；
梯度使用 `np.gradient` 的现有一阶端点规则。寻峰与三点二次细化复用现有信号函数。

### 3.2 返回载荷与诊断

| 类型 | 字段 | 规则 |
|---|---|---|
| `ProfileData` | `gray,smoothed,gradient: Array (W,)`；`sample_points_xy: Array (W,2)` | float64；数组逐样本对应 |
| 同上 | `source_to_probe: Mapping2D`；`roi_pixels: Array (H,W)` | ROI 像素 uint8；映射方向明确 |
| `EdgeValue` | `point: XY`；`amplitude: float`；`profile_position: float` | 幅度带符号，正为沿探针方向暗→亮；位置为亚像素列 |
| `EdgeSet` | `edges: Tuple[EdgeValue,...]`；`distances: Array (K-1,)` | 按 profile_position 升序；间距是选择完成后相邻点欧氏距离 |
| `EdgePairValue` | `edge_a,edge_b: EdgeValue`；`center: XY`；`width: float` | A 沿探针先出现、B 后出现；中心为中点、宽度为欧氏距离 |
| `EdgePairSet` | `pairs: Tuple[EdgePairValue,...]`；`center_distances: Array` | 完整配对选择后计算中心间距，长度 max(K-1,0) |

边缘及边缘对的 `diagnostics` 必填：`threshold_gray: float`、`probe: ProbeSpec`、
`profile: ProfileData`（探针越界时为 None）、`candidate_count: int`、`selected_count: int`。
未找到边缘时观察无效，载荷为 None，诊断仍可展示真实轮廓。

配对流程固定：检测所有正负边缘→按位置遍历相邻边缘→按首边极性筛完整对→成功后前进两边、
失败前进一边→选择完整对。positive 为 `+,-`，negative 为 `-,+`。
`endpoints` 保留现有特例：所有边缘中的首尾组成一对，忽略配对极性，并记录
`pairing_mode='endpoints'`；不足两边无效。普通模式记录 `pairing_mode='adjacent_opposite'`。

旧 `measure_pairs()` 的十项元组顺序保持；灰度阈值通过适配转入，不能因函数重命名改变配对次序。
V2 单点/单对端口只在所选结果恰好一项时输出；多项不得偷偷取第一项当作唯一端口。

## 4. 线圆探针测量与拟合

### 4.1 拟合函数的兼容方式

已有 `fit_line_points(points)->dict`、`fit_circle_points(points)->dict` 保留签名及历史行列字段。
不能直接把其返回字典的 `center/start/end` 当成 xy。新增规范入口：

```python
# measure/geometric_fits.py
def fit_line_observation(points_xy: Array, *, space: Space) -> Observation[LineValue]: ...
def fit_circle_observation(points_xy: Array, *, space: Space) -> Observation[CircleValue]: ...
```

两种入口使用同一私有求解主体：旧函数序列化旧字典，新函数组装规范观察，禁止各拟合一次。
保持现有直线正交 SVD 与圆平移条件化代数最小二乘；本期不额外增加鲁棒拟合模式。
有效线至少两个不同位置；有效圆至少三个不共线位置。格式正确但不足或退化返回无效观察。
非有限样本属于契约异常。线系数与有符号残差同时规范符号，不能只翻系数不翻残差。

新增 `FitDiagnostics` 字段，通过 `diagnostics['fit']` 返回：

| 字段 | 类型 / 默认 | 含义 |
|---|---|---|
| `samples_xy` | `Array (N,2)` | 真正送入拟合的所选点，含原顺序 |
| `sample_count` | `int` | N，不是成功探针数量 |
| `residuals` | `Optional[Array (N,)]` | 成功时直线为 `a*x+b*y+c`；圆为 `norm(p-center)-radius`；失败为 None |
| `mean_absolute_error,max_absolute_error,rms_error` | `Optional[float]` | 由真实残差计算，失败为 None |
| `residual_space` | `Space` | 残差的坐标系和长度单位 |
| `requested_probe_count,successful_probe_count` | `Optional[int] = None` | 纯点拟合未知，不填 0；探针拟合必须填写 |
| `sample_probe_indices` | `Optional[Array (N,)] = None` | int64，采样来源探针编号；纯点拟合无该身份 |

圆成功结果的诊断另有 `min_radius,max_radius,sampled_diameter_spread: float`、
`min_radius_point,max_radius_point: XY`；极值并列取输入顺序第一项。
`sampled_diameter_spread=2*(max_radius-min_radius)`，不命名为椭圆度或标准圆度。

### 4.2 完整图像测量入口

```python
# measure/measure2D.py
def probe_and_fit_line(
    image: Array, expected: LineSpec, options: ProbeFitOptions, *, space: Space
) -> Observation[LineValue]: ...

def probe_and_fit_circle(
    image: Array, expected: CircleSpec, options: ProbeFitOptions, *, space: Space
) -> Observation[CircleValue]: ...
```

| 参数类型 | 字段 | 默认与限制 |
|---|---|---|
| `LineSpec` | `start,end: XY`；`probe_count: int=20` | 至少 2 个探针；重合起终点返回 `coincident_expected_points` |
| `CircleSpec` | `center: XY`；`radius: float`；`probe_count: int=24` | radius>0，至少 3 个探针 |
| 同上 | `start_angle_rad=0`；`sweep_angle_rad=2*pi` | `0<sweep≤2*pi`，不使用含混的等起终角；API 旧起终角先转换 |
| 同上 | `radius_limits: Optional[Tuple[float,float]]=None` | None 表示按下文 expected_band 生成；显式范围需 0≤min≤max |
| `ProbeFitOptions` | `search_half_length=15`；`average_half_width=5`；`sigma=1`；`threshold=30`；`threshold_unit='gray'`；`polarity='all'` | 与第 3 节同义 |
| 同上 | `selection='nearest'`；`interpolation='linear'`；`border_policy='reject'` | selection 支持 nearest/first/last/all |
| 同上 | `radius_filter='expected_band'` | `expected_band/unbounded`；仅圆使用；显式 radius_limits 与 unbounded 不能并用 |

圆默认带宽为 `[max(0,radius-search_half_length),radius+search_half_length]`。
如 V2 现有配置明确要求不限制半径，适配为 unbounded，必须显式保存该策略；不得因 None 被重新解释而静默改变资产。
新请求和未显式声明的共有默认使用 expected_band。

直线在起终点间含两端均匀布置探针，方向 `atan2(end-start)+pi/2`。
nearest 选择到**有限预期线段**距离最小的候选，投影参数夹到 `[0,1]`。
圆在 `start+i*sweep/probe_count` 布置探针，不重复闭合端点，方向指向圆心；
先做包含边界的半径筛选，再按到预期半径的偏差选 nearest。
first/last/all 均按探针位置顺序解释，不按圆半径重排；nearest 并列取先出现者。

结果诊断必填 `fit: FitDiagnostics`、`probes: Tuple[ProbeRecord,...]`。
`ProbeRecord` 字段：`index:int`、`spec:ProbeSpec`、`status:'selected'/'no_edge'/'outside'`、
`candidates:Tuple[EdgeValue,...]`、`selected:Tuple[EdgeValue,...]`、`reason:Optional[str]`。
候选记录为极性过滤后、几何半径筛选前的数据；选中记录是真正进入拟合的数据。
每个预定探针都有记录；成功探针是至少一个选中点的探针，一个探针选多边不能增加成功探针数。
越界探针记录 outside 后继续其他探针；最终不足返回 `insufficient_samples`，不丢诊断。
意外异常仍传播；不能复制旧类的宽泛 `except Exception: print(); continue`。

### 4.3 旧类与 V2 的字段映射

| 调用来源 | 转换规则 |
|---|---|
| `Halcon1DMeasure` | row/col→xy；length1→search_half_length，length2→average_half_width；旧边界行为显式 replicate |
| `LineMeasureObject` | **measure_length2→search_half_length，measure_length1→average_half_width**；不能按数字顺序对接 |
| `CircleMeasureObject` | **measure_length1→search_half_length，measure_length2→average_half_width**；使用构造器已经解析的 radius_min/max |
| 旧探针数量 | 原配置显式传入 10/12 或保存值；不能被规范默认 20/24 覆盖 |
| V2 `ImageFitRequest` | 参数大部分同义；角度 deg→rad，输入点 x/y 原样；nearest 改用上述线段规则并列为有意行为变化 |
| 旧结果包装类 | 从同一个观察转换成 `LineResult/CircleResult`；透传 fit、探针计数和实际样本，修复 D09 |

旧圆radius_min若为负，适配器可取max(0,radius_min)，因为真实欧氏半径不可能为负，
此转换保持有效候选集合不变；不得因这一等价规范化拒绝原本可用的旧模型。

## 5. 像素预处理

位置：新增 `measure/preprocessing.py`，B+C；旧预处理类继续负责注册和序列化，
内置计算委托共享函数。V2 Pydantic 判别联合转换为下列普通参数类型。

```python
def preprocess_pixels(image: Array, options: PreprocessSpec) -> Array: ...
```

`PreprocessSpec` 使用普通 dataclass 判别联合，每种只带自己的字段：

为使实现与测试使用同一组可构造类型，名称固定为 `RawSpec`、`CannySpec`、`SobelSpec`、
`ClaheSpec`、`ThresholdSpec`，定义在 `measure/preprocessing.py`；
`PreprocessSpec` 是这五种类型的 `Union` 别名，不是接收任意字段的通用构造器。

| type | 入参及默认值 | 出参 |
|---|---|---|
| raw | `channel_mode='gray'`，可选 gray/preserve | gray 先 BGR→灰度，输出 float32；preserve 保留通道，输出 float32 |
| canny | `low_threshold=50`、`high_threshold=150`；0≤low≤high≤255 | 先灰度；uint8，0/255 |
| sobel | `kernel_size=3`，只允许 1/3/5/7 | 先灰度；`sqrt(gx**2+gy**2)`，float32，不压到 255 |
| clahe | `clip_limit=2.0>0`；`tile_grid_wh=(8,8)`，各自为正整数 | 先灰度；CLAHE 后转 float32 |
| threshold | `threshold=127`，`mode='binary'` | 先灰度；uint8；模式支持 binary/binary_inv/truncate/tozero/tozero_inv |

输出空间不变，无观察有效性，因为这是确定的像素转换；非法参数抛契约异常。
匹配必须对模板和待检图使用完全相同参数。模板搜索只接受灰度输出；raw/preserve
用于独立图像预处理，传给模板准备时必须拒绝，不能静默改变通道。
旧内置预处理本就使用灰度，适配为 gray；V2 独立 raw 图像操作适配 preserve，
V2 模板 raw 灰度约定需要重建相应资产并记录变更。
自定义旧 preprocessor 注册仍可保留；只有明确的纯计算 callable 才可在 Python 教学层运行，
不能不带身份地进入可序列化共享模板，V2 也不据此新增任意代码执行能力。

## 6. 旋转裁剪、重采样与旧 ROI 转换

已有 `crop_and_straighten(image,center_rc,size_hw,angle_deg,...)->(patch,H)` 保持实际签名：
实际代码参数仍叫 `center,size`，此处 rc/hw 只说明含义。H 实际是原图→patch，修正文档，
不能在修复文档时悄悄将返回矩阵取逆。旧点映射函数仍按历史行列参数适配。

新入口选择有实际数据契约的名字，不新增仅作同义包装的 `crop_rotated`：

```python
# measure/rectify.py
def crop_region(
    image: Array, box: RotatedBox, *, source: Space, destination: Space,
    options: ResampleOptions
) -> CropResult: ...

def warp_mapped(
    image: Array, mapping: Mapping2D, output_size_wh: Tuple[int, int], *,
    options: ResampleOptions
) -> CropResult: ...

def legacy_roi_to_box(
    *, row: float, col: float, width: float, height: float,
    angle_deg: float
) -> RotatedBox: ...
```

`ResampleOptions`：`interpolation='linear'`（nearest/linear/cubic）、
`border_mode='constant'`（constant/replicate）、`border_value=0`（uint8 灰度值或 BGR 三元组）。
`CropResult`：`pixels:Array`、`mapping:Mapping2D`、`valid_mask:Array(H,W),bool`、
`requested_size_wh:Tuple[float,float]`；warp 的 requested_size 等于显式输出尺寸。
矩阵合法性由公共构造器验证；空输出或非 px 图像空间抛契约异常。
crop_region和warp_mapped还允许预处理后的float32灰度/BGR数组，保持dtype；
其他类型拒绝。旧crop_and_straighten不因新包装而扩大其已声明输入范围。

crop 保留 size/2 原点，映射为：
`H_patch_to_source=[[cos,-sin,cx-cos*w/2+sin*h/2],
[sin,cos,cy-sin*w/2-cos*h/2],[0,0,1]]`，返回 mapping 是它的逆。
两个裁剪入口和旧函数共用一次重采样主体。
有效掩膜表示像素中心逆映射是否落在原图中心范围内，不能因 replicate 填充而变为真实像素；
需要插值邻域完全有效的模板匹配，还必须按核半径收缩有效区域。

`legacy_roi_to_box` 输入就是旧 ROI 连续矩形，输出 xy、弧度；固定矩阵构造由 crop 共享。
旧工具局部坐标与 V2 `(size-1)/2` 的转换写入转换器的显式平移，不在此增加版本开关。
旧 ROI 转换只做几何转换，不读模型目录或写新资产。

## 7. 模板准备与搜索

位置：新增 `measure/template_search.py`，B+C+D。来源：`TemplatePoint` 与 V2 templates。
不将旧类中修改 self.template、临时改尺寸后恢复的方式带入共享搜索。

```python
def prepare_support(
    reference_image: Array, spec: SupportSpec, preprocessing: PreprocessSpec, *,
    reference_space: Space, support_space: Space
) -> Observation[PreparedSupport]: ...

def search_support(
    image: Array, prepared: PreparedSupport, options: SearchOptions, *, image_space: Space
) -> Observation[SearchData]: ...
```

### 7.1 教学与准备参数

| 类型 | 字段 | 契约 |
|---|---|---|
| `SupportSpec` | `box:RotatedBox`；`anchor_reference:XY` | 矩形支持区、参考图中的锚点；两者不要求同中心 |
| 同上 | `mask:Optional[Array]=None` | 原始支持区整数尺寸的 bool 掩膜，True 参与相关；不是原图尺寸 |
| `PreparedSupport` | `pixels:Array(H,W)`；`valid_mask:Array(H,W),bool` | pixels 只允许 uint8/float32；完全独立、只读；mask 与像素同尺寸 |
| 同上 | `reference_space,support_space:Space`；`reference_to_support:Mapping2D` | 两空间单位均 px；映射包含支持区裁剪和方向 |
| 同上 | `anchor_support:XY`；`box:RotatedBox` | anchor_support 由 reference_to_support 得到；必须在有效支持区的几何包围范围内 |
| 同上 | `preprocessing:PreprocessSpec`；`sampling_revision:str` | 保存解析后的全部参数；初次新契约修订值使用 `shared-1` |

准备顺序固定：在原始分辨率对完整参考图预处理→裁剪处理后的支持区；
搜索顺序为原始分辨率完整待检图预处理→限制有效搜索窗口→生成金字塔和位姿变体。
不在某个候选crop上重新做CLAHE/Canny，也不对已处理的模板再做一次预处理。
这避免因模板crop边界与检测图边界不同而形成两种局部滤波过程。
该顺序与部分旧路径不同，必须重建相关模板并验证；图像姿态/CLAHE分块仍可能影响分数，
不能把任意旋转目标与参考图的score=1当作契约。
预处理顺序属于sampling_revision，改变顺序必须重建模板像素。

有效 mask 为用户 mask 与裁剪真实有效 mask 的交集。少于 4 个有效像素、有效像素无方差时返回
`insufficient_support` / `template_no_texture`；不能把固定黑色背景当纹理。
参考图不覆盖支持区时默认返回 `support_outside_image`，不以填充像素偷偷完成教学。
锚点只表示所测特征位置，不等同于矩形中心，也不等同于相关峰左上角。

### 7.2 搜索参数

`SearchOptions` 全部为普通值；新调用默认如下。显式保存的旧值由适配器传入，不能被覆盖。

| 字段 | 类型 / 默认 | 校验与语义 |
|---|---|---|
| `region` | `Optional[PixelWindow]=None` | None 全图；窗口需完全位于输入图像 |
| `score_threshold` | `float=0.5` | `[0,1]`，含等号通过 |
| `max_matches` | `int=1` | ≥0；0 表示不限数量，仍受适配层资源上限控制 |
| `max_overlap` | `float=0.3` | `[0,1)`；按变换后支持区多边形交集面积 / 较小面积做抑制 |
| `angle_min_rad,angle_max_rad` | `float=0,0` | min≤max；跨度≤2π；不自动把反向区间解释为跨零 |
| `angle_step_rad` | `float=pi/180` | >0，≤2π |
| `scale_min,scale_max` | `float=1,1` | 0<min≤max |
| `scale_step` | `float=0.02` | >0 |
| `subpixel` | `bool=True` | 位置细化；角度/尺度只在局部邻点可用时细化 |
| `coarse_fine` | `bool=True` | 两阶段；关闭时全精网格 |
| `coarse_score_threshold` | `float=0.3` | `[0,1]` |
| `coarse_angle_step_rad` | `float=5*pi/180` | ≥精角步长 |
| `coarse_scale_step` | `float=0.1` | ≥精尺度步长 |
| `fine_window_radius_px` | `float=8.0` | >0；精阶段anchor搜索窗口半径下限，单位原图px |
| `refinement_association_radius_px` | `float=3.0` | >0；相邻位姿关联同一目标的中心距离上限，单位原图px |
| `pyramid_decimation_exponent` | `int=0` | ≥0；0不降采样，n为2**n倍；不称为 levels |
| `max_coarse_template_side` | `int=400` | ≥8；仅指数>0时继续加粗直到模板长边不超过该值，实际指数写诊断；指数0禁用降采样 |
| `min_valid_fraction` | `float=1.0` | `(0,1]`；有效重叠像素 / 变体有效支持像素 |

网格为 `low+i*step≤high`，若最后一项与 high 相差大于浮点容差则补 high；重复端点去重。
角度和尺度扫描顺序固定为角度升序、尺度升序，再按 y/x 升序扫描峰值。
粗阶段产生候选区域后，精搜索使用请求的精网格与候选邻域交集，不以粗阶段位姿直接作为最终结果。
每个粗候选的精anchor窗口为原图候选anchor±max(fine_window_radius_px,2**actual_decimation_exponent)，
窗口与region相交；不同候选窗口可以合并计算，但输出候选仍各自追踪到目标。
V2 `pyramid_levels` 需根据旧实际层级行为解析成实际降采样指数和尺寸策略；无等价转换返回
`asset_incompatible`，不得直接复制数字。候选数量/内存限制在适配层检查参数后、分配前执行。

### 7.3 搜索结果与不变性

| 类型 | 字段 | 契约 |
|---|---|---|
| `SearchData` | `matches:Tuple[TemplateMatchValue,...]` | 最终排序：score 降序，其后 anchor.y、anchor.x、angle、scale 升序 |
| `TemplateMatchValue` | `center,anchor:XY`；`angle_rad,scale,score:float` | 均为待检图空间；center 是支持矩形几何中心，anchor 是教学特征 |
| 同上 | `reference_to_image,support_to_image:Mapping2D` | 映射包含旋转扩展后画布偏移，锚点由该映射算出 |
| 同上 | `support_corners_xy:Array(4,2)` | 变换后实际支持区角点，供 NMS 和绘制 |
| 同上 | `refinement:Mapping` | 必含 position/angle/scale 三个状态：`refined/discrete/no_neighbors` |

核心 score 定义为有效 mask 上的零均值归一化相关：分子 `sum((T-meanT)*(I-meanI))`，
分母为两个有效样本平方和乘积开方，截断数值误差到 `[-1,1]`。
零方差位置不是合法候选；mask 旋转用 nearest，像素旋转用 linear；
模板/图像均仅统计相同有效支持像素。阈值仍只接受 `[0,1]`。

**D07：不能按变体外接矩形过大就静默丢弃角度。** 搜索可补计算画布，但填充像素不计分，
仍按有效支持与 min_valid_fraction 判断。若某角度/尺度确实不存在合法放置，记录 unavailable，
不能把“未搜索”解释成“完整搜索未找到”。

**D08：细化必须关联同一目标。** 先固定尺度查同一目标的相邻角度，再固定角度查相邻尺度；
关联窗口以该候选中心及相邻变体预测支持区建立，只在相交且最近中心的局部峰中选择。
距离还必须≤refinement_association_radius_px；支持区比较在原图坐标进行。
存在多个等距峰或缺少左右邻点时不细化；角度/尺度的插值极值必须处于邻点区间内。
位置细化使用该目标局部相关峰；不能取其他目标的最大分数作为邻点。
重评分后重新执行阈值、排序和 NMS。

搜索诊断必填：`requested_variant_count/searched_variant_count/unavailable_variant_count:int`、
`unavailable_variants:Tuple[(angle_rad,scale,reason),...]`、`effective_region:PixelWindow`、
`actual_decimation_exponent:int`、`search_complete:bool`、`search_clip:bool`、
`clip_sides:Tuple[str,...]`、`reachable_anchor_range:Optional[(x0,y0,x1,y1)]`。
多位姿无法用单个轴对齐范围精确表示时该范围为 None，另返回逐变体范围，不能给出误导的单范围。
保留现有 `check_roi_covers_template` 用于教学配置校验；边界峰本身不能证明教学点被裁剪。

无匹配返回 invalid/no_matches，value=None；诊断区分完整无目标与角度覆盖不足。
requested_variant_count为去重后的最终精网格数量；另记pruned_variant_count，表示粗阶段已淘汰所有候选窗口的变体。
searched/unavailable/pruned在同一精网格计数、互斥且和等于requested；粗网格数量另记coarse_variant_count。
search_complete表示按声明的搜索策略完成且unavailable=0，不表示关闭了粗筛或穷举了每个像素；
诊断必须同时回传coarse_fine，不能将被粗筛剪枝的变体写成已实际扫描。
整个搜索不修改 prepared 或调用方数组；相同参数重复调用不依赖上一帧。
核心不分配业务 target_id，V1 适配为1基整数，V2 按现有稳定字符串规则生成。

## 8. 相似变换估计与控制点对齐

位置：已有 `measure/alignment.py`，B+C+D；合并旧 `SimilarityTransform` 和 V2 估计器。

```python
def estimate_similarity(
    source_points_xy: Array, destination_points_xy: Array, *,
    source: Space, destination: Space, options: SimilarityOptions
) -> Observation[SimilarityValue]: ...

def correct_alignment(
    initial_source_to_patch: Mapping2D,
    reference_controls_xy: Array, observed_controls_xy: Array, *,
    reference_patch: Space, control_ids: Tuple[str, ...], options: AlignmentOptions
) -> Observation[Mapping2D]: ...
```

| 类型 | 字段 | 默认与规则 |
|---|---|---|
| `SimilarityOptions` | `method='least_squares'`；`estimate_scale=True` | 方法 least_squares/ransac；无反射 |
| 同上 | `allow_single_translation=False`；`minimum_inliers=2` | 明确允许单点且 minimum_inliers=1 时，N=1只解纯平移，scale=1、angle=0 |
| 同上 | `ransac_threshold=3.0`；`max_residual:Optional[float]=3.0` | 阈值单位为目标长度单位；None 表示不按残差拒绝 |
| 同上 | `max_iterations=2000`；`confidence=0.99`；`random_seed=0` | iterations>0，0<confidence<1；局部随机生成器，不设置全局 OpenCV/NumPy RNG |
| `SimilarityValue` | `mapping:Mapping2D`；`scale,angle_rad:float` | mapping source→destination |
| 同上 | `residuals:Array(N,)`；`inliers:Array(N,),bool`；`rms_residual:float` | 保留所有输入对应点残差；RMS 只统计最终内点 |
| `AlignmentOptions` | `solver:SimilarityOptions`；`minimum_controls=3`；`failure_policy='fail'` | fail/rigid_fallback；求解器最少点数与业务最少控制点数分开 |

两组点 `(N,2)` 数量相同，单位相同；N=0、N>1但源或目标全重合时返回
degenerate_correspondences。N≥2并不要求点集秩为2，两点可以确定保向相似变换。
多点拟合使用现有 SVD 主体；有效尺度必须大于 `1e-12`，所有结果有限。
两组点均先转float64；n=1但未显式允许时返回insufficient_correspondences。
ransac_threshold必须>0，max_residual非空时>0，minimum_inliers不能超过N。
失败诊断也保留已计算的residuals/inliers；尚未得到模型时它们为None，不伪造全零残差。
RANSAC 用局部采样与同一最小二乘求解器，采用现有“拟合模型后检查残差”的行为；
最终 `inliers = robust_mask & (residual≤max_residual)`，若无 max_residual 则为 robust_mask。
不再因最后一次残差过滤额外重拟合；least_squares 模式要求所有点满足阈值，否则无效。
有效内点不足 minimum_inliers 返回 insufficient_inliers/excessive_residual。

控制点的 Q 定义为**参考 patch→初次裁剪所得观测 patch**，P 为**源图→初次观测 patch**。
因此修正必须为 `Q.inverse_matrix @ P.matrix`，即源图→参考 patch，修复 D03。
返回目标空间采用 reference patch 的身份；诊断含初始 P、Q、实际控制点数、残差和是否回退。
输入 controls 只包含匹配成功且通过分数阈值的配对；调用方另传成功匹配身份到诊断，不能用失败的零点凑数。
reference_controls位于reference_patch；observed_controls位于初始P.destination。
两空间均为px，control_ids数量等于对应点数、身份非空且不重复；Q的两个空间据此构造。
不足控制点或估计失败时 fail 返回无效；rigid_fallback 返回初始 P 及
`diagnostics.fallback_used=True`、`fallback_reason`，不得把回退标成成功估计了 Q。
回退映射保持P的数值矩阵，但目标身份显式设为reference_patch，表示只采用名义裁剪而未做控制点修正；
调用方必须保留fallback标记，不能用它证明控制点已经对齐。
旧策略显式保留 minimum_controls=3 和原保存回退设置；V2 现有 minimum_controls 按保存值传入。

## 9. 几何组合与显式点转换

位置：新增 `measure/geometry_ops.py`，B+C+D。所有组合接收观察对象，
先查空间，任何输入无效则返回 invalid_input，不解引用无效载荷。

```python
def circle_center(circle: Observation[CircleValue]) -> Observation[XY]: ...
def line_through_points(a: Observation[XY], b: Observation[XY]) -> Observation[LineValue]: ...
def point_distance(a: Observation[XY], b: Observation[XY]) -> Observation[float]: ...
def point_line_distance(
    point: Observation[XY], line: Observation[LineValue]
) -> Observation[float]: ...
def line_angle(a: Observation[LineValue], b: Observation[LineValue]) -> Observation[float]: ...
def point_circle_distance(
    point: Observation[XY], circle: Observation[CircleValue]
) -> Observation[float]: ...
def sampled_diameter_spread(
    circle: Observation[CircleValue], samples: Observation[Array]
) -> Observation[float]: ...
```

| 函数 | 载荷与结果契约 | 不允许照搬的行为 |
|---|---|---|
| circle_center | 输出同 frame/unit 的圆心点；无效时传播原 reason 和来源诊断 | 不能直接把含 center 的圆字典绑定为含 point 的点 |
| line_through_points | 精确通过两点；重合则 coincident_points；规范 xy 系数、真实端点 | 不能继续生成行列系数 |
| point_distance | `hypot(dx,dy)`，同长度单位 | 不允许对错空间点直接运算 |
| point_line_distance | 无限线垂距 `abs(a*x+b*y+c)` | 修复 D01；不是探针选边所用的有限线段距离 |
| line_angle | `acos(clip(abs(a1*a2+b1*b2),0,1))`；结果 `[0,pi/2]`、单位 rad、frame不变 | V2 输出 deg 在适配层转换 |
| point_circle_distance | `abs(norm(point-center)-radius)`，圆内同样非负 | 不能把圆心距当圆周距 |
| sampled_diameter_spread | 至少2点；`2*(max(norm(p-center))-min(...))`；同长度单位 | 不推断椭圆轴或标准圆度；缺样本不是0 |

samples 观察的值必须为 `(N,2)` 数组并与圆同空间。其有效标记只能表示样本是否可用，
不能把上游无效样本过滤后仍称为完整样本；允许筛选的业务必须在调用前显式形成新样本观察并记录来源。
半径和直径无需另建复杂接口：有效圆分别读取 radius 和 `2*radius`，旧 V2 circle_dimensions
通过一个适配函数组装三个独立结果；缺样本只影响 spread，不使有效半径失效。

两条旧工作流、多目标分支及 V2 geometry 全部委托上述函数。
图校验中允许 circle→point 时，运行时 read_port 和直接类型化参数入口都必须调用 circle_center。
EdgePair 的 A/B 由显式端口提供，不将整个 pair 或裸标签隐式当点。

## 10. 完整几何映射与采样后计量

位置：新增 `measure/geometry_mapping.py`，B+C+D。现有点映射、V2 SampleMapper、
result_mapping 与 alignment 中的公式合并，不能复制成第四套实现。

### 10.1 精确映射

```python
def transform_geometry(
    geometry: Observation[GeometryValue], mapping: Mapping2D
) -> Observation[GeometryValue]: ...

def map_geometry_for_display(
    geometry: Observation[GeometryValue], chain: MappingChain, *, curve_samples: int = 128
) -> Observation[GeometryValue]: ...
```

`GeometryValue = Union[XY,LineValue,CircleValue,RotatedBox,PolylineValue]`。
核心用实际 dataclass/元组类型区分，不依赖 API 的 `type` 字符串猜字段。
输入空间必须等于映射 source；输出为 destination。

| 几何 | 精确支持 | 规则 |
|---|---|---|
| 点 | 可逆相似/仿射/透视 | 齐次除法；无穷远返回 point_at_infinity |
| 无限线 | 可逆相似/仿射/透视 | `l_destination=H^-T @ l_source`，再归一化；端点同步映射；映到无穷远线则无效 |
| 圆 | 相似 | 圆心映射、半径乘s；其他映射返回 unsupported_geometry_mapping |
| 旋转矩形 | 相似 | 中心、尺寸、角同步更新；其他映射需转成闭合四点 polyline，精确接口不谎称仍为矩形 |
| 折线 | 可逆相似/仿射/透视 | 逐点映射，逐边检查分母是否跨零；保留 closed |

精确变换不复制原 fit 残差到目标空间。若有真实 samples，调用下节目标拟合重新计算；
无 samples 时原诊断置于 `source_diagnostics`，明确 source space，不产生虚构的目标残差。
显示映射允许将圆离散成闭合折线，采样角 `2*pi*i/curve_samples`，curve_samples≥16；
曲线采样只用于显示，诊断写 `approximation='display_curve'`，不得作为计量拟合的真实样本。

### 10.2 采样映射类型与入口

| 类型 | 字段 | 契约 |
|---|---|---|
| `MatrixStep` | `mapping:Mapping2D` | 相似、仿射或透视 |
| `LensStep` | `source,destination:Space`；`camera_matrix:Array(3,3)`；`distortion:Array(D,)`；`direction` | 两空间单位 px；direction 为 undistort/distort；D∈{4,5,8,12,14} |
| `MappingChain` | `steps:Tuple[Union[MatrixStep,LensStep],...]` | 至少一步；相邻 destination 必须精确等于下一 source；禁止猜单位 |
| `MetricFitValue` | `samples_xy:Array(N,2)`；`model:'points'/'line'/'circle'`；`geometry:Optional[Union[LineValue,CircleValue]]` | points 时 geometry=None，其他必须有相应有效几何 |

```python
def map_sample_points(points_xy: Array, chain: MappingChain) -> Observation[Array]: ...

def map_metric_samples(
    samples: Observation[Array], chain: MappingChain, *, target_model: str
) -> Observation[MetricFitValue]: ...
```

LensStep undistort 使用 `cv2.undistortPoints(K,D,P=K)`，保留以 K 表达的校正像素空间；
distort 将校正像素经 `K^-1` 变为射线，再 `projectPoints` 恢复畸变，不重复进行 K 换算。
只支持已声明针孔畸变模型，不接受任意 Python callback 冒充持久化映射。
新链替代旧 SampleMapper 的匿名 image_to_source，直接内存调用也需能检查中间空间。

map_sample_points 输出空间取链尾，空数组合法并返回有效空数组；触及地平线使本次整组无效，
diagnostics 给出失败点索引，不静默剔除点。
map_metric_samples 保持 N 和顺序，target_model 只接受 points/line/circle，
目标线圆使用第4节同一拟合入口；不足或退化返回无效，保留映射后真实样本到 fit 诊断。
带探针身份时 sample_probe_indices、请求/成功探针数原样透传，残差重新计算并标记目标空间。

标量比例转换必须构造显式 scale matrix（含源/目标单位），不能只改 unit 标签。
相机内参本身不能提供物理比例，缺少标量标定或平面/深度约束时拒绝 physical 计量。
平面链固定为 patch→图像源像素→去畸变校正像素→物理平面；
已经在校正画布上的点直接 canvas→plane，不再去一次畸变。
文件中的 camera_id、setup_id、sha256、采集配对身份由适配层校验并记录，不进入数值步骤。

EdgePair 映射后用映射后的 A/B 重新计算中点和宽度；这是声明的端点测量模型，
不是声称透视变换保持原中心点。边缘幅度仍为原采样灰度量，保留 sampling_space，不能乘物理比例。

## 11. 标定检测、求解与数值校验

位置：已有 `measure/measure_calibration.py`，B+C+D。检测和求解分开；
从图像检测到的点与外部人工输入点必须进入相同求解器，不能保留两个 calibrateCamera 分支。

### 11.1 公共标定数据

以下类型属于 measure_calibration 模块；其中相机参数使用 `CameraModel` 命名，
避免与现有持久化教学类 `CameraCalibration` 混淆。

| 类型 | 字段 | 契约 |
|---|---|---|
| `BoardSpec` | `columns,rows:int`；`spacing:float`；`unit:str`；`symmetric:bool=True` | columns/rows≥2，spacing>0，unit为物理长度单位 |
| `BoardObservation` | `view_id:str`；`image_space:Space`；`object_space:Space`；`image_size_wh:Tuple[int,int]` | ID非空；image单位px、object为物理单位 |
| 同上 | `image_points_xy:Array(N,2)`；`object_points_xyz:Array(N,3)`；`ordering:str` | ordering 为 row_major/explicit_correspondences；数组一一对应，不能单独排序一组 |
| `CameraModel` | `camera_id:str`；`image_space:Space`；`image_size_wh:Tuple[int,int]` | camera_id为数值模型身份标签，不检查文件哈希；image单位px |
| 同上 | `matrix:Array(3,3)`；`distortion:Array(D,)` | K和D有限；K最后行为[0,0,1]，fx/fy>0，可逆；D长度4/5/8/12/14 |
| `Pose3D` | `source_frame,destination_frame:str`；`unit:str`；`rotation:Array(3,3)`；`translation:Array(3,)` | `p_destination=R @ p_source+t`；R.T@R=I、detR=1，容差1e-6 |
| `RejectedView` | `view_id,reason:str` | 只描述观测可用性；不装文件引用 |

单个 view 至少6对不同点、两组点数量相同、图像点在[0,W)×[0,H)、物体点位于Z=0平面，
物体与图像点的中心化秩均为2。重复点、不完整对应、尺寸不符、非平面属于该 view 的拒绝原因。
非有限数和数组形状错误仍是契约异常；不能把内存对象格式错误伪装成可忽略坏图。
重复观测按有序物体点+图像点的内容判定；相同内容更换 view_id 不算独立新视图。
所有接受 view 的 object frame、unit 一致；不同姿态可以来自不同采集，但保持同一板坐标定义。

### 11.2 圆点阵检测

```python
def detect_dot_grid(
    image: Array, board: BoardSpec, options: DotDetectionOptions, *,
    image_space: Space, object_frame: str, view_id: str
) -> Observation[BoardObservation]: ...
```

`DotDetectionOptions`：`min_area=5.0`、`max_area=10000.0`（max>min>0，单位px²）、
`min_circularity=0.3`、`min_convexity=0.3`（均在(0,1]）、`color='dark'`（dark/light）、
`refine=True`、`refine_window_wh=(5,5)`、`refine_max_iterations=30`、`refine_epsilon=0.001`。
blob 阈值扫描固定1到255、步长5，禁用惯性筛选；参数进入 sampling_revision，后续改变需记录。

先转灰度，findCirclesGrid 使用对称/非对称标志，不自动再加 clustering；
refine 使用 cornerSubPix 和上述终止条件。未检测齐全则 invalid/grid_not_found。
物体点 row-major：对称为 `(col*spacing,row*spacing,0)`；
非对称为 `((2*col+row%2)*spacing,row*spacing,0)`，spacing 此时是公式基础间距，
不是同一行相邻两点的距离。排序采用检测器完整网格顺序，不能按原图 x/y 排序破坏旋转板的对应。
观察的 space 为 image_space；诊断必含 grid、refined、symmetric、detected_count。

### 11.3 相机内参求解

```python
def fit_camera_intrinsics(
    observations: Tuple[BoardObservation, ...], *, camera_id: str,
    image_space: Space, image_size_wh: Tuple[int, int], options: IntrinsicOptions
) -> Observation[IntrinsicFitValue]: ...
```

`IntrinsicOptions`：`flags:int=0`、`initial_camera:Optional[CameraModel]=None`、
`max_iterations:int=30`、`epsilon:float=2.220446049250313e-16`、`minimum_views:int=3`。
参数完整传给 OpenCV，不依赖各调用方的隐式默认。flags 仅允许针孔 calibrateCamera 标志，
实现白名单限制为 USE_INTRINSIC_GUESS、FIX_ASPECT_RATIO、FIX_PRINCIPAL_POINT、
ZERO_TANGENT_DIST、FIX_FOCAL_LENGTH、FIX_K1—K6、RATIONAL_MODEL、THIN_PRISM_MODEL、
FIX_S1_S2_S3_S4、TILTED_MODEL、FIX_TAUX_TAUY 的组合；未知位拒绝。
依赖初值的标志必须提供尺寸、空间一致的 initial_camera；不得猜 K。
minimum_views≥3；两侧现有内参路径均要求至少3个有效视图，此规则保留。

| `IntrinsicFitValue` 字段 | 类型 / 含义 |
|---|---|
| `camera` | CameraModel，纯数值结果 |
| `poses` | Tuple[Pose3D,...]，按 accepted_view_ids 顺序，板→相机，t为板单位 |
| `accepted_view_ids` | Tuple[str,...]，保持输入顺序 |
| `rejected_views` | Tuple[RejectedView,...] |
| `per_view_residuals_px` | Tuple[Array(Ni,),...]，欧氏重投影误差 |
| `solver_rms_px` | OpenCV 求解器原始 RMS，不用均值替代 |
| `mean_error_px,max_error_px,rms_error_px` | 对全部接受点的欧氏误差独立聚合，RMS=`sqrt(mean(e²))` |

有效观察的 space 为 image_space；t 的 frame/unit 由 Pose3D 自身携带。
不足视图返回 `insufficient_views`，diagnostics 含已接受与拒绝视图，不提供半成品 K。
失败与非有限解返回 solver_failed/nonfinite_solution；内参不表示已获得 physical scale。
API 的历史 `reprojection_error_px` 仍按既有约定映射为 mean_error_px，
同时保留显式 RMS 字段，不能只保留一个含混 error。

### 11.4 双目求解

```python
def fit_stereo_rig(
    pairs: Tuple[StereoObservationPair, ...], camera_a: CameraModel,
    camera_b: CameraModel, *, unit: str, options: StereoOptions
) -> Observation[StereoFitValue]: ...
```

`StereoObservationPair`：`capture_id:str`、`a,b:BoardObservation`；两者 view_id=capture_id。
`StereoOptions`：`fix_intrinsics=True`、`minimum_pairs=3`（≥3）、
`max_iterations=100`、`epsilon=1e-9`。
两相机图像尺寸相同；每对接受点数、物体坐标和排序一致，object unit=unit；
图像空间分别与 camera_a/b 一致。拒绝重复内容对，同 capture_id 出现两次为契约错误。

`StereoFitValue`：`camera_a,camera_b:CameraModel`、`a_to_b:Pose3D`、
`essential,fundamental:Array(3,3)`、`baseline:float`、`solver_rms_px:float`、
`accepted_capture_ids:Tuple[str,...]`、`rejected_pairs:Tuple[RejectedView,...]`、
`pair_diagnostics:Tuple[StereoPairDiagnostics,...]`。
`StereoPairDiagnostics` 必含 capture_id、a_own_residuals_px、b_own_residuals_px、
a_from_b_residuals_px、b_from_a_residuals_px 四组数组以及两个方向实际投影点 `(N,2)`。

固定内参使用 CALIB_FIX_INTRINSIC；精修使用 CALIB_USE_INTRINSIC_GUESS。
输入相机不得原地修改，精修结果返回新对象。A→B 的公式为 `Xb=R@Xa+t`，
E=`[t]x@R`，F=`inv(Kb).T@E@inv(Ka)`；baseline=`norm(t)>1e-9`，单位为unit。
所有输出与派生量一致，R为保向旋转。观察空间为 `Space('camera:'+camera_b.camera_id,unit)`。
V1 原终止精度1e-6与此规范1e-9不同，统一采用1e-9，并以独立标定数据验证，列为有意变化。
不要把求解首对板的平面映射当成任意高度物体的空间映射；平面产物通过第12节另行组装。

### 11.5 已有棋盘格、比例和数值校验

棋盘格不新增检测算法，保留 `CheckerboardScaleCalibration.calibrate` 与 `compute_scale`。
V2 继续将 xy 转 `(row,col)` 交给已有方法；检测 ROI 的右下端点为包含边界的
`(y+h-1,x+w-1)`，不能当半开切片传入。

新增的共享数值校验入口放在对应模块，返回 None 表示通过，否则抛契约异常：

```python
# measure/pixel_calibration.py
def validate_scalar_scale(
    scale_per_px: float, *, square_size: Optional[float] = None,
    spacing_px: Optional[float] = None
) -> None: ...

# measure/measure_calibration.py
def validate_camera_model(camera: CameraModel) -> None: ...
def validate_pose(pose: Pose3D) -> None: ...
```

scale>0且有限；square_size和spacing_px必须同时提供或同时缺省，提供时均>0，
`scale_per_px≈square_size/spacing_px`（rtol=1e-6，atol=1e-12）。
compute_scale 直接调用也拒绝少于6个、重复或共线角点；其余邻距统计复用既有实现。
K/D、R/t 的具体规则见前文；Pydantic 校验器调用同一数值函数。
资产 kind、文件版本、sha256、适用设备/分辨率身份与允许访问路径仍由 API 校验。

## 12. 平面位姿、画布、投影与拼接

位置：新增 `measure/plane_geometry.py`，B+C。求解使用第11节类型；
不导入现有 V2 PlaneCalibration Pydantic 类型。

```python
def solve_plane_pose(
    observation: BoardObservation, camera: CameraModel
) -> Observation[Pose3D]: ...

def build_plane_view(
    camera: CameraModel, board_to_camera: Pose3D, *, plane_space: Space
) -> PlaneViewModel: ...

def build_plane_canvas(
    views: Tuple[PlaneViewModel, ...], *, pixel_size: float, canvas_space: Space
) -> Observation[PlaneCanvasModel]: ...

def project_camera_points(
    points_xyz: Array, camera: CameraModel, *, source_frame: str, unit: str,
    pose: Optional[Pose3D] = None
) -> Observation[ProjectionValue]: ...

def stitch_plane_views(
    images: Tuple[Array, ...], views: Tuple[PlaneViewModel, ...],
    canvas: PlaneCanvasModel, *, options: StitchOptions
) -> Observation[StitchValue]: ...

def rebase_plane(
    views: Tuple[PlaneViewModel, ...], origin_xy: XY, *, destination_plane: Space
) -> Tuple[PlaneViewModel, ...]: ...
```

build_plane_view/build_plane_canvas 是原建议 solve_plane_pose、stitch_plane_views 之间的
必要数据准备接口，由旧 `_compute_canvas`、V2 make_plane_view/plane_parameters 提取，
不是新增求解器。rebase_plane 对应已有 origin adjustment 的纯数值部分。

### 12.1 数据类型与计算规则

| 类型/函数 | 字段或规则 |
|---|---|
| `PlaneViewModel` | camera:CameraModel；board_to_camera:Pose3D；plane_space:Space；corrected_space:Space；plane_to_corrected、corrected_to_plane:Mapping2D |
| corrected space | `Space(camera_id+':pinhole','px')`；源图先去畸变到该空间，K保持一致 |
| 平面映射 | `H_plane_to_corrected=K@[R[:,0],R[:,1],t]`，求逆得到 corrected_to_plane；不能直接应用于有畸变原图 |
| `PlaneCanvasModel` | plane_space,canvas_space:Space；pixel_size:float；bounds_xyxy:Tuple[float,float,float,float]；size_wh:Tuple[int,int]；plane_to_canvas:Mapping2D |
| `ProjectionValue` | points_xy:Array(N,2)；inside_image:Array(N,),bool；destination_space:Space |
| `StitchOptions` | blend='average'，支持 average/first/last；return_warped_views=False；interpolation='linear'（本期固定linear） |
| `StitchValue` | pixels:Array；valid_masks:Tuple[Array(H,W),bool,...]；coverage:Array(H,W),uint16；warped_views:Optional[Tuple[Array,...]]；source_to_canvas:Tuple[MappingChain,...] |

solve_plane_pose 使用已校验的一一对应、SOLVEPNP_ITERATIVE，输出 board→camera。
观察 space 为相机三维 frame 和板物理unit；全部板点必须在相机前方，Z≤0返回 board_behind_camera。
diagnostics 必含 `projected_xy:Array(N,2)`、`residuals_px:Array(N,)`、mean/max/rms_error_px。
禁止把这个像素误差误标成毫米残差。

build_plane_view 检查 pose.source_frame=plane_space.frame，t单位=plane_space.unit，
pose.destination_frame与相机身份一致。平面H奇异抛契约异常。
相机单位并不约束K；K始终是像素内参，只有空间坐标和t使用物理单位。

build_plane_canvas 支持1或2个视图，要求相同物理平面空间和pixel_size>0，canvas_space单位px。
将各图像边界中心四角经 lens undistort 再映到平面，以四角并集定义当前画布bounds；
这沿用现有四角画布政策，不宣称强畸变时能容纳每一条边界曲线。
若需强畸变全边覆盖，必须单独引入完整边界采样和正确性用例后再修改此契约。
地平线穿越返回 plane_horizon；原 V2 只映四角而未去畸变的路径在此明确修正，列为行为变化。
设bounds为x0,y0,x1,y1，画布宽高分别为
`ceil((x1-x0)/pixel_size-1e-6)+1`、`ceil((y1-y0)/pixel_size-1e-6)+1`。
plane_to_canvas矩阵为 `[[1/p,0,-x0/p],[0,1/p,-y0/p],[0,0,1]]`。
核心检查尺寸为正；适配层在分配输出前检查32768等现有尺寸/像素资源上限。

投影的points为已知三维点，不估计深度。若pose缺省，source_frame必须为该相机三维空间；
否则pose显式连接源空间→相机空间；输入点unit和pose.unit通过已声明的物理比例转为一致。
N=0返回有效空数组；任一点相机Z≤0则整组无效/behind_camera，诊断含索引。
输出点即使落在图像之外仍是有效投影，由inside_image标记。

拼接 images 与 views 位置一一对应，数量1或2，尺寸和CameraModel一致，
图像格式必须都是灰度或都是BGR。去畸变→平面→画布的链只执行一次。
掩膜独立于像素颜色，真实黑色也有效；返回每视图bool mask，coverage为其逐像素求和。
average只平均有效像素、float64累加并np.rint后转uint8；first/last依输入顺序覆盖有效像素；
无覆盖位置为0。views可用且无覆盖时返回 no_plane_coverage；返回数组不得与输入共用可写内存。

rebase_plane 把当前平面中的 origin_xy 变成新(0,0)，重新计算所有视图平移和H，
不只改元数据origin；destination_plane必须与当前平面同单位并使用新frame，调用后必须重建画布。
按 `t_new=t_old+R@[origin_x,origin_y,0]` 更新新平面→相机位姿，source_frame改为destination_plane.frame。
原点应为与views相同单位的坐标；不能把画布像素原点直接当物理原点。

## 13. 产品评定与质量汇总

位置：新增 `measure/evaluation.py`，C+D。复用 V2 evaluations 与 V1 quality 的已有计算，
质量信息与允差判定分别返回，不制造一个自动“综合质量分”。

```python
def evaluate_range(value: Observation[float], limits: LimitsSpec) -> EvaluationResult: ...

def diameter_variation(
    values: Tuple[Observation[float], ...], limits: LimitsSpec
) -> EvaluationResult: ...

def centerline_deviation(
    points: Tuple[Observation[XY], ...], limits: LimitsSpec, *, space: Space
) -> EvaluationResult: ...

def summarize_quality(
    *, fit: Optional[FitDiagnostics] = None,
    probes: Optional[Tuple[ProbeRecord, ...]] = None,
    amplitudes: Optional[Array] = None
) -> QualityMetrics: ...
```

### 13.1 评定参数与结果

| 类型 | 字段 | 契约 |
|---|---|---|
| `LimitsSpec` | `unit:str`；`minimum:Optional[float]=None`；`maximum:Optional[float]=None` | 至少一侧非空，有限，minimum≤maximum；两端包含 |
| 同上 | `invalid_policy='not_evaluated'` | not_evaluated/fail/error/ignore |
| `EvaluationResult` | `measurement:Observation[float]`；`evaluation_status:str` | status仅pass/fail/not_evaluated；无效测量也可能按策略判fail |
| 同上 | `calculation:str`；`input_count,valid_count:int`；`limits:LimitsSpec` | 完整回显计算定义、数量和已生效策略 |

range计算一个标量，calculation='inclusive_range'。variation至少两个有效直径，
计算 `max(value)-min(value)`，calculation='max_diameter_minus_min_diameter'。
centerline至少两点，计算 `max(x)-min(x)`，calculation='max_center_x_minus_min_center_x'，
不是拟合中心轴后的垂距或直线度；名称保留原语义，不能扩大产品承诺。

| invalid_policy | 有无效输入或有效数不足时的行为 |
|---|---|
| not_evaluated | measurement无效，status=not_evaluated |
| fail | measurement无效，status=fail；value仍None |
| error | 抛 ComputationContractError('evaluation.invalid_input',...) |
| ignore | 仅聚合有效项；仍不足最少数量则measurement无效、status=not_evaluated；range没有可替代项，不足时同样不评定 |

无效输入原因使用 invalid_input/insufficient_valid_samples；有有效值时按包含边界的上下限评定，
不通过允差不会让measurement.valid变成False。单个观察单位必须等于limits.unit。
centerline所有点frame必须与显式space一致，包括无效点；空列表仍从space获得明确输出空间。
diameter_variation比较标量长度，沿用V2允许不同frame的行为：单位一致即可，
不同frame的输出frame固定为aggregate；空列表也为aggregate。这是标量汇总的显式例外，
不能用来放宽点、线或中心偏差的空间检查。无效观察按策略处理，不允许NaN混入有效项。

### 13.2 质量字段

`QualityMetrics` 是普通dataclass，字段均可空：

| 字段 | 类型 | 计算来源 |
|---|---|---|
| sample_count | Optional[int] | 拟合或实际所选点总数；未知为None |
| requested_probe_count | Optional[int] | 真实请求探针数；缺失不是0 |
| successful_probe_count | Optional[int] | 至少选中一点的探针数 |
| coverage_ratio | Optional[float] | successful/requested；requested未知或为0则None |
| rms_error,mean_absolute_error,max_absolute_error | Optional[float] | FitDiagnostics的实测残差统计 |
| residual_space | Optional[Space] | 有残差时必须有空间 |
| amplitude_mean,amplitude_abs_min | Optional[float] | 所选边缘的有符号均值、绝对幅度最小值；无幅度则None |

probes与fit均提供时校验计数和sample_probe_indices一致，不一致抛contract.invalid_parameter。
成功数不得超过请求数；selection=all使样本数大于探针数是合法的，coverage仍按成功探针计算。
不在核心round；V1显示层按原小数位格式化。旧edge_amplitude_min的含义需逐调用核对，
不能直接把新的绝对最小值冒充原有符号最小值；如原契约不同保留旧显示字段并明确映射。
V1 compute_quality 由此组装旧字典：expected_edges未知时省略或null，coverage未知时省略或null，
不得继续0.0。V2 FitDiagnostics的探针计数改成可选类型，纯点拟合输出null，实际请求0个则才是0。
这些是D09有意响应变化，需要同步OpenAPI、接口参考和调用方解析测试。

## 14. 栅格绘制与数值诊断

位置：已有 `measure/viz.py`，A+B+C。保留to_bgr/to_gray等工具，新增纯返回图像入口。

```python
def rasterize_geometry(
    image: Array, overlays: Tuple[RasterPrimitive, ...], *, image_space: Space
) -> Array: ...

def render_profile(
    profile: ProfileData, *, threshold_gray: Optional[float] = None,
    size_wh: Tuple[int, int] = (480, 240)
) -> Array: ...

def render_probes(
    image: Array, probes: Tuple[ProbeRecord, ...], *, image_space: Space
) -> Array: ...
```

`RasterPrimitive`：`geometry:Observation[GeometryValue]`、`visible:bool=True`、
`selected:bool=False`、`label:Optional[str]=None`、`color_bgr:Tuple[int,int,int]=(0,255,0)`、
`thickness:int=1`。样式检查uint8颜色与正厚度，空间必须为image_space且unit=px。
无效或不可见图元跳过；不可见不表示可以混入错误空间，先检查元数据再过滤。
输出都是独立uint8 BGR图，原图和浮点几何不被修改；仅栅格绘制时round。
无限线由方程与图像边界求交；没有真实位置的错误或标量不能凭空绘制成点。

render_profile绘制gray/smoothed/gradient真实曲线与±threshold_gray线，返回像素而非PNG bytes。
render_probes消费真实ProbeRecord的区域与selected，不从拟合线重新猜探针。
V2原curve_preview返回bytes的签名可保留，但只负责调用本函数后编码PNG。
overlay ID、node/target/port筛选、JSON组装、哈希、NPZ安全读取、PNG编码和文件发布继续在V2。
显示前空间变换统一调用第10节，不能在rasterize内部偷做单位换算或复制几何公式。

## 15. 调用方迁移、资产契约与实现顺序

### 15.1 需要修改的现有调用点

| 现有位置 | 必须修改为 | 删除或保留的内容 |
|---|---|---|
| `measure1D.py`旧方法 | 第3节函数 | 保留旧元组、调试适配；删除重复平滑/寻峰/配对主体 |
| `measure2D.py`旧类 | 第4节完整探针函数 | 保留教学属性、保存、visualize；结果与_debug_info从返回诊断填充 |
| `geometric_fits.py`旧字典函数 | 与新观察入口共用私有求解主体 | 不再为API、旧类分别拟合；旧行列输出继续适配 |
| `measure_template.py` | 第5、7节 | 保留教学/持久化类；删除搜索计算和缓存临时改写；旧自定义预处理按第5节限制 |
| `measure_workflow.py` | 第8、9、13节和本节执行规则 | 删除组合公式与相似变换重复求解；保留旧对象和标签适配 |
| `multi_target_workflow.py` | 第6—10节 | 删除重复组合和不完整_map_result_to_original；每工具单独处理失败 |
| V2 `families/measurements.py` | 第3、4节 | 保留文件/资源校验和公开结果；移除probe循环、profile重算 |
| V2 `preprocessing.py`、`families/templates.py` | 第5、7节 | Pydantic转普通参数；移除_scan/_peaks/_suppress等重复计算 |
| V2 `families/images.py` | 第5、6节 | 保留编解码、FileReference和ArtifactWriter；裁剪/数值变换共享 |
| V2 `families/alignment.py`、`families/geometry.py` | 第8—10节 | 保留公开请求/响应；去掉本地估计、归一化、距离公式 |
| V2 `recipe_metrology.py`、`result_mapping.py` | 第10节 | 保留资产解析与操作到测量模型选择；移除匿名/重复数值映射和重拟合 |
| V2 `families/calibrations.py`、`families/plane_stereo.py` | 第11、12节 | 保留身份配对、文件与产物；删除OpenCV求解和重复数值验证 |
| V2 `calibration_assets.py` | 第11节数值校验 | 保留kind/格式/适用性；矩阵与比例不变量委托共享 |
| V2 `families/evaluations.py`、V1 `quality.py` | 第13节 | 保留外层适配；移除_eval/coverage重复计算 |
| V2 `result_rendering.py`、`result_diagnostics.py` | 第14节 | 保留节点选择、编码/发布；移除重复绘制和伪造探针 |
| V2 `legacy_conversion.py` | 第6、9节 | 旧格式识别留在API；ROI矩阵与圆心转换不另实现 |
| V2 `recipe_graph.py`、`recipe_execution.py` | 第9节统一端口转换规则 | 类型允许与运行绑定同步，不仅改accepts |
| GUI适配 | 使用现有measure对象 | 无须新增GUI算法；调整无效/未知质量显示 |

### 15.2 数值结果到公开字段的适配

此表定义字段来源，不扩大HTTP请求范围。共享类型不直接替换公开Pydantic schema。

| 共享字段 | V2现有表示 | V1/旧类表示与约束 |
|---|---|---|
| Observation.space.frame/unit | frame/unit | 引擎组合前保持空间元数据；最终历史响应可按接口契约省略，不能在计算前丢失 |
| valid/reason/value=None | valid=False、reason；point/value/a/b/c/center/radius为null | 按具体历史接口保留零值或省略，但保留valid=False，不能回流为有效数值 |
| 点value=(x,y) | PositionedPoint.point={x,y} | PointResult.col=x,row=y |
| 线value.coefficients | Line.a/b/c | LineResult.a/b/c直接接收规范xy系数；旧拟合字典才是params字段，不能混用 |
| 线value.start/end | Line.start/end={x,y} | LineResult.start_row/start_col/end_row/end_col分别赋值；旧拟合字典start/end才是(row,col)元组 |
| 圆value.center/radius | Circle.center={x,y}、radius | CircleResult.center_row/center_col/radius；无效圆不传radius=0给V2正半径类型 |
| diagnostics.fit.samples_xy | LineFitResult/CircleFitResult.samples | 旧edge_points/meta透传真实样本，数组与端点字段不可混用 |
| diagnostics.fit残差及计数 | FitDiagnostics同义字段，纯点拟合未知计数null | meta.rms_error、num_measures/成功探针数等明确来源；quality消费规范结构 |
| EdgeSet.edges | EdgeResult.edges，恰好一个时point端口有值 | Halcon旧row/col/amplitude列表，由单次共享结果拆开 |
| EdgePairSet.pairs | PairResult.pairs，恰好一个时A/B/center/width端口有值 | 保留十项元组及<label>_A/B身份，不能按幅度交换A/B |
| ProfileData三条曲线与sample_points_xy | ProfileResult.gray/smoothed/gradient/sample_points | last_profile/last_gradient从本次结果填充，不能成为下一次计算输入 |
| TemplateMatchValue | center/anchor、angle_degrees=rad*180/pi、scale/score及两个mapping | matched_col/row取anchor，不取支持框中心；策略特有绝对位姿继续在旧适配层转换 |
| EvaluationResult.measurement | Scalar的value/valid/frame/unit，加evaluation_status等字段 | 无V1对应评定时不虚构旧语义；工作流汇总分别记录执行与评定 |

顶层观察和嵌套line/circle的valid/reason/space必须一致；不允许顶层无效却携带旧有效圆。
LineResult.angle_rad同步按规范方向d=(-b,a)计算并归一化到[0,pi)，不能继续使用当前
`atan2(-b,a)`的旧系数假设；angle_deg由该弧度转换。该修正归入D01生产/消费约定的统一验收。
V2 geometry line-angle保持原公开deg单位，只在适配层转换一次；物理长度不乘角度。
V2 CoordinateMapping若外部同时给出matrix和inverse_matrix，适配时调用make_mapping后比较外部逆矩阵，
不一致即按原公开错误契约拒绝；内部Mapping2D始终由引擎计算逆矩阵。

### 15.3 执行契约：修复D05/D06

不新增通用DAG引擎；在现有两条工作流和V2执行循环使用相同规则。
每次run/每个target建立新的结果表，禁止把对象上一帧result当依赖输入。

建议执行记录是普通结构，归各执行器所有，不放进纯计算函数参数：

| 字段 | 类型 | 含义 |
|---|---|---|
| run_id,target_id,node_id | str | 本次执行身份；V1外部整数target_id可适配为内部字符串 |
| execution_status | succeeded/failed/blocked | 函数正常返回即succeeded，包括valid=False |
| observation | Optional[Observation] | succeeded必须有观察；failed/blocked为None |
| error_code,error_message | Optional[str] | 仅failed/blocked；不作为测量数值字段 |
| required | bool | 是否影响执行完成汇总；来自显式保存策略，现有工具默认True |

依赖failed/blocked或无效观察时，消费几何的后续节点blocked；
评定节点若契约允许处理无效值，可以按其invalid_policy运行，不一律提前阻断。
一个工具失败不阻止独立工具；外层错误不得向有类型的测量字典插入字符串_error。
汇总独立返回：`execution_complete`为所有required节点succeeded；
`measurements_valid`为所有required且产出测量的节点观察有效；
`evaluation_status`仅聚合实际评定（有fail则fail、全部存在且通过才pass，其余not_evaluated）。
没有评定节点时为not_evaluated，不能从execution_complete推断产品合格。
V1历史错误表示在最终序列化时适配，始终来自本次新记录。

### 15.4 资产与默认值

新增语义元数据不是要求V1和V2共用同一资产schema；由各存储层在现有可扩展元数据中保存：

| 字段 | 类型 | 需要记录的值 |
|---|---|---|
| computation_revision | str | 初次规范实现为shared-1；用于追溯，不选择旧算法 |
| coordinate_convention | str | xy_col_row |
| angle_convention | str | radians_image_clockwise |
| crop_origin | str | half_continuous_size |
| preprocessing | 结构化值 | 包含type及全部解析后参数，含raw通道策略 |
| support_geometry | 结构化值 | 连续box、整数pixel_size_wh、mask身份、anchor_support、reference_to_support |
| effective_parameters | 结构化值 | 实际默认值/显式值、真实金字塔指数、选边/半径策略、求解参数 |

V1旧保存路径、V2 TemplateMetadata需要按各自版本约定增加可选元数据和加载校验；
旧资产缺字段不自动认作shared-1。能证明等价时进行转换并发布新身份；
不能确定预处理、原点或金字塔语义时拒绝导入并要求重建。
V1模板搜索默认0.5/粗精开启与V2旧0.7/关闭的差异必须显式处理：
新默认统一0.5/开启；已保存的0.7/关闭继续显式传入；没有保存默认值的旧资产按可识别旧格式解析，
不可识别则asset_incompatible。不得通过“缺字段就用新默认”改变已教好的模型。

像素/支持区/预处理变化必须重建像素和掩膜、发布新哈希；矩阵元数据修正不能替代重建。
V1若需要提升manifest格式版本，独立修改其schema/加载器并保留旧格式读取测试，
不能仅在本文约定一个新版本号后就假设所有GUI能读取。
V2更新docs/v2-openapi.json及docs/v2-api-reference.md，尤其是未知探针计数null、
新增诊断、默认值和有意修复；保留现有路由和operation ID。

### 15.5 分步交付与完成条件

| 步骤 | 交付 | 前置与完成条件 |
|---|---|---|
| I01 | computation_types、构造校验、D01—D04正确性测试 | 先冻结空间/错误/几何规则；测试用独立真值 |
| I02 | 新拟合观察、geometry_ops、相似估计和完整映射 | 复用旧拟合主体；旧/V2组合路径都委托；D01—D04通过 |
| I03 | 工作流全新结果和逐节点失败 | D05/D06及成功→失败→恢复通过，不残留旧依赖 |
| I04 | 1D和完整线圆探针提取 | 先冻结探针尺寸/选边差异；所有点、顺序、探针身份可核对 |
| I05 | 预处理和模板准备/搜索 | 无可变共享缓存；D07/D08、已知旋转/多目标、资产重建通过 |
| I06 | 采样计量、质量和产品评定 | 目标空间重新拟合、D09/D10、各invalid_policy通过 |
| I07 | 检测/内参/双目/平面/拼接 | 外部观测与图像检测共用求解器；矩阵/残差/单位验证通过 |
| I08 | 绘制、资产适配、删除旧计算与更新契约 | 无API/GUI依赖进入measure；公开输出兼容与文档同步 |

每步先独立测试正确性，再比对不受修复影响的旧行为；不要先冻结已知错误输出。
同一能力的纯函数、旧类和V2适配在同一步改完，不能在最终状态保留双实现。
参数默认值在对应dataclass唯一声明；Pydantic/旧构造器要么引用规范默认，
要么显式保持历史兼容默认并在适配时传入，必须由测试约束，不能靠复制常量同步。

## 16. 最小调用示例与验收清单

下面展示拟实现后的调用形状，不声称现有仓库已经可以运行。
示例1和2可在I02直接转成单测；示例3在I04转成共享函数/V1/V2三方集成用例。

```python
import numpy as np

from measure.computation_types import CircleValue, Observation, Space, make_mapping
from measure.geometric_fits import fit_line_observation
from measure.geometry_ops import point_line_distance
from measure.geometry_mapping import transform_geometry

pixel_space = Space("camera:left/source", "px")
line = fit_line_observation(
    np.array([[0.0, 10.0], [20.0, 10.0]]), space=pixel_space
)
point = Observation(space=pixel_space, valid=True, value=(5.0, 13.0))
distance = point_line_distance(point, line)
assert distance.valid and abs(distance.value - 3.0) < 1e-9

patch_space = Space("target:1/patch", "px")
mapping = make_mapping(
    np.diag([2.0, 2.0, 1.0]), source=patch_space,
    destination=pixel_space, kind="similarity"
)
circle = Observation(
    space=patch_space, valid=True, value=CircleValue(center=(3.0, 4.0), radius=5.0)
)
mapped = transform_geometry(circle, mapping)
assert mapped.valid and mapped.value.center == (6.0, 8.0)
assert mapped.value.radius == 10.0
```

```python
from measure.measure2D import LineSpec, ProbeFitOptions, probe_and_fit_line

options = ProbeFitOptions(
    search_half_length=15, average_half_width=5,
    sigma=1, threshold=30, threshold_unit="gray", polarity="all",
    selection="nearest", interpolation="linear", border_policy="reject"
)
result = probe_and_fit_line(
    image, LineSpec(start=(20.0, 30.0), end=(200.0, 30.0), probe_count=20),
    options, space=pixel_space
)
fit = result.diagnostics["fit"]
assert fit.requested_probe_count == 20
assert 0 <= fit.successful_probe_count <= 20
assert fit.sample_count == len(fit.samples_xy)
```

### 16.1 应新增或修改的测试

首批公开 Python 契约测试已编写于 `tests/shared_computation/`，
46 个函数入口的实际覆盖、运行方法和当前 Red 基线见
[测试说明](../../tests/shared_computation/README.md)。下表仍列完整迁移验收目标；
HTTP 适配、资产兼容和其余分支以接入实现后的集成测试继续补齐。

以下文件名为拟新增；既有同主题测试优先扩展，避免重复建立同一用例。

| 测试位置 | 直接断言，不仅做快照 |
|---|---|
| `tests/test_computation_contracts.py` | 非有限/形状错误抛契约异常；无效观察无载荷；空间不匹配不被valid=False绕过；输入数组不被改写 |
| `tests/test_shared_geometry_ops.py` | D01距离3；两点成线与拟合线同约定；圆心转点无效传播；角度rad和API deg一致；并列/退化点 |
| `tests/test_shared_geometry_mapping.py` | D02平移端点残差近0、缩放半径10；透视圆精确拒绝、显示可离散；采样重拟合残差在目标空间；地平线拒绝 |
| `tests/test_shared_alignment.py` | D03位移(+3,+2)修正回原位；D04重复源点无效；单点须显式允许；RANSAC内点、回退诊断、无全局RNG副作用 |
| 既有工作流/多目标测试 | D05成功→失败不重用旧结果→恢复；D06中间工具异常后独立工具运行，依赖工具blocked；必需/可选汇总 |
| 既有1D/2D测试 | 完整配对极性与endpoints；线圆尺寸映射分别正确；first/last/all/nearest与并列；圆弧末端不重复；sigma0与边界策略 |
| `tests/test_shared_template_search.py` | 已知非零角/尺度、多目标不同位姿；D07超大变体不静默跳过；D08不串目标；mask不计填充；准备缓存跨调用不变；ROI可达性 |
| `tests/test_shared_metrology.py` | 标量/透视/畸变链；不重复去畸变；EdgePair端点映射后重算；真实样本与显示离散点不能混用 |
| `tests/test_shared_evaluation.py` | D09未知计数None、有多边探针coverage≤1；D10精确定义；允差边界；四种invalid_policy；不同frame直径汇总与中心偏差拒绝 |
| `tests/test_shared_calibration.py` | 对称/非对称物点公式；至少3独立视图；精修不修改输入K/D；A→B方向、E/F一致；重复观测/零基线/相机背后点 |
| `tests/test_shared_plane_geometry.py` | 平面H与pose一致；校正像素再投影；画布+1尺寸；真实黑色mask；三种融合；rebase真正移动几何 |
| `tests/test_shared_rendering.py` | 原图/浮点几何不被修改；错误空间拒绝；从真实探针绘制；栅格函数不编码/写文件/弹窗 |
| `tests/v2/`各操作、配方及资产测试 | 同等内存输入下三方观察一致；圆到点图校验与read_port一致；未知值null；历史默认显式解析；产物/任务生命周期仍成立 |

### 16.2 必须说明的有意变化

发布说明至少列出D01—D10，以及本规格中进一步明确的行为：
线nearest使用有限线段；未声明的新圆半径默认采用expected_band；
模板新默认0.5与粗精开启、raw模板灰度规则、金字塔参数不能直接改名；
模板完整图预处理后再裁剪支持区；
纯点拟合探针数改为未知；标定至少3视图/3对及双目终止精度统一；
画布四角先去畸变；坐标规范化可能翻转线系数与残差符号但几何不变。
无效载荷、契约异常和产品不通过分别测试，不能统一压成False或0。

### 16.3 结构验收

- `measure/` 不导入 FastAPI、Pydantic、measure_api、GUI、Flask、任务上下文或产物发布器。
- 相同能力只保留一个计算主体；旧类、单目标、多目标、V2直接/批量/配方都实际委托。
- 已有抓取函数继续复用，不新增同义路由或机器人控制代码。
- OpenAPI、中文实施文档、参数默认来源、资产兼容说明与实际行为一致。
- 不以版本开关选择两套数值实现；资产修订仅用于识别和追溯。

## 17. 源码依据

- [总体方案与D01—D10](SHARED_COMPUTATION_DESIGN.md)
- [V1/V2历史差异报告](../measure-api/references/v1-v2-divergence.md)
- [1D](../../measure/measure1D.py)、[2D](../../measure/measure2D.py)、
  [拟合](../../measure/geometric_fits.py)、[模板](../../measure/measure_template.py)、
  [裁剪](../../measure/rectify.py)、[对齐](../../measure/alignment.py)
- [相机与双目](../../measure/measure_calibration.py)、[棋盘格比例](../../measure/pixel_calibration.py)、
  [V1质量](../quality.py)
- [V2测量](families/measurements.py)、[预处理](preprocessing.py)、[模板](families/templates.py)、
  [几何](families/geometry.py)、[对齐](families/alignment.py)
- [V2标定](families/calibrations.py)、[平面与双目](families/plane_stereo.py)、
  [采样计量](recipe_metrology.py)、[显示映射](result_mapping.py)、[评定](families/evaluations.py)
- [V2绘制](result_rendering.py)、[诊断](result_diagnostics.py)、
  [原子结果类型](atomic_types.py)、[标定资产](calibration_assets.py)
