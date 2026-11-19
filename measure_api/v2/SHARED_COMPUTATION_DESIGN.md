# V1、V2 与 GUI 的共享计算引擎统一设计

状态：设计方案。本文中的迁移和缺陷修复尚未实施。代码核对和针对性复现于 2026 年 11 月 18 日完成。
本次修订替代此前“以 V1 作为唯一数值基准”的提议。

## 目标与行为裁决原则

V1、V2 和 GUI 的共有计算能力必须统一调用 `measure/` 中的唯一实现。
V2 保留 HTTP 适配、公开请求校验、受控文件访问、任务生命周期、错误转换、结果序列化和产物发布职责。

V1 是兼容性参考，不是正确性的唯一标准。其现有实现存在几何约定不一致、结果映射不完整和失败处理缺陷。
仅证明结果与旧 V1 一致，不能证明计算正确。

| 行为分类 | 示例 | 迁移规则 |
|---|---|---|
| 正确且已有调用依赖的行为 | 完整边缘对筛选、探针尺寸含义、期望半径筛选 | 保留，并通过回归测试约束 |
| 表示方式或兼容约定 | 行列字段、无效值占位、资产布局 | 由适配层转换，不保留第二套数值实现 |
| 已证实缺陷或正确性不足 | 直线约定混用、映射不完整、旧结果残留、退化输入被接受 | 根据独立正确性测试修复，记录 V1/V2 的有意变化 |
| 独立且需要支持的能力 | 鲁棒估计、掩膜模板、额外阈值模式 | 在共享引擎中定义明确语义并实现一次，不能仅因 V1 缺少就删除 |

未经评审确认的行为变化阻断迁移。已记录的缺陷修复允许改变 V1 结果，受影响共享能力的所有调用方必须共同采用修复。
将保持行为的代码提取与缺陷修复分成可独立评审的变更；最终引擎中不得残留按 API 版本选择算法的开关或旧数值实现。

每项能力迁移前，应记录 V1/V2 调用路径、等价输入范围、实际生效默认值、参数含义、坐标和单位规则、
候选排序、无效条件、失败回退策略、最终采用的行为、受影响字段与资产，以及验收测试。
函数名称相近，不代表语义等价。

共享计算不意味着 V1 会话与 V2 原子操作可以互换，也不意味着两者资产格式相同。
新增 OBB 接口或定位模式属于独立范围。龙门抓取服务是第三套 HTTP 入口，不是旧版会话 API。

## 模块职责

~~~text
V1 会话 / GUI 对象 / 旧模型 ──────┐
                                ├──> measure/ ──> NumPy / OpenCV / SciPy
V2 路由 / 注册表 / 批量 / 配方 ──┘
        │
        └──> 校验、受控读写、任务调度、序列化、产物发布
~~~

共享计算接收内存图像、数组、几何值，以及普通 Python 参数或数据类，返回观察结果、所选采样点和诊断信息。
不得导入 FastAPI、Pydantic、`measure_api`、GUI、产物写入器或请求上下文。
计算函数不写文件、不弹窗、不修改调用者数组、不保存请求级状态。
现有有状态类作为教学、持久化和可视化适配层保留；搜索过程不得修改预先准备好的模板缓存。

V2 可以转换字段、行列顺序和角度单位。数值坐标映射、物理单位换算、预处理、采样、筛选、拟合、
匹配、残差和质量计算归 `measure/`。适配层校验公开数据表示，共享引擎也必须为直接 Python 调用
检查数值和语义不变量。文件身份、采集条件适用性、资源限制、幂等、取消和原子发布不属于数值计算。

同时消除 `measure/` 内部的重复计算。目前组合公式分别存在于 `measure_workflow.py`、
`multi_target_workflow.py` 的独立分支和 V2 几何操作中，三者必须共同调用同一套几何函数。
只把 V2 公式搬到另一个引擎文件，不足以实现统一。

## 共享接口契约

### 坐标、映射与单位

- 内部点坐标统一为 `(x, y) = (col, row)`。直线使用归一化系数 `a*x + b*y + c = 0`；
  系数、端点和采样点属于同一坐标系，并使用相同长度单位。
- 映射明确命名源坐标系和目标坐标系。采用列向量约定：
  `p_destination = H_source_to_destination @ p_source`。
  调用方需要时提供正反向映射，替换 `M_inv` 等含混的内部名称；旧字段别名留在适配层。
- 图像方向角使用弧度和 `atan2(dy, dx)`，x 向右、y 向下。
  OpenCV 模板旋转角和输出绝对位姿分别声明约定并显式转换。
  定位方式特有的绝对位姿转换保留在策略中，不能从历史字段名猜测正负号。
- 共享旋转裁剪保留现有局部原点约定 `x-width/2, y-height/2`，明确输出整数尺寸、插值、
  边界处理和矩阵方向。V2 原有的 `(size-1)/2` 约定需要进行资产和映射迁移，但它本身不构成数值错误。
- 观察结果使用普通 Python 类型携带坐标系和单位信息；组合计算拒绝不兼容的坐标系或单位，
  直接 Python 调用也必须遵守。换算需要显式映射或标定；角度单位与长度单位分开处理。
- 变换必须为有限值，并满足调用方要求的可逆性和模型约束。
  相似变换拒绝无法确定解的多点输入、非正尺度和奇异尺度。
  支持明确声明的单点纯平移；零组对应点无效。

`crop_and_straighten()` 实际返回原图到 patch 的映射，但其部分文档写成了 patch 到原图。
提取时保持实际裁剪行为，修正文档并提供明确的映射字段。
通过教学中心点和坐标往返测试验证方向，不能只依赖历史变量名。

### 参数、策略与默认值

每项能力在 `measure/` 中有唯一的规范参数定义和默认值来源。
旧构造器和 V2 请求模型从该来源生成默认值，或由测试证明实际生效值一致。
显式保存的配置不能因新默认值而被重新解释。

明确数据类型与通道、转灰度顺序、探针范围与取整、候选排序与并列处理、所选采样点顺序和失败行为。
共享线圆测量覆盖从探针到拟合的完整过程，诊断信息返回实际使用的探针和采样点。

算法策略属于能力参数，不属于 API 版本参数。
共享相似变换估计器可以同时支持最小二乘和 RANSAC，并复用输入校验、矩阵构造和残差计算。
普通点对应估计与控制模板匹配具有不同的输入和流程契约。

### 观察结果、执行失败与产品评定

共享观察结果包含 `valid`、稳定的原因码、坐标系与单位、可选数值载荷、采样点和诊断信息。
有效结果必须包含完整的有限数值；无效结果不提供可供计算的数值载荷。
未知质量信息表示不可用，不能替换成零质量。无搜索定位的分数保持 `None`。

V1 适配层可以按具体接口契约输出历史零值或省略字段，V2 可以保留 `null`。
二者必须从同一引擎结果获得有效性，占位值不得进入后续计算。
不能为了模仿旧数据类而将 V2 无效圆半径设为零：当前正半径校验在 `valid=False` 时也会拒绝零。

| 概念 | 含义 |
|---|---|
| 执行状态 | 操作成功、失败，或因依赖而阻断 |
| 观察有效性 | 本次执行是否产生可用测量数据 |
| 产品评定 | 有效输入通过或未通过允差，或未进行评定 |

无效测量不等于执行异常，有效测量也可能不满足产品允差。
评定上下限和无效输入策略是明确的共享参数。
对于 V2 独有、V1 没有对应项的评定能力，不应人为构造一个 V1 语义作为基准。

### 精确几何变换与采样映射后计量

当 `p_destination = H @ p_source` 时，精确直线按 `H^-T` 变换；
映射后的端点必须满足返回的直线方程。
相似变换同步缩放圆半径、长度和具有长度量纲的残差。
诊断信息保留其真实坐标系，不能把 patch 残差直接标成物理坐标残差。

一般仿射可能把圆变为椭圆，精确变换应返回支持的几何类型或明确报告不支持。
透视和非线性映射不能仅移动圆心并保留标量半径。
物理计量采用独立过程：映射实际所选采样点，在目标坐标系中拟合所声明的测量模型，
并计算目标坐标系残差，同时记录模型和坐标系。
适用时保留 V2 现有的采样点映射后重新拟合方式。

## 不能直接作为共享基准的 V1 问题

以下内容描述已核对代码中的问题，不表示已经修复。
已执行的检查是内存中的小型复现，尚未作为新的回归测试提交。
修改生产代码前先将其转成针对性测试。
某条引擎路径存在问题，不代表所有 V1 HTTP 接口都经过该路径。

| 编号 | 问题与证据 | 修复要求及验收 |
|---|---|---|
| D01 | 拟合器输出 xy 直线系数，点线组合计算却使用 `a*row+b*col+c`，两点成线也使用行优先系数。已执行复现：水平拟合线 `y=10` 与点 `(x=5,y=13)` 的距离应为 3，实际约为 5。见[拟合](../../measure/geometric_fits.py)、[组合计算](../../measure/measure_workflow.py)。 | 所有直线生产方与消费方统一遵循 xy 契约。使用非对称坐标和已知距离测试拟合线与两点成线；旧的错误距离不能作为数值基线。 |
| D02 | 结果映射修改端点却复制旧系数，移动圆心却不缩放半径。已执行复现：平移后端点代入直线方程的残差为 200；patch 到原图放大两倍，半径仍为 5，而非应有的 10。见[结果映射](../../measure/multi_target_workflow.py)。 | 变换完整几何及与坐标系相关的诊断量。覆盖平移、旋转、缩放和不支持的非相似变换；区分精确映射与采样后重新拟合。 |
| D03 | 多控制点对齐求出参考到观测的 `Q`，却组合为 `Q @ P`。三个成功的合成控制点位移 `(+3,+2)` 后，实际又施加 `(+3,+2)`，将观测点 `(13,12)` 映到 `(16,14)`，而非 `(10,10)`。见[对齐](../../measure/alignment.py)。 | 原图到参考 patch 的修正使用 `Q^-1 @ P`，保留 V2 已有的这一正确方向。测试非零平移、旋转、缩放、图像内容和坐标往返。 |
| D04 | 已执行复现：`SimilarityTransform.from_correspondences()` 将两个完全重合的源点与不同目标点接受为有效输入。见[对应点求解](../../measure/measure_workflow.py)。 | 拒绝非有限值、退化多点输入和无效尺度。保留显式定义的单点纯平移，不能把重复点接受为完整相似变换求解。 |
| D05 | `MeasurementWorkflow.measure()` 仅在 `obj.result is None` 时覆盖异常结果。已执行自定义对象复现：首帧成功、次帧抛异常，第二帧仍返回同一个旧的有效结果。见[工作流](../../measure/measure_workflow.py)。 | 每次运行创建新结果，异常覆盖本次节点结果。测试成功、失败、恢复及旧依赖阻断。该结论针对可复用工作流路径。 |
| D06 | 多目标工具循环共用一个外层 `try`，异常会跳过后续工具，并向测量字典插入字符串 `_error`。见[多目标执行](../../measure/multi_target_workflow.py)。 | 按工具记录失败，独立工具继续，依赖工具阻断。执行错误与有类型的观察结果分开；不能将截断行为复制到 V2 配方。 |
| D07 | 模板搜索静默丢弃尺寸超过搜索图像的旋转候选。已有[差异文档](../measure-api/references/v1-v2-divergence.md)记录大 ROI 角度失败，核对代码仍有该跳过逻辑。 | 定义支持区及重叠规则，报告候选丢失或角度覆盖退化。裁剪、补边方案需要有效像素掩膜或重叠规则，避免把人工填充像素当作匹配证据。测试已知非零旋转，不能只做自匹配。 |
| D08 | 角度与尺度亚像素细化按整个候选集聚合最大分数，没有关联具体目标，也未实际固定另一维度。见[模板搜索](../../measure/measure_template.py)。 | 在固定另一搜索维度的条件下，关联同一目标的相邻候选。缺少邻点时保留离散估计并给出诊断；测试不同目标位姿及角度、尺度联合搜索。 |
| D09 | V1 质量逻辑将缺失探针数视为零，并读取 `rms`、`fit_rms`，而非 `rms_error`。已执行复现：拟合残差可用，但质量输出期望数量为 0、覆盖率为 0.0，且没有 RMS；包装类也遗漏部分诊断量。见[质量](../quality.py)、[包装类](../../measure/measure_workflow.py)。 | 返回并透传真实请求探针数、成功探针数及残差。适配层保留定义明确的 RMS；未知覆盖率表示不可用，不能伪装成实测零覆盖率。 |
| D10 | `ellipticity` 实际为 `2*(max sampled radius-min sampled radius)`，不是椭圆轴指标，也不是一般意义的圆度评定。见[圆拟合](../../measure/geometric_fits.py)。 | 共享语义命名为 `sampled_diameter_spread`。旧名只作为有文档说明的序列化别名，产品允差必须明确对应指标。 |

提取时还必须处理以下风险：

- 模板金字塔与精搜索临时修改模板像素、尺寸和参数，恢复过程未统一使用 `finally` 保护。
  不能把这类请求级可变状态放进共享模板缓存。改用局部工作变量，并测试异常恢复、独立调用及并发搜索。
- 搜索 ROI 可能使教学点击点不可达，导致峰值固定在边界。
  该配置问题已有 `diagnostics.search_clip` 和 `check_roi_covers_template()`。
  校验已知教学与转换配置，保留运行时边界诊断；不能放宽距离允差，也不能认定每个边界匹配都说明教学点被裁剪。
- V1 内部的无效结果序列化也不统一：单工具结果可能有零值，多目标映射后的点、线、圆则省略数值字段。
  应按接口分别冻结表示，不存在一种通用的 V1 无效结果格式。

### 必须保留的正确现有行为

- `measure_pairs()` 先以 `transition=all, select=all` 检测，再筛选完整边缘对。
  将极性和首末选择提前传入单边缘检测会导致回归。
- 保留直线搜索深度与平均宽度的含义，在旧适配层将历史尺寸交换转换为明确的共享参数名。
- 有限线段最近边缘和期望半径筛选保留为具名策略，同时约束并列处理、顺序、圆弧步进和探针失败计数。
- 保留圆心作为点的组合能力、EdgePair 端点身份、明确声明的单点纯平移、由策略负责的位姿转换，
  以及无搜索定位的空分数。
- 保留 V2 坐标系与单位检查、显式失败策略、采样点映射后计量，以及执行完成与产品验收的区分。

## 计算归属、现状与改造清单

本节区分“新增模块”“新增共享函数入口”和“新增计算能力”，不能将三者混为一谈。
以下状态按 2026 年 11 月 18 日的仓库代码核对；建议名称及参数尚未冻结，不能视为已经可以调用的接口。
本方案没有要求新增 HTTP 路由，现有 HTTP 操作通过适配调用共享计算。

需要直接实施时，使用独立的[共享计算实施规格](SHARED_COMPUTATION_IMPLEMENTATION.md)：
其中逐项定义拟实现的函数、类型、入参出参、默认值、错误与迁移契约。
本节保留现状与改造来源索引；具体接口细节以实施规格为准，接口仍处于待实现状态。

### 分类说明

| 标记 | 当前状态 | 实施含义 |
|---|---|---|
| A：已有共享实现 | `measure/` 中已经存在可复用函数或类方法 | 保留并复用；按缺陷清单修复必要问题，不为统一名称另造一套算法 |
| B：从旧类提取 | 计算已经存在于 `measure/` 的类方法或工作流分支中，但缺少所建议的纯函数入口 | 提取现有计算，新增共享入口；旧类改为委托调用 |
| C：从接口层下沉 | 计算已经存在于 V2，或 V1 的 API 辅助层，但尚未形成合适的共享接口 | 去掉请求、Pydantic、文件和任务依赖，迁入共享模块，再由原接口适配调用 |
| D：新增或补齐契约 | 现有代码不足以满足正确性、类型或生命周期要求 | 补齐数据契约、转换、校验和缺陷修复；不代表从零重写整套算法 |

同一接口可以同时属于 B、C、D：例如相似变换需要合并两侧已有计算，并补齐退化输入校验。
“已有同名函数”仅表示某个来源中存在该名称，不表示它已经是 `measure/` 的共享接口，
也不表示参数、结果和语义可以原样照搬。

### 已有共享接口：优先复用

| 现有接口 | 源码位置 | 当前情况与处理 |
|---|---|---|
| `fit_line_points`、`fit_circle_points` | [measure/geometric_fits.py](../../measure/geometric_fits.py) | A：已是共享函数，V2 几何操作已经调用。保留拟合入口；统一其周围的坐标、结果适配与诊断，避免再次实现拟合。 |
| `crop_and_straighten` | [measure/rectify.py](../../measure/rectify.py) | A：已有内存旋转裁剪及返回映射。优先复用并修正矩阵方向文档。原建议 `crop_rotated` 是暂定命名，不强制新增同义包装。 |
| `compute_rotated_box_corners`、`build_forward_transform`、`build_inverse_transform`、`map_point_to_original`、`map_point_via_affine` | [measure/rectify.py](../../measure/rectify.py) | A：矩形顶点、矩阵构造和点映射均已有函数。按已声明方向复用；完整几何映射仍需补齐，不能据此认为 D02 已解决。 |
| `CheckerboardScaleCalibration.calibrate`、`compute_scale` | [measure/pixel_calibration.py](../../measure/pixel_calibration.py) | A：棋盘格检测与比例计算已有共享类方法，V2 已调用。先复用，必要时再提取纯校验或计算；不是新建棋盘格算法。 |
| `to_bgr`、`to_gray`、`draw_text_shadow`、`draw_legend` | [measure/viz.py](../../measure/viz.py) | A：已有图像与绘制辅助函数。保留；扩展图形绘制与诊断时复用。存在这些函数不表示当前所有绘制路径都已共享。 |
| `GantryPick.calibrate_vision`、`calibrate_offset`、`locate`、`offset_homogeneous_matrix` | [measure/pick_api.py](../../measure/pick_api.py) | A：已有抓取共享入口，V2 抓取适配已经委托；本次保持该结构。 |
| `ImageRobotCalibration.fit`、`CameraGripperOffset.from_samples`、`apply` | [measure/robot_calibration.py](../../measure/robot_calibration.py) | A：已有机器人图像标定与偏移计算，由抓取入口复用；不新增第二套标定。 |

### 拟提取或下沉的共享入口：逐项说明

下表“名称现状”针对建议名称及其归属，不把“底层已有算法”等同于“建议函数已经存在”。
暂定入口只在确实需要独立共享时开放；寻峰、网格构造和局部残差等步骤原则上保持模块私有。

| 建议共享入口与目标模块 | 名称现状 | 已有实现来源 | 改造类型与要求 |
|---|---|---|---|
| `sample_profile` → `measure1D.py` | `measure/` 尚无该函数 | [Halcon1DMeasure](../../measure/measure1D.py) 的 `extract_roi`、`get_profile`、`measure_pos`；[V2 测量](families/measurements.py)的 `extract_profile` | B+C：提取轮廓、平滑、梯度与采样坐标，消除 V2 重算；新增入口，不新增轮廓算法。 |
| `detect_edges` → `measure1D.py` | `measure/` 尚无该函数 | [Halcon1DMeasure.measure_pos](../../measure/measure1D.py) | B：提取现有边缘检测；旧方法和 V2 调用同一实现。 |
| `detect_edge_pairs` → `measure1D.py` | `measure/` 尚无该函数 | [Halcon1DMeasure.measure_pairs](../../measure/measure1D.py) | B：提取完整边缘对流程，保留先检测全部边缘再筛选完整对的规则。 |
| `probe_and_fit_line` → `measure2D.py` | `measure/` 尚无该函数 | [LineMeasureObject](../../measure/measure2D.py) 的探针生成、选边与 `measure`；[V2 measure_line](families/measurements.py) | B+C：合并完整探针到拟合流程，拟合继续调用现有 `fit_line_points`。 |
| `probe_and_fit_circle` → `measure2D.py` | `measure/` 尚无该函数 | [CircleMeasureObject](../../measure/measure2D.py) 的探针生成、半径筛选与 `measure`；[V2 measure_circle](families/measurements.py) | B+C：统一圆弧、半径范围、选边和诊断，拟合继续调用现有 `fit_circle_points`。 |
| `prepare_support` → `template_search.py` | 模块和函数均拟新增 | [TemplatePoint](../../measure/measure_template.py) 的构造与 `_compute_crop_bounds`；[V2 build_template](families/templates.py) | B+C：提取支持区、锚点、映射和数值模板准备。图像读取与资产写入留在适配层。 |
| `search_support` → `template_search.py` | 模块和函数均拟新增 | [TemplatePoint.measure 及 _match_*](../../measure/measure_template.py)；[V2 match_template、_scan、_peaks、_suppress](families/templates.py) | B+C+D：统一搜索，不另写平行匹配器；修复 D07/D08，并分离请求状态与准备缓存。 |
| `preprocess_pixels` → `preprocessing.py` | 共享模块和函数均拟新增；V2 已有不同名实现 | [旧预处理类的 __call__](../../measure/measure_template.py)；[V2 apply_preprocessing](preprocessing.py) | B+C：合并现有预设计算，统一灰度顺序和数据类型；旧类保留序列化。 |
| `crop_rotated` → `rectify.py` | 建议名不存在，但已有对应共享入口 | [crop_and_straighten](../../measure/rectify.py)；[V2 crop_image、roi_to_patch_matrix](families/images.py) | A+C：优先复用现有裁剪函数。是否改名由最终接口决定，不为名称增加无实际作用的包装。 |
| `warp_mapped` → `rectify.py` | `measure/` 尚无该函数 | [现有裁剪](../../measure/rectify.py)、[引擎对齐](../../measure/alignment.py)及 [V2 _warp_alignment](families/alignment.py) 中的图像变换 | B+C：提取已有重采样与映射计算；插值、边界和矩阵方向统一。 |
| `estimate_similarity` → `alignment.py` | V2 已有同名函数，`measure/` 尚无该函数 | [SimilarityTransform.from_correspondences](../../measure/measure_workflow.py)、[MultiPointAlignment.align](../../measure/alignment.py)、[V2 estimate_similarity、_fit_similarity](families/alignment.py) | B+C+D：合并普通与鲁棒估计，补齐 D03/D04。V2 同名函数当前依赖请求模型和上下文，不能直接改导入路径。 |
| `line_through_points` → `geometry_ops.py` | 共享模块和函数均拟新增；V2 入口名不同 | [TwoPointsLineObject 及工作流公式](../../measure/measure_workflow.py)、[多目标分支](../../measure/multi_target_workflow.py)、[V2 two_point_line](families/geometry.py) | B+C+D：消除三处组合路径的重复公式，修正 D01；最终两点成线可复用适合该语义的已有拟合计算。 |
| `point_distance`、`point_line_distance`、`line_angle`、`point_circle_distance` → `geometry_ops.py` | V2 已有同名函数，尚非共享纯函数 | [引擎组合对象](../../measure/measure_workflow.py)、[多目标组合分支](../../measure/multi_target_workflow.py)、[V2 geometry](families/geometry.py) | B+C+D：提取并合并现有公式，统一 xy、单位和无效性；点线距离需修正 D01。 |
| `sampled_diameter_spread` → `geometry_ops.py` | 已有结果字段及计算，但没有该共享函数 | [fit_circle_points 的 ellipticity](../../measure/geometric_fits.py)、[V2 circle_dimensions](families/geometry.py) | A+C+D：复用已有采样半径极差计算，统一名称和含义，处理 D10；不新增圆度算法。 |
| `detect_dot_grid` → `measure_calibration.py` | V2 已有同名函数，`measure/` 尚无该顶层函数 | [CameraCalibration.calibrate、_make_blob_detector 与 StereoRigCalibration._detect_board_centers](../../measure/measure_calibration.py)、[V2 detect_dot_grid](families/calibrations.py) | B+C：从现有检测中提取，统一圆点参数与排序，支持后续内存求解。 |
| `fit_camera_intrinsics` → `measure_calibration.py` | `measure/` 尚无该函数 | [CameraCalibration.calibrate](../../measure/measure_calibration.py)、[V2 compute_camera](families/calibrations.py) | B+C：提取现有 OpenCV 内参求解与残差；不重写内参算法。 |
| `fit_stereo_rig` → `measure_calibration.py` | `measure/` 尚无该函数 | [StereoRigCalibration.calibrate_stereo](../../measure/measure_calibration.py)、[V2 compute_stereo](families/plane_stereo.py) | B+C：合并观测点准备之后的双目求解与残差，读图和资产发布留在外部。 |
| `solve_plane_pose` → `plane_geometry.py` | 模块和函数均拟新增 | [StereoRigCalibration._solve_board_pose](../../measure/measure_calibration.py)、[V2 pose、make_plane_view、plane_parameters](families/plane_stereo.py) | B+C：整理现有位姿和平面计算，接口纯化，不新增一套位姿算法。 |
| `project_camera_points` → `plane_geometry.py` | 模块和函数均拟新增；两侧已有投影能力 | [StereoRigCalibration.project_to_camera_image](../../measure/measure_calibration.py)、[V2 project_camera、project_points](families/plane_stereo.py) | B+C：按坐标与畸变契约提取公共计算，保留不同投影输入的明确含义。 |
| `stitch_plane_views` → `plane_geometry.py` | 模块和函数均拟新增 | [StereoRigCalibration._compute_canvas、stitch](../../measure/measure_calibration.py)、[V2 stitch_images](families/plane_stereo.py) | B+C：提取画布、变换和融合计算；输入图像和输出产物通过内存接口衔接。 |
| `transform_geometry` → `geometry_mapping.py` | V2 已有同名函数，目标共享模块拟新增 | [引擎 _map_result_to_original](../../measure/multi_target_workflow.py)、[V2 transform_geometry](families/alignment.py)、[map_geometry](result_mapping.py) | B+C+D：共享完整几何变换并修复 D02，不能把存在同名函数视为已经完整支持所有映射。 |
| `map_metric_samples` 及目标模型拟合 → `geometry_mapping.py` | 目标模块和入口拟新增，现有实现是类方法 | [V2 SampleMapper.map_points、convert](recipe_metrology.py) | C：下沉所选采样点映射、畸变修正和目标坐标系拟合；保留 V2 的现有正确计量过程。 |
| `evaluate_range` → `evaluation.py` | V2 已有同名函数，共享模块拟新增 | [V2 evaluate_range、_evaluate](families/evaluations.py) | C：下沉范围评定与无效输入策略；HTTP 和异常响应适配仍在 V2。 |
| `diameter_variation`、`centerline_deviation` → `evaluation.py` | 建议共享名称不存在；V2 已有不同名入口 | [V2 evaluate_variation、evaluate_deviation](families/evaluations.py) | C：提取现有直径极差和中心 x 极差，明确其定义，不套用不存在的 V1 对应语义。 |
| 质量计算入口，名称待定 → `evaluation.py` | API 层已有 `compute_quality`，没有规范的共享入口 | [V1 compute_quality](../quality.py)、[V2 fit_diagnostics](families/geometry.py) | C+D：共享已测残差与探针质量规则，补齐 D09 的数据透传与不可用语义。 |
| 标定数值校验入口，名称待定 → 对应标定模块 | 规则已有，但分散在类与校验器中 | [引擎标定](../../measure/measure_calibration.py)、[V2 ScalarParameters.consistent_scale、CameraParameters.check_camera](calibration_assets.py)、[平面双目校验器](families/plane_stereo.py) | B+C+D：提取并补齐数值不变量；文件身份、适用性和格式校验不下沉。 |
| 数值诊断与栅格绘制入口，名称待定 → `viz.py` | 共享辅助函数已有，完整路径尚未统一 | [引擎绘制](../../measure/viz.py)、[V2 rasterize](result_rendering.py)、[curve_preview、probe_provenance](result_diagnostics.py) | A+B+C：复用已有辅助函数，提取数值绘制；探针消费真实测量结果，编码和文件发布留在适配层。 |
| 旧 ROI 几何转换 → `rectify.py`，不强制新增名字 | 共享矩阵函数已有，V2 转换中仍有数值步骤 | [rectify](../../measure/rectify.py)、[V2 convert_legacy](legacy_conversion.py) | A+C：旧模型转换调用现有共享矩阵函数，只在无法表达现有几何时扩展；不另保留一套矩阵构造。 |

### 需要新增或补齐的契约与行为

这些内容没有现成的完整共享契约，但已有局部数据类型或规则可供整合。
类型名称、辅助函数名称与最终文件归属在实现时确定，不在此预先声明不存在的已发布接口。

| 项目 | 现有基础 | 实际新增或补齐的部分 |
|---|---|---|
| 普通 Python 观察结果、几何和参数契约 | [GeometricResult 等旧类型](../../measure/measure_workflow.py)、[V2 Pydantic 观察类型](atomic_types.py)及各请求参数 | D：建立不依赖 Pydantic 的统一规则及默认值来源；不是直接把 V2 请求模型搬入引擎。 |
| 圆心转点的纯转换 | 旧 `CircleResult.row/col` 别名、[V2 accepts](recipe_graph.py) 与 [read_port](recipe_execution.py) | D：补齐图校验与运行时一致的显式转换，传播有效性、原因、坐标系和单位。 |
| 完整几何一致性与退化校验 | 上述已有求解、映射和校验代码 | D：修复 D01—D04；这些行为不能只靠移动函数或增加包装获得。 |
| 每次运行的全新结果与逐节点失败处理 | 两条旧工作流和 V2 配方已有执行循环 | D：修复 D05/D06，明确独立继续、依赖阻断和必需/可选汇总；这是执行语义改造，不要求新增通用 DAG 引擎。 |
| 可信搜索诊断、局部细化及质量语义 | 现有模板诊断、峰值细化和质量字段 | D：修复 D07—D10；不能声称旧实现原样提取后已经满足新契约。 |
| 资产采样约定与计算修订追溯 | 两侧已有模板和标定资产元数据 | D：补齐约定、等价转换、必要的资产重建与新身份；不保留按修订版本切换的旧算法。 |

### 模块文件的新增情况

| 文件状态 | 模块 | 说明 |
|---|---|---|
| 已存在，扩展或提取内部计算 | `measure1D.py`、`measure2D.py`、`geometric_fits.py`、`rectify.py`、`alignment.py`、`pixel_calibration.py`、`measure_calibration.py`、`viz.py` | 不新建同义模块；现有类和纯函数按上表保留、提取或修复。 |
| 已存在，继续复用 | `pick_api.py`、`robot_calibration.py` | 抓取相关共享计算已存在，不因本方案新增第二套入口。 |
| 拟新增 | `measure/computation_types.py` | 实施规格补充的公共类型、空间、观察结果及映射契约，不放业务算法。 |
| 拟新增 | `measure/template_search.py` | 收拢已有模板准备与搜索计算，同时修复缺陷。 |
| 拟新增 | `measure/preprocessing.py` | 收拢旧预处理类和 V2 的现有计算。 |
| 拟新增 | `measure/geometry_ops.py` | 合并两条引擎工作流与 V2 几何组合。 |
| 拟新增 | `measure/plane_geometry.py` | 收拢现有平面、投影、画布及拼接计算。 |
| 拟新增 | `measure/geometry_mapping.py` | 合并完整几何映射与采样点计量。 |
| 拟新增 | `measure/evaluation.py` | 收拢质量和产品评定，保持两者语义独立。 |

以上共 7 个拟新增模块文件，其中 6 个计算模块沿用本方案拆分，1 个公共类型模块由实施规格补充；
不代表新增 7 套业务算法。
函数入口是否新增、从哪里迁入、是否需要补齐行为，应分别按前述清单判断。
已有类方法和 V2 路由作为兼容适配保留；迁移完成后删除旧计算主体，
不能让新函数与旧类方法或 V2 辅助函数各保留一份公式。

## 迁移中的具体行为裁决

| 领域 | 最终采用的规则 |
|---|---|
| 模板默认值 | 共有能力的新请求使用共享分数阈值 0.5，并默认启用粗精两阶段搜索。保留显式保存的配置。金字塔参数按实际降采样和搜索含义转换，不能直接重命名数值；没有等价映射时拒绝或重建。 |
| 模板支持区 | 点击方形、矩形、掩膜和锚点使用同一实现。验证方形无掩膜输入的等价性；根据独立位姿测试修复 D07/D08，不能冻结错误分数和角度。 |
| 预处理 | 每种预设明确转灰度顺序和数据类型，并只有一个实现。共有 Sobel 预设经一致性检查后使用现有 V1 表达式。额外阈值模式保留为明确的共享能力。 |
| 裁剪约定 | 使用已声明的 `size/2` 原点及共享插值和边界规则。迁移支持区、锚点映射，必要时重建像素；测试奇偶尺寸、边界和映射链。 |
| 线圆选边 | 共有默认策略为有限线段最近边缘、期望半径最近边缘及期望半径范围。将 V1 尺寸转换为共享搜索深度和平均宽度。支持的 first、last、all 策略保持显式，不得静默重映射已保存选择。 |
| 对应点估计 | 普通共享估计使用最小二乘，明确支持单点纯平移。保留 RANSAC，并声明内点、重新求解、残差拒绝和阈值规则。拒绝退化输入，不按 API 版本分支。 |
| 控制点对齐 | 匹配控制点，估计参考到观测的 `Q`，应用 `Q^-1 @ P`。旧适配层明确保留三个成功控制点的要求。最少控制点数量和失败回退策略区别于求解器最少点数；保留显式保存的策略，新共有对齐默认失败，除非请求明确允许回退。 |
| 标定 | 分离检测和求解，记录圆点参数、排序、标志、停止条件和残差定义。默认值需要标定正确性证据，不能仅因 V1 更早就优先。外部观测点只跳过检测，使用相同参数进入同一求解器。 |
| 点端口 | 圆转点必须提取圆心，并保留有效性、原因、坐标系和单位。图校验、运行时绑定、支持该输入的直接类型化调用及旧模型转换共同遵守。 |
| 无效观察结果 | 共享有效性规则，不提供不可用数值载荷。保留接口特定的 V1 占位或省略方式及 V2 空值语义，未知分数和质量保持不可用。 |
| 评定 | 按已声明含义共享计算。V2 独有能力进入引擎；执行异常、测量无效和产品不通过仍然分开。 |

等价输入包含适配后的模板像素与支持区、探针几何、实际生效参数、标定、映射链和评定策略。
以不同方式准备输入后调用同一个函数，不能证明 V1/V2 行为一致。

## 适配层、工作流与资产

只修改 `recipe_graph.accepts()` 不能启用圆转点。
当前 `read_port()` 原样返回圆载荷，而 `PositionedPoint` 需要 `point`，不是 `center` 和 `radius`，
直接转换会校验失败。定义唯一的纯转换，并在图校验和运行时适配中复用其规则。
无效圆转换为没有可用坐标的无效点。

EdgePair 发布明确的 A/B 端点，模型转换和图连接保留这些身份。
不能含混地将裸标签绑定为点，缺失端口和占位值不能成为有效依赖。

每次运行和每个目标创建新结果。异常覆盖当前节点结果，独立工具继续，依赖工具阻断；
必需与可选工具的汇总遵循显式工作流策略。
两条引擎工作流与 V2 配方都遵守这些规则，但共享数值计算不要求引入新的通用 DAG 框架。
确需保留的旧错误字段仅在响应格式化时生成。

记录两代接口的变化：默认值、重命名或移除字段、修正后的数值、失败语义和受影响调用方。
实施时同步更新 V2 OpenAPI、接口参考及相关 V1 文档。

预先准备的资产记录格式、预处理和数据类型、支持区与锚点映射、采样约定、实际搜索参数和计算修订版本。
修订版本用于追溯，不是保留旧求解器的开关。
导入适配层转换明确支持的旧表示；含义不明时返回具体诊断并拒绝导入。

裁剪像素、掩膜或预处理改变时，应生成新模板资产，并使用教学图和检测图重新验证。
仅修改元数据不能修复模板像素差异。发布新身份和哈希，不得静默修改已固定身份的资产。
哈希和发布由 API 适配层负责。V1/V2 继续使用各自存储格式；本次只更新受影响的既有转换路径，
不承诺任意工程格式的双向转换。

## 实施顺序

1. **确定行为并建立证据。** 记录 V1/V2 实际参数和解码后的数值基线。
   将 D01—D10 转为针对性测试，区分已执行复现和代码核对发现的风险。
   分开管理不变行为、有意修复和仅表示方式变化。
2. **先统一几何与观察结果。** 确定 xy 直线、映射方向、无效载荷规则、全新执行结果及依赖转换。
   分成可评审步骤修复 D01—D06，避免后续提取冻结这些问题；保留必要的接口表示适配。
3. **提取完整测量过程。** 从灰度轮廓、边缘、边缘对及线圆探针开始。
   保留正确成对筛选与选边，返回实际诊断。
   每项能力同时覆盖两条引擎工作流，以及 V2 直接、批量和配方路径。
4. **迁移预处理与模板。** 分开准备数据与请求状态，统一搜索，修复候选覆盖和目标局部细化。
   测试已知旋转、联合尺度、多目标、ROI 可达性及失败恢复。
   重建受影响资产，不保留第二套匹配器。
5. **迁移标定、计量、评定和绘制。** 统一求解参数和残差，保留采样点映射后重新拟合，
   修复质量信息透传，将数值资产校验和 ROI 转换迁入对应模块。
   保留 V2 文件访问和任务生命周期。
6. **删除重复代码并检查归属。** 删除已迁移的 V2 扫描、寻峰、相似变换辅助函数、重复 OpenCV 求解、
   探针循环及引擎组合公式。更新契约、生成文档、迁移说明和发布说明。
   不因某策略原先来自 V2 就删除合理的共享能力。

## 验证与验收

| 验证层次 | 必须具备的证据 |
|---|---|
| 提取一致性 | 不受修复影响的轮廓、带符号边缘和边缘对、所选采样点、顺序、像素和参数保持等价。记录平台或 OpenCV 浮点容差，不能放宽容差掩盖算法变化。 |
| 独立正确性 | 点线距离符合已知真值，映射端点满足直线方程，半径正确缩放，控制点回到参考位置，退化求解无效，失败帧不返回旧有效结果。 |
| 模板正确性 | 覆盖非零角度与尺度、大支持区、多种目标位姿、ROI 裁剪、细化邻点不可用及异常恢复；自匹配不够。 |
| 标定与计量 | 使用已知合成变换与标定及代表性图像，检查目标坐标系中的残差与物理几何，验证完整的所选采样点映射链。 |
| 跨调用方一致性 | 共享函数、V1 对象、V1 多目标、V2 直接或注册表调用、批量和配方，在等价输入与参数下产生等价观察结果；直线系数符号不同时比较几何等价性。 |
| 失败与依赖语义 | 圆心转换在校验和运行时都有效，无效值不能作为零测量参与计算，缺失端点阻断依赖，独立工具继续，必需与可选汇总规则明确。 |
| 公开契约兼容性 | V1 各接口占位或省略、V2 空值与模型、默认值、显式策略、资产导入或重建、清单及生命周期符合修订后契约。 |
| 架构 | 核心不导入 API、GUI 或持久化层，已迁移调用方委托指定模块，API 和引擎调用方都不保留重复公式。 |

保留 V2 的 GUI 与旧接口导入阻断测试。
API 中的 OpenCV 调用限制为明确列出的文件编解码适配，数值 OpenCV、SciPy 和 NumPy 运算进入引擎模块；
NumPy 序列化仍然允许。

库函数调用检查只能作为防线，手写算术、间接调用和 Pydantic 直线归一化校验器同样可能重复计算。
数值校验器应调用共享规则，不能重写公式。
核对计算归属，并通过共享接口验证 API 和旧引擎调用方都实际委托了共享实现。

最终验收要求：每项能力只有一个实现；等价输入产生等价观察结果；
不受影响的基线保持不变；经过独立验证的缺陷修复被所有受影响调用方采用；
兼容性和资产变化有明确说明；不保留按 API 版本切换的数值分支。

## 代码依据

- [V1/V2 差异报告](../measure-api/references/v1-v2-divergence.md)
- [一维测量](../../measure/measure1D.py)、[二维测量](../../measure/measure2D.py)、
  [拟合](../../measure/geometric_fits.py)、[模板](../../measure/measure_template.py)
- [对象工作流](../../measure/measure_workflow.py)、
  [多目标工作流](../../measure/multi_target_workflow.py)、
  [对齐](../../measure/alignment.py)、[裁剪矫正](../../measure/rectify.py)
- [V1 适配](../project.py)、[质量](../quality.py)、
  [V2 观察结果](atomic_types.py)、[配方图](recipe_graph.py)、
  [配方执行](recipe_execution.py)、[采样计量](recipe_metrology.py)
- [V2 测量](families/measurements.py)、[几何](families/geometry.py)、
  [对齐](families/alignment.py)、[模板](families/templates.py)、
  [标定](families/calibrations.py)、[平面双目](families/plane_stereo.py)、
  [评定](families/evaluations.py)
