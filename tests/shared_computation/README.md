# 共享计算接口契约测试

测试依据：[共享计算引擎实施规格](../../measure_api/v2/SHARED_COMPUTATION_IMPLEMENTATION.md)。
边界采用该文档定义的公开 Python 函数、数据类型，以及需要修正的现有公开工作流。
本轮只编写测试，不实现新入口，也不修改生产算法。

## 运行方式

在仓库根目录执行：

```bash
.venv/bin/python -m pytest tests/shared_computation -q --tb=short
```

收集测试，不执行计算：

```bash
.venv/bin/python -m pytest tests/shared_computation --collect-only -q
```

实现时按单个行为推进，例如先修正水平线距离，再运行对应现有适配器回归：

```bash
.venv/bin/python -m pytest \
  tests/shared_computation/test_shared_geometry.py::test_d01_fitted_horizontal_line_is_three_pixels_from_point \
  tests/shared_computation/test_shared_legacy_regressions.py::test_d01_existing_workflow_uses_xy_coefficients_for_point_line_distance \
  -q --tb=short
```

既有契约的通过基线：

```bash
.venv/bin/python -m pytest \
  tests/shared_computation/test_existing_fits.py \
  tests/shared_computation/test_shared_legacy_regressions.py::test_existing_workflow_preserves_three_four_five_point_distance \
  tests/shared_computation/test_shared_calibration.py::test_existing_scalar_calibration_preserves_row_column_input_contract \
  tests/test_1d_measure.py::TestHalcon1DMeasure::test_measure_pairs_polarity_and_select_are_pair_level \
  -q --tb=short
```

## 当前 Red 基线

2026-11-18：本目录共 **183 个参数化后的测试用例**，**6 个通过、177 个失败**，无收集错误。
另行运行上述既有契约基线，**7 个全部通过**，其中包含目录外已有的完整边缘对测试。

| 分类 | 数量 | 含义 |
|---|---:|---|
| 已有契约通过 | 6 | 旧线圆拟合、退化异常、两点距离、棋盘格比例计算 |
| 新契约尚未实现 | 168 | 缺少模块、函数或类型；在执行测试时明确失败 |
| 现有行为回归失败 | 9 | 已经通过现有公开入口复现的问题，详见下表 |

新接口尚未实现的用例，其数值断言还没有执行到；不能将这些失败当作新算法已经得到验证。
实现时应逐条完成 Red→Green，不能仅新增空函数让导入通过，也不能以改变独立真值迁就错误实现。
本目录没有 skip、xfail 或“接口不存在就返回”的通过分支，直接运行全量 pytest 也会包含这些 Red 用例。

本次完整输出保存在被 git 忽略的目录：

- `tests/output/shared_computation/red.log`：详细失败输出。
- `tests/output/shared_computation/red.xml`：JUnit 结果，可供工具按用例读取。

### 现有实现已复现的问题

位置：[test_shared_legacy_regressions.py](test_shared_legacy_regressions.py)。

| 问题 | 用例数 | 独立真值与当前错误 |
|---|---:|---|
| D01 点线距离 | 1 | `y=10` 到 `(5,13)` 应为 3，目前约为 5 |
| D01 两点成线 | 1 | 水平线真实端点必须满足 xy 方程，目前残差为 -10 |
| D01 线方向角 | 1 | 水平端点、规范 xy 系数对应 0°，目前返回 -90° |
| D04 相似变换退化 | 1 | 重复源点不能确定变换，目前仍标为有效 |
| D05 帧隔离 | 3 | 成功→异常→恢复：异常帧的点、依赖距离、工作流有效性均复用了旧成功结果 |
| D06 节点隔离 | 2 | 单列探针运行时异常后，后续独立工具未执行，测量字典混入字符串 `_error` |

D05 使用 `MeasureObject` 公开扩展基类提供确定性点源，通过
`MeasurementWorkflow.add/measure/get_result/is_valid` 观察结果；不访问其私有依赖表。
失败帧和恢复帧都实际执行，独立点仍应有效，恢复后的距离应为 7。

D06 通过 `MultiTargetWorkflow.add_measurement/measure/remove_measurement` 构造和恢复场景：
旧入口接受 `length1=0.5`，实际 ROI 只有一列，梯度计算抛异常。
未来允许在注册前拒绝此配置，但必须不留下半注册节点；如果已接受配置，
执行失败必须隔离，独立工具仍有本次结果，错误不能作为字符串塞入测量字典。
这个用例覆盖独立节点和字典类型，完整的执行记录、required 汇总及多目标依赖阻断仍需接入实现后补充。

## 46 个函数入口的覆盖索引

这里的“新增”指新增共享层的公开入口；部分算法或同名函数已存在于 V1/V2 适配层，
其来源分类仍以实施规格第 1.1 节为准。测试调用的是规格中的共享层归属。
下表覆盖该文档所有 `def` 函数签名，同时测试若干必要的数据构造契约。

| 模块 / 入口 | 测试文件 | 主要行为与真值 |
|---|---|---|
| `computation_types.make_mapping` | [contracts](test_shared_contracts.py) | 方向、逆矩阵、奇异与非有限矩阵、拒绝伪相似和反射 |
| `measure1D.sample_profile` | [measurements](test_shared_measurements.py) | 灰度坡面 30…69、sigma=0、源坐标、非整数半宽 |
| `measure1D.detect_edges` | 同上 | 正极性 first/last/all、fraction 与 gray 阈值等价、空白和越界 |
| `measure1D.detect_edge_pairs` | 同上 | 完整配对后筛极性、首末对、endpoints 宽度 70 |
| `geometric_fits.fit_line_observation` | [geometry](test_shared_geometry.py) | 水平线、带符号残差 ±1、RMS=1、退化及非有限输入 |
| `geometric_fits.fit_circle_observation` | 同上 | 圆心 `(3,4)`、半径 5、零残差、共线退化 |
| `measure2D.probe_and_fit_line` | [measurements](test_shared_measurements.py) | 合成阶跃 y=59.5、7 个探针、搜索深度和平均宽度、失败诊断 |
| `measure2D.probe_and_fit_circle` | 同上 | 合成半径 40、径向筛选、16 个样本身份、整圆不重复末探针 |
| `preprocessing.preprocess_pixels` | [pixels](test_shared_pixels.py) | 五类预处理、五种阈值模式、Sobel=1020、BGR 灰度、输入不变 |
| `rectify.crop_region` | 同上 | 连续 size/2 中心、源到 patch 矩阵、裁剪像素 |
| `rectify.warp_mapped` | 同上 | 复制边界的像素值与真实有效掩膜分离 |
| `rectify.legacy_roi_to_box` | 同上 | row/col 仅交换一次，角度 90°→π/2 |
| `template_search.prepare_support` | [templates](test_shared_templates.py) | 锚点与中心分离、只读缓存、常量模板无效、拒绝 preserve |
| `template_search.search_support` | 同上 | 已知平移、无限匹配数、无旧结果、D07 大旋转、D08 多目标不同角度/尺度 |
| `alignment.estimate_similarity` | [alignment](test_shared_alignment.py) | 90°、2 倍、平移 `(3,4)`、重复点、显式单点策略、RANSAC 和 NumPy RNG |
| `alignment.correct_alignment` | 同上 | 观测偏移 `(+3,+2)` 修正回参考位置、fail 与 rigid_fallback |
| `geometry_ops.circle_center` | [geometry](test_shared_geometry.py) | 圆心、空间和无效原因传播 |
| `geometry_ops.line_through_points` | 同上 | xy 系数、真实端点、重合无效 |
| `geometry_ops.point_distance` | 同上 | 3-4-5、无效不变零、无效也不能绕过空间检查 |
| `geometry_ops.point_line_distance` | 同上 | D01 水平无限线距离 3 |
| `geometry_ops.line_angle` | 同上 | 无向小夹角为 0 或 π/2，输出 rad |
| `geometry_ops.point_circle_distance` | 同上 | 圆内、圆上、圆外均为到圆周的非负距离 |
| `geometry_ops.sampled_diameter_spread` | 同上 | 采样半径 3/5/4→直径极差 4；不足样本未知 |
| `geometry_mapping.transform_geometry` | [mapping](test_shared_mapping.py) | D02 线系数/端点一起平移、圆半径缩放、旋转框、源空间检查 |
| `geometry_mapping.map_geometry_for_display` | 同上 | 非相似圆显示成闭合折线，声明 display_curve |
| `geometry_mapping.map_sample_points` | 同上 | 顺序、空间接续、地平线、空数组、镜头畸变正反方向、两步顺序 |
| `geometry_mapping.map_metric_samples` | 同上 | points/line/circle 三种模式，目标单位重拟合与 RMS=0.5 |
| `measure_calibration.detect_dot_grid` | [calibration](test_shared_calibration.py) | 合成 6×5 圆点阵、row-major 物点、空白图无网格 |
| `measure_calibration.fit_camera_intrinsics` | 同上 | 5 个已知板位姿恢复 fx=320、fy=300，复制视图不算独立视图 |
| `measure_calibration.fit_stereo_rig` | 同上 | A→B 平移 `(-10,0,0)`、基线 10、E 矩阵、输入 K 不变、少于三对 |
| `pixel_calibration.validate_scalar_scale` | 同上 | 5mm/10px→0.5；零、负、NaN、缺配对和比例冲突 |
| `measure_calibration.validate_camera_model` | 同上 | 有效针孔模型、零焦距拒绝 |
| `measure_calibration.validate_pose` | 同上 | 正旋转有效，反射不能作为旋转 |
| `plane_geometry.solve_plane_pose` | [plane](test_shared_plane.py) | 已知板到相机平移 `(-10,-8,100)`、重投影误差 |
| `plane_geometry.build_plane_view` | 同上 | 平面原点→`(160,120)`；`(192,150)`→平面 `(10,10)` |
| `plane_geometry.build_plane_canvas` | 同上 | 像素中心边界、0.5mm/px 时尺寸 `(201,161)` |
| `plane_geometry.project_camera_points` | 同上 | 已知三维点、图外有效、背后整组无效、单位换算、缺 pose 拒绝 |
| `plane_geometry.stitch_plane_views` | 同上 | 真实黑色参与覆盖，average/first/last 分别为 50/0/100 |
| `plane_geometry.rebase_plane` | 同上 | 新原点 `(10,20)` 改变位姿平移和投影，不只改元数据 |
| `evaluation.evaluate_range` | [evaluation](test_shared_evaluation.py) | 闭区间、四种 invalid_policy、产品 fail 与观察有效性分离 |
| `evaluation.diameter_variation` | 同上 | 极差、跨 frame 同单位汇总、有效数量、空集合 |
| `evaluation.centerline_deviation` | 同上 | x 极差定义、空集合、禁止混 frame |
| `evaluation.summarize_quality` | 同上 | 未知探针数为 None、RMS=1、多边不增加成功探针数、幅度统计 |
| `viz.rasterize_geometry` | [rendering](test_shared_rendering.py) | 独立 BGR、浮点几何不变、不可见/无效不绘制、空间先检查 |
| `viz.render_profile` | 同上 | 返回指定尺寸 BGR 像素，不返回编码数据 |
| `viz.render_probes` | 同上 | 使用真实探针记录、产生彩色标记、原图不变 |

`test_shared_contracts.py` 还覆盖 Space 非空、Observation 有效性与载荷一致、
有限数、合法零、LineValue 归一化和 CircleValue 半径约束。
`test_existing_fits.py` 单独固定保留接口 `fit_line_points/fit_circle_points` 的历史表示和异常行为。

## 测试数据和限制

- 所有图像和标定观测在内存生成，不需要 `data/sample/`、相机、模型目录、HTTP 服务或 GUI。
- 几何使用手算距离和矩阵；轮廓使用灰度坡面/条带；标定输入由预先指定的 K、R、t 正向投影生成，
  求解结果与这些已知参数比较。没有调用被测求解器生成期望结果。
- 模板姿态用固定随机纹理和 OpenCV 正向仿射生成；断言留有栅格重采样容差，不要求旋转后 score=1。
- 不 mock 内部拟合器、检测器或执行私有方法；`conftest.py` 只负责在用例执行时解析公开符号，
  缺少目标模块/符号时调用 `pytest.fail`，依赖模块自身的导入错误照常暴露。
- 这是首批共享计算契约测试，不等同于全部 HTTP 集成与资产兼容验收。
  适配器接入后仍需增加 V1/V2 同输入结果一致性、OpenAPI/null 表示、圆到点端口、
  模板资产重建和历史默认值迁移测试；点阵非对称、双目精修和零基线等分支也需继续扩展。
