# 模板匹配本轮变更说明（给软件同事）

**日期**：2026-11-27
**分支**：`step5-gui-unify`
**读者**：负责现场部署、模型调参、接口对接的同事

---

## 一、一句话概括

模板匹配现在有**两种算法**（`ncc` 灰度相关 / `shape` 边缘方向），参数体系向 Halcon 对齐，
搜索行为收紧了一条硬规则：**模板必须完整落在搜索区内**（不再补零）。

**现有模型默认行为不变**；需要你注意的是「模板接近画面尺寸」的那几个模型，以及**两个算法的分数不可互比**。

本轮下半程还有一件事：**v2 接口和 GUI 现在跑的是同一套搜索编排**（在此之前是两份实现，
已经在窗口半径、并列峰、单趟精化上分叉）。归一过程修掉了三个引擎缺陷，见 §三-6。

---

## 二、新增参数

| 参数 | Halcon 名 | 取值 | 默认 | GUI |
|---|---|---|---|---|
| `algorithm` | （独立算子）| `ncc` / `shape` | `ncc` | 「匹配算法」|
| `metric` | `Metric` | `use_polarity` / `ignore_global_polarity` | `use_polarity` | 「极性 Metric」|
| `min_verify_level` | `NumLevels` 第二值 | 0..5 | `0` | 「验证层 NumLevels 下限」|
| `min_contrast` | `MinContrast` | 0..255 | `0`（自动）| 暂未暴露 |

manifest 的 `matching` 段、v2 的 `TemplateSearch`、GUI 三处同名可用。

---

## 三、行为变化（重点）

### 1. 模板必须装进搜索区，否则该位姿被跳过

以前引擎会把画面**补零**到模板大小再相关。现在不加 padding：

- 放不下的位姿**直接跳过**，计入 `unavailable_variant_count`
- 整个模板都放不下时返回原因码 **`support_exceeds_search_area`**（不再是 `no_matches`）

**判据**：`模板长边 × (|cos θ| + |sin θ|) ≤ min(搜索区高, 搜索区宽)`

**影响**：模板宽度等于画面宽度的模型，现在只有 0° 附近能被搜索到。

| 模型 | 跳过位姿（共）|
|---|---|
| 瓶身直径测量 | 16 / 27 |
| 瓶全宽 | 16 |
| 测试2 | 4 |
| 瓶颈直径圆度测量 | 3 |
| 瓶身垂直偏差 | 8 |

实测这些模型的真实位姿都在 0°±1°，**定位结果未变**，但角度覆盖收窄。
若现场确需全角度覆盖，需把模板边长收到 `画面短边 / (|cos θ| + |sin θ|)`
（3648 画面、±15° 约 **2978 px**）——属于**模型重建**。

**GUI 在保存/运行前会弹窗告警**，但不阻止保存。

### 2. `Metric` 极性

| 值 | 行为 |
|---|---|
| `use_polarity`（默认）| 明暗关系必须与教学时一致 |
| `ignore_global_polarity` | 整体反相也能找到（取分数绝对值）|

代价：反相模式同时接受该图案的"负片"，误匹配率上升。只在工件可能翻面/换光源时启用。

### 3. `min_verify_level` 精度换时间

精扫不必回到原分辨率。每跳一层，精化误差翻倍。

| 模型 | =0 | =1 | =2 |
|---|---|---|---|
| 测试2 | 77.3s / 位置正确 | 17.3s / 位置相同 | **2.3s / 位置相同** |
| 瓶颈直径圆度测量 | 39.4s | 8.1s | 1.1s（差 0.5px）|

### 4. 新增歧义诊断 `ambiguous`

画面里若还有别的位置几乎同样匹配（周期图案），返回：

```json
{ "ambiguous": true, "runner_up_score": 0.9992,
  "runner_up_distance_px": 222.1, "rival_count": 22 }
```

**这时坐标不可信**，需人工确认。`测试3`（圆点阵列）就属于这类：NCC 和 shape 各选了一个格子，
**两个都对**，但没有唯一答案。

### 5. ⚠️ 两种算法的分数不可互比

| | NCC | Shape |
|---|---|---|
| 分数含义 | 归一化灰度相关系数 | 梯度方向平均余弦 |
| 典型好匹配 | 0.95 ~ 1.00 | 0.60 ~ 0.85 |

**切换算法必须重新标定 `match_score_threshold`。**

### 6. v2 与 GUI 共用一套编排；三个引擎缺陷已修

以前 `measure/template_search.py`（引擎，GUI/测量工具走它）和
`measure_api/v2/families/templates.py`（v2 接口）各写了一份「金字塔 → 粗扫 → 邻域细扫
→ 抑制」。NCC 走 v2 那份、只有 shape 走引擎，两份已经在下面三处**实际分叉**。
现在 v2 只做参数翻译，两种算法都调用引擎，分叉不可能再出现。

这几条**不是重构，是行为修正**，会带来小幅数值变化：

| # | 缺陷 | 现场表现 |
|---|---|---|
| 1 | 细窗口半径 `max(expand, 0.75×长边)`：两个约束取了最大值而非相加 | 32×24 模板只得 24 px（需 ~20 px 自身半对角线**加**余量），粗定位差一格就整块找不到 |
| 2 | 单趟扫描用 `subpixel=False` 跑完就当结果 | 请求的亚像素被**静默忽略**，位置停在整像素（实测偏 0.5 px）|
| 3 | 并列分数用非稳定排序 | 平坦分数平台（周期性件、合成图）上选中平台内任意一格，不同调用结果不同 |

修法分别是：① 半径改为 `半对角线 + fine_search_expand_px` 并夹在搜索区内；
② 单趟即最终结果，按请求精化，只有喂邻域的粗扫保持整像素；
③ 稳定排序，并列时**左上优先**，与 `argmax` / `minMaxLoc` 一致。

**对你的影响**：

- 走 GUI / 测量工具的模型：8 个仓库模型实测**位置和分数逐位未变**——这条路径的模型都开启了 `coarse_fine` / 金字塔，三条缺陷在这条路径上没有触发。
- 走 v2 的调用：命中位置可能变化 **≤ 0.5 px**，且只会出现在「之前被忽略亚像素」的
  单趟路径上；阈值不需要重标。
- v2 的 `fine_search.windows` / `neighborhood_radius` / `pyramid_scale` 现在报告引擎
  **实际使用**的窗口和比例，不再由接口层按自己的公式推算——数值可能和上一版不同，
  但它们描述的是真实行为。

---

## 四、性能实测（定位均未变）

| 模型 | 改动前 | 改动后 |
|---|---|---|
| 瓶身直径测量 | 172.5s | **10.4s** |
| 瓶全高 | 65.3s | **9.2s** |
| 瓶颈直径圆度测量 | 123.5s | **40.2s** |
| 测试2 | 137.1s | **76.9s** |
| 测试 | 5.2s | **4.3s** |

提升来自三处：去掉无意义的补零全图扫描、位姿并行（实测 6.6×）、可选的验证层降级。

---

## 五、v2 接口变化

### 新增请求字段

```python
algorithm: Literal["ncc", "shape"] = "ncc"
min_contrast: float = 0.0
metric: Literal["use_polarity", "ignore_global_polarity"] = "use_polarity"
```

### 新增响应诊断

```
algorithm, ambiguous, rival_count, runner_up_score, runner_up_distance_px,
verify_level, unavailable_variant_count,
search_plan.ignored_fields
```

### 新增失败原因码

| 原因码 | 含义 |
|---|---|
| `support_exceeds_search_area` | 模板或全部旋转变位都装不进搜索区 |

⚠️ **原因码词汇变了**：没找到目标时不再返回 `no_match`，而是引擎的
**`no_matches`**（`support_exceeds_search_area` 仍是独立原因码）。

```python
# 旧
if result["reason"] == "no_match": ...
# 新：建议只判 valid，reason 只用于日志/归类
if not result["valid"]: ...
```

如果代码里有 `reason == "no_match"` 的分支，请改成判 `valid`，或改字符串。

### shape 在 v2 的限制（显式上报，不静默）

| 情况 | 行为 |
|---|---|
| 请求多尺度（`scale_min != scale_max`）| 回退 NCC，诊断 `shape_scale_fallback` |
| 请求 `fine_search_policy="whole"` | 忽略，列入 `ignored_fields` |
| 请求 `fine_angle_policy="hierarchical"` | 忽略，列入 `ignored_fields` |
| 彩色模板 | shape 需要单通道，内部转灰度 |

---

## 五之二、测量工具「模板匹配点」也支持这些参数

`TemplateMatchPoint`（模板匹配点，点对由两个这样的点组成）**有自己的一份 `TemplatePoint`**，
参数不再跟随主面板，而是**每个工具独立配置**（因为它有自己的模板尺寸、角度范围和阈值）。

| 参数 | 是否支持 |
|---|---|
| `algorithm`（ncc/shape）| ✅ 对话框新增「匹配算法」|
| `metric`（极性）| ✅ 已有，**本轮之前是存了但没生效**，现已修复 |
| `min_verify_level` | ✅ 随 manifest 存取（对话框中暂未单列，可用默认 0）|
| `min_contrast` | ❌ 暂未暴露（shape 自动取值）|

**注意**：改动后**不需要重新教学**——参数变更会触发工具内部重建 TemplatePoint。

**小模板提醒**：测量工具默认 `template_size = 40`，实测 40px 上 shape 可用且与 NCC 同位置
（score 0.977 vs 1.000），但边缘特征更少的图案可能点数不足而返回 `template_no_texture`，
此时该点 `valid=false` 并带原因。

## 五之三、编排归一：v2 不再有自己的扫描实现

之前的状况是**同一个算法两套实现**：`measure/template_search.py` 里有完整的
「金字塔 → 粗扫 → 邻域细扫 → 抑制」编排，而 `measure_api/v2/families/templates.py`
里又写了一遍（`_grid` / `_variant` / `_scan` / `_scan_grid` / `_suppress` /
`_fine_window` / `_decimate`，约 400 行）。NCC 走 v2 自己那套，只有 shape 走引擎。
两套实现在窗口半径、并列峰处理和单趟精化上**已经出现了实际差异**（§三-6 有具体表现）。

现在 `match_template()` 只做两件事：把 v2 参数翻译成引擎参数、把引擎结果翻译回 v2
结构。NCC 与 shape 都调用 `search_support()`；`/control-alignments` 的控制点定位
也改走同一入口。v2 侧只剩 `locate_support` / `_engine_support` / `_engine_options` /
`_engine_diagnostics` 四个翻译函数，`templates.py` 从 1282 行降到 851 行。

三个引擎缺陷随归一暴露并已修复，属于**行为变化**，写在 §三-6（那里是必读区）。

### 仍然存在的已知红基线

`tests/v2/test_template_precision.py::test_long_support_localization_matches_golden`
（`whole` / `neighborhood` 两个用例）在**本次改动之前的 HEAD 上就已是失败状态**，
差异为 3e-6 ～ 1.6e-5 px，属于 golden 记录时的 OpenCV/NumPy 版本漂移，与本轮无关。
处理方式：要么按当前平台重新生成 golden 并用 `git archive` 对照验证，要么整体放宽到
1e-4 px；在此之前不要把它当作回归信号。

---

## 六、迁移指南

### 现有模型

**无需改动**：所有新参数都有默认值，默认行为与改动前一致（除 §三-1 的跳过规则）。

### 要切到 shape 的模型

1. GUI 里把「匹配算法」选成 `shape`，**重新点「创建/更新模板」并保存**
   （⚠️ 旧版本 GUI 存在 bug：选了 shape 但保存成 ncc，两个看似不同的模型其实都是 ncc）
2. 重新标定 `match_score_threshold`（见 §三-5）
3. 检查 `ambiguous` 与 `unavailable_variant_count`

### 现场排查清单

| 现象 | 先看 |
|---|---|
| 旋转件找不到 | `unavailable_variant_count` 是否很大 → 模板太大 |
| 坐标可信度 | `ambiguous` 是否为 true |
| 分数突然变低 | 是否切了算法；`metric` 是否改成反相 |
| 变慢 | `min_verify_level` 是否为 0；线程数是否被限制 |
| v2 与 GUI 结果不一致 | 两边 `search` 参数是否一致：**v2 默认 `coarse_fine=false`、`score_threshold=0.7`，GUI 默认 `true` / `0.5`**（实现已是同一套，差异只可能来自参数）|

---

## 七、已知限制

| 项 | 说明 |
|---|---|
| shape 不支持多尺度 | 回退 NCC |
| shape 不支持 `fine_search_policy` / `fine_angle_policy` | 显式列入 `ignored_fields` |
| `coarse_template_side` 与 `decimate_exponent` 语义重叠 | 两者取更激进者；`cap` 是像素值、指数是 2 的幂 |
| 建模期不预渲染所有角度 | 内存不可行（361 角度 × 大模板）|
| 不实现 `AngleStep='auto'` | 缺数据支撑 |

更详细的 Halcon 对照见 [halcon-alignment.md](halcon-alignment.md)。
