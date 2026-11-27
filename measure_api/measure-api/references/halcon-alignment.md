# 与 Halcon 模板匹配对齐：现状与差异

**日期**：2026-11-27
**分支/提交**：`step5-gui-unify`（c468755、164600c、ab208c1、d398630、6d43053）
**读者**：后续维护 `measure/`、`measure_api/`、`measure_gui/` 的同事

---

## 一、一句话概括

搜索行为按 Halcon 的语义收紧了两处（**模板必须完整落在搜索区内**、**支持对比度极性开关**），
但**没有照抄**它"建模期预渲染所有角度"的内存换时间做法。下面逐条说明抄了什么、没抄什么、为什么。

---

## 二、参数对照表

| 概念 | Halcon | `measure/` 内部 | v2 API | GUI manifest |
|---|---|---|---|---|
| 最小分数 | `MinScore` | `score_threshold` | `score_threshold` | `match_score_threshold` |
| 最大实例数 | `NumMatches` | `max_matches` | `max_matches` | `max_matches` |
| 最大重叠 | `MaxOverlap` | `max_overlap` | `overlap` | `overlap` |
| 亚像素 | `SubPixel` | `subpixel` | `subpixel` | `use_subpixel` |
| 角度范围 | `AngleStart/AngleExtent` | `angle_min_rad/angle_max_rad` | `angle_min/angle_max` | `angle_range` |
| 角度步长 | `AngleStep` | `angle_step_rad` | `angle_step` | `angle_step` |
| 尺度 | `ScaleMin/Max/Step` | `scale_*` | `scale_*` | （GUI 未暴露）|
| 金字塔层数 | `NumLevels` | `decimate_exponent` + `coarse_template_side` | `pyramid_levels` + `pyramid_max_template_size` | 同 v1 |
| 层数第二值 | `NumLevels = [start, track_to]` | `min_verify_level` | **未暴露**（v2 有自己的编排，未实现该机制）| `min_verify_level`（GUI「验证层」）|
| **对比度极性** | **`Metric`** | **`metric`** | **`metric`** | **`metric`** |
| 贪婪度 | `Greediness`（仅 `find_shape_model`）| `greediness`（shape 专用）| 未暴露 | 未暴露 |
| **匹配算法** | 独立算子（`find_ncc_model` / `find_shape_model`）| **`algorithm`**（`ncc`/`shape`）| 未暴露 | **`algorithm`**（GUI「匹配算法」）|

> `pyramid_decimate` / `pyramid_max_template_size` 是历史拼写，仍兼容读取；见
> [coarse-search-renaming.md](coarse-search-renaming.md)。

---

## 三、已对齐的行为

### 1. 模板必须完整落在搜索区内（无 padding）

Halcon 原文：

> The model is searched within those points ... in which the model **lies completely
> within the image**. ... the model will **not be found if it extends beyond the borders**.

我们之前会把画面补零到模板大小再相关，导致：

- 分数建立在**虚构的零像素**上；
- 一个大模板每个位姿都要做一次整幅相关，`瓶身直径测量`（模板 5138×3648）曾把
  5472×3648 的画面补到 10884²。

现在：放不下的位姿直接跳过并计入 `unavailable_variant_count`；整套模板都放不下时
返回稳定原因码 **`support_exceeds_search_area`**，而不是 `no_matches`
（后者会被读成"零件不在画面里"）。

### 2. `Metric` 对比度极性

| 值 | 含义 |
|---|---|
| `use_polarity`（默认）| 工件明暗关系必须与教学时一致 |
| `ignore_global_polarity` | 整体反相也能找到（取分数绝对值）|

**代价**：反相模式会同时接受该图案的"负片"，误匹配率上升。只在工件可能翻面/换光源时启用。

### 3. `MaxOverlap` 用旋转矩形计算

与 Halcon 的 `smallest_rectangle2` 口径一致，不是轴对齐框。

### 4. 亚像素靠插值，边界处退化为整像素

Halcon 原文：

> The subpixel accurate determination ... is only possible if the found instance lies
> **at least 2 pixels away from the image border** of the lowest used pyramid level.

我们的 `refine_peak_xy` 在峰值贴边时保持整数坐标，行为一致。**这不是缺陷**，
搜索大模板时出现的 6–11 px 偏差属于这一类。

### 5. `NumLevels` 第二值：不总是追到原分辨率

`min_verify_level = v` 表示"精化只做到第 v 层"，不再回到原分辨率。每跳一层，精化误差翻倍，
所以这是**拿精度换时间**，默认 0（追到原分辨率）保持既有行为。

实测（同一目标、同一位置）：

| 模型 | `min_verify_level=0` | `=1` | `=2` |
|---|---|---|---|
| 测试2 | 77.3s / off 182.4px | 17.3s / off 182.4px | **2.3s / off 182.4px** |
| 瓶颈直径圆度测量 | 39.4s | 8.1s | 1.1s（位置差 0.5px）|

### 6. 金字塔逐层淘汰

粗阶段用放宽的 `coarse_score_threshold` 选候选，精阶段用调用方的阈值判定。

---

## 四、故意不对齐的部分

| Halcon 做法 | 我们为什么不做 |
|---|---|
| 建模时**预渲染所有角度**存内存（`create_ncc_model` 明确说内存正比于角度数×模型点数）| 我们有 361 个角度位姿，模板可达 5138²；预渲染内存不可行。改为搜索时按需渲染，代价是单次慢一些 |
| 模型超出画面就"找不到" | 行为已对齐（返回 `support_exceeds_search_area`），但**没有**在建模期强制拒绝，改为教学期告警（见下）|
| `AngleStep = 'auto'` 按模型尺寸推导 | 缺本仓库数据支撑，先不做，避免又一个无依据的启发式 |
| `Greediness` 在**点数递增**过程中提前终止 | `cv2.matchTemplate` 一次性返回完整分数图，无法中途窥探。shape 算法用自研点集评分可实现层间淘汰，但尚未接出 |
| `NumLevels` 第二值在 **v2** 上 | v2 有自己的扫描编排，未接入该机制；v2 请求里不暴露该字段，避免给出一个不生效的旋钮 |

---

## 五、与现场相关的行为变化

### 教学期会告警：模板装不进图像

新增 `measure.check_support_fits_search_area()`，判据是**旋转后的包围盒**：

```
支撑长边 × (|cos θ| + |sin θ|)  ≤  min(搜索区高, 搜索区宽)
```

它不只检查角度范围两端，也检查矩形最宽/最高的临界角——一个 100×10 的支撑在 5.7°
时比在 0° 或 90° 时更高，只查两端会漏。

**GUI 在保存和运行前会弹窗告警**，但**不阻止保存**：既有模型必须仍能打开和运行。

### 受影响的模型

模板宽度等于画面宽度的模型，现在**只有 0° 附近能被搜索到**：

| 模型 | 支撑 | 画面 | ±角度全范围？ |
|---|---|---|---|
| 测试 / 测试3 / 瓶口圆度测量 / real | 明显小于画面 | ✅ | ✅ |
| 测试2 / 瓶颈直径圆度测量 / 瓶全高 / 瓶身垂直偏差 / 瓶身直径测量 | 接近或等于画面宽度 | ⚠️ | ❌ 极限角度被跳过 |

这些模型实测真实位姿都在 0°±1°，**定位结果未变**，但角度覆盖范围收窄了。
若现场确需全角度覆盖，需把模板边长收到 `min(画面高, 宽) / (|cos θ| + |sin θ|)`
（3648 画面、±15° 约为 **2978 px**），属于模型重建。

### 新的结果字段

`TemplatePoint.measure()` 的 `diagnostics` 现在包含引擎报告：

```json
{
  "search_strategy": "two_stage | shape_two_level",
  "metric": "use_polarity",
  "unavailable_variant_count": 4,
  "ambiguous": true,
  "runner_up_score": 0.9992,
  "runner_up_distance_px": 222.1,
  "rival_count": 22
}
```

**`ambiguous`** 表示画面里还有别的位置几乎同样匹配（周期图案的典型症状）。
`测试3`（圆点阵列）就是这种情况：NCC 和 shape 各选了一个格子，**两个都对**，
坐标本身不可信。调用方应把 `ambiguous=true` 当作"需要人工确认"。

---

## 六、未完成项

| 项 | 状态 |
|---|---|
| shape 的 `greediness` 接出到 manifest/v2/GUI | 引擎已支持，未暴露 |
| `AngleStep = 'auto'` | 未做 |
| 建模期预渲染 | 不做（理由见上）|
