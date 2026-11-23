# 模板匹配粗搜索参数改名：给后续开发的说明

**日期**：2026-11-23
**影响版本**：main（提交 46a3ca7、b41df88、ff7ce7d 及本次）
**读者**：后续维护 `measure/`、`measure_api/`、`measure_gui/` 的同事

---

## 一、一句话概括

两个粗搜索参数在**内部改了名**，但 **v2 API 的公开字段保持不变**。你在不同层看到的字段名可能不一样，这是**有意设计**，不是笔误。

---

## 二、字段对照表

| 概念 | `measure/` 内部 | GUI manifest / config.yaml | **v2 API / Recipe（公开契约）** |
|---|---|---|---|
| 降采样指数 | `decimate_exponent` | `decimate_exponent` | **`pyramid_levels`** |
| 粗阶段模板边长 | `coarse_template_side` | `coarse_template_side` | **`pyramid_max_template_size`** |

**旧拼写（仅存在于历史资产，已兼容读取）**：`pyramid_decimate`、`pyramid_max_template_size`（GUI manifest 里曾用这两个名字）。

> 注意：`pyramid_decimate`（GUI/v1）和 `pyramid_levels`（v2）**本来就是同一个量**，只是两代 API 拼写不同，数值等价，从未做过换算。

---

## 三、为什么这样做

### 为什么要改名

原名有两个缺陷，容易被误判为可删除的冗余参数：

1. **`pyramid_` 前缀**暗示它只与「金字塔」有关。实际上它和另一个参数**共同决定同一个降采样因子**，取两者中更激进的那个。
2. **`max_template_size`** 容易和 `template_size`（模板自身边长）混淆——前者是「缩完之后的上限」，后者是「模板本来多大」。

新名 `coarse_template_side` 直接表达「粗阶段模板边长上限」。

### 为什么 v2 不改

`pyramid_levels` / `pyramid_max_template_size` 是**已发布的 API 契约**：
- `docs/v2-openapi.json`
- `measure_api/v2/SHARED_COMPUTATION_IMPLEMENTATION.md`
- `measure_api/measure-api/references/api_reference.md`

改它属于**破坏性变更**，会让所有外部调用方返回 422（v2 是 `StrictModel`，`extra_forbidden`）。所以内部统一、外部保持。

---

## 四、改代码时必须知道的四件事

### 1. v2 与内部之间只有一处转换

`measure_api/v2/families/templates.py::request_pyramid_factor()` 是**唯一的双拼写交汇处**。
读取 `options.pyramid_levels` / `options.pyramid_max_template_size`，转成引擎的 `SearchOptions`。

**如果你要再加一个粗搜索参数，请在这里做映射，不要在别处复制一份。**

### 2. 别用 `_pyramid_factor` 当本地函数名

v2 的 `timing` 模块会按函数名自动装饰（`@timed`）。本地函数如果和引擎的 `_pyramid_factor` 同名，会被错误装饰，报 `TypeError: missing 1 required positional argument`。
现有适配器叫 `request_pyramid_factor`，就是为了避开这个坑。

### 3. legacy 转换的输入输出方向相反

```
输入 v1 资产  → pyramid_decimate / pyramid_max_template_size      （旧拼写，白名单必须接受）
输出 recipe   → pyramid_levels / pyramid_max_template_size        （v2 公开名）
```

`legacy_conversion.py` 里 `PARAMETERS` 与 `sections["matching"]` 的**白名单是校验输入的**，必须接受旧拼写；随便改成新名会报 `Unknown legacy calculation field`。

同理，`diagnostic()` 的 `field_path` 要指向**源资产里的真实字段**（`matching.pyramid_decimate`），不是内部新名。

### 4. 测试里「输入参数」和「输出断言」同名不同向

```python
teach_template(coarse_template_side=cap)        # 输入：内部新名
assert search["pyramid_max_template_size"] ...  # 输出：v2 公开名
```

批量替换字段名时极易把断言也改掉，导致 `KeyError`。

---

## 五、兼容性处理

| 场景 | 如何兼容 |
|---|---|
| 旧 GUI manifest（`pyramid_*`） | `measure/measure_template.py::pyramid_setting()` 双拼写读取 |
| 旧 v1 资产转 v2 | `legacy_conversion.py::_pyramid_field()` 双拼写读取 |
| GUI 模板导入 v2 | `template_imports.py` 白名单同时接受新旧两套 |
| config.yaml | 只读新名（配置由我们自己维护） |

**新增字段时的约定**：读取端一律用辅助函数兼容旧拼写，写入端一律写新名。

---

## 六、尚未完成的事

1. **manifest 里目前两种拼写并存**：新保存的模型写新名，旧模型仍是旧名。计划在 manifest 升级时做一次性转换，目前未实施。
2. **`pyramid_max_template_size` 的语义偏「上限」**：模板小于该值时不会降采样（`factor` 会被夹到 1.0）。跨尺寸场景下它实际起的是「目标边长」作用。改名 `coarse_template_side` 已朝这个方向靠，但 `min(1.0, ...)` 的夹紧逻辑未改。

---

## 七、相关背景

完整的四步重构计划（算子下沉 → 编排压平 → 候选数可配 → 配置面合并）见
`docs/coarse-search-unification-plan.md`（注意：`docs/` 目录被 `.gitignore` 排除，该文件不在版本库中）。

前三个提交：
- `46a3ca7` 算子下沉（亚像素 / 峰值 / 抑制 / 位姿网格各收敛到一处）
- `b41df88` 编排压平（`pyramid` 与 `coarse_fine` 成为正交旋钮）
- `ff7ce7d` 候选数可配 + `score_stage` 标记