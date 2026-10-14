# 问题跟踪：本地 Markdown

本仓库的问题和规格以 Markdown 文件形式存放于 `.scratch/`。

## 约定

- 每个功能使用独立目录：`.scratch/<feature-slug>/`。
- 规格文件为 `.scratch/<feature-slug>/spec.md`。
- 实现任务按问题分别存放：
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`，编号从 `01` 开始。
- 每个问题文件顶部附近使用 `Status:` 行记录分诊状态。
  状态字符串见 `triage-labels.md`。
- 评论和对话历史追加到文件底部的 `## Comments` 标题下。

## 当技能要求“发布到问题跟踪系统”时

在 `.scratch/<feature-slug>/` 下创建对应文件；目录不存在时先创建目录。

## 当技能要求“获取相关任务”时

读取引用路径对应的文件。用户通常会直接提供文件路径或问题编号。

## Wayfinder 操作

供 `/wayfinder` 使用。每项工作包含一个地图文件，每个子任务使用独立文件。

- 地图：`.scratch/<effort>/map.md`，包含 Notes / Decisions-so-far / Fog 内容。
- 子任务：`.scratch/<effort>/issues/NN-<slug>.md`，编号从 `01` 开始。
  使用 `Type:` 行记录 `research`、`prototype`、`grilling` 或 `task`。
  Wayfinder 子任务的 `Status:` 使用 `open`、`claimed` 或 `resolved`。
- 阻塞关系：在文件顶部附近使用 `Blocked by: NN, NN` 行记录。
  只有所有列出的阻塞任务均为 `resolved` 时，该任务才解除阻塞。
- 待办选择：扫描该工作的任务文件，筛选未完成、无阻塞且未认领的任务，
  按编号选择第一个。
- 认领：开始工作前，设置 `Status: claimed` 并保存文件。
- 完成：在 `## Answer` 下追加答案，设置 `Status: resolved`，
  然后在 `map.md` 的 Decisions-so-far 中添加摘要和链接。
