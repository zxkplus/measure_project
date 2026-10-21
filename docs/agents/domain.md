# 领域文档

本仓库采用单一上下文的领域文档布局。

## 探索代码前的读取规则

- 阅读仓库根目录的 `GLOSSARY.md`。
- 阅读 `docs/adr/` 下与当前工作领域相关的架构决策记录。

如果这些文件不存在，直接继续工作，无需提示缺失或提前建议创建。
`/domain-modeling` 技能会在术语或决策得到明确结论后按需创建这些文档，
包括通过 `/grill-with-docs` 或 `/improve-codebase-architecture` 调用时。

## 文件布局

```text
/
├── GLOSSARY.md
└── docs/
    └── adr/
        ├── 0001-<decision-slug>.md
        └── 0002-<decision-slug>.md
```

## 使用词汇表中的术语

在问题标题、重构提案、假设或测试名称中命名领域概念时，
使用 `GLOSSARY.md` 定义的术语，避免使用词汇表明确排除的同义词。

如果需要的概念尚未收录，先判断项目是否实际使用该概念。
如果确实存在术语缺口，记录下来供 `/domain-modeling` 处理。

## 明确指出 ADR 冲突

如果提案与现有 ADR 冲突，明确标出对应 ADR，
并解释为何应重新评估该决策。
