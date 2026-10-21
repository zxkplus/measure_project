# 分诊标签

技能使用五个标准分诊角色。下表将角色映射到本地问题文件使用的状态字符串。

| 标准角色 | 本地状态 | 含义 |
| --- | --- | --- |
| `needs-triage` | `needs-triage` | 需要维护者评估的问题 |
| `needs-info` | `needs-info` | 等待报告者补充信息 |
| `ready-for-agent` | `ready-for-agent` | 规格完整，可交给无人值守代理实现 |
| `ready-for-human` | `ready-for-human` | 需要人工实现 |
| `wontfix` | `wontfix` | 不予处理 |

当技能提到某个分诊角色时，在问题文件的 `Status:` 行中使用对应的本地状态。

需要调整标签名称时，修改“本地状态”列。
