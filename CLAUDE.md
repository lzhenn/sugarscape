## 开启新会话读取
@session_log.md
@.claude/persistent_context.md


## 上下文压缩规则

### Preserve in priority order:
1. Architecture decisions (NEVER summarize)
2. Modified files and their key changes
3. Current verification status (pass/fail)
4. Open TODOs and rollback notes
5. Tool outputs (can delete, keep pass/fail only)

### Operations:
- 在执行 compact 之前，必须先将本次 session 的工作摘要**追加**到 `session_log.md`，格式：
  - 时间戳yyyy-mm-dd HH:MM、做了什么、关键决策、未完成事项
- compact 之后，必须立即读取 `session_log.md` 恢复上下文

## 必须保留的关键上下文（压缩不可丢弃）
@.claude/persistent_context.md


## 执行原则
- 运用第一性原理 思考，拒绝经验主义和路径盲从，不要假设我完全清楚目标，保持审慎，从原始需求和问题出发，若目标模糊请停下和我讨论，若目标清晰但路径非最优，请直接建议更短、更低成本的办法。
- 运用苏格拉底提问法，通过持续、深入、结构化的提问，引导对方（或自己）发现矛盾、澄清概念、挑战假设，最终接近真理或更清晰的认知。
- 运用奥卡姆剃刀原理，在解释同一个现象的多种假设中，如果其他条件（解释力、预测力）相同，应该优先选择假设最少、最简单的那个。