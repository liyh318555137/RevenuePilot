"""销售决策的共享规则常量。

`app.py` 用它做路由，`reviewer.py` 用它做评审，两边必须引用同一份定义，
否则阈值漂移会让评审员对着一条已经不成立的规则打分。
"""

ALLOWED_DECISIONS = (
    "优先推进",
    "潜在机会",
    "建议终止",
)

HUMAN_APPROVAL_THRESHOLD = 500000

APPROVAL_STATUSES = (
    "APPROVED",
    "REJECTED",
)
