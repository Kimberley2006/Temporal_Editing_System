"""预先公开的任务默认规则:身份/背景保持与可编辑量。

这些要求不是来自用户指令,而是任务本身的默认约束;
解析器输出时会在 provenance 中标注每条要求来自 instruction 还是 task_default。
"""

TASK_DEFAULTS = {
    "preserve": {
        "target_identity": True,   # 目标对象身份保持
        "background": True,        # 背景保持
    },
    "editable": ["target_path", "target_speed"],  # 规划器可调整的量
}
