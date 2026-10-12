"""编辑依赖器:分析一个修改如何影响其他事件/修改,输出直接编辑、受影响事件、
修复要求、冲突与不确定项,以及编辑先后顺序(供规划器消费)。

输入与指令解析器一致:(instruction, scene);scene 可选附带 timeline
(检测跟踪模块提供的既有视频事件,含时间窗与成立条件)。
"""

from .analyzer import EditDependencyAnalyzer
from .llm import LLMError, MockQwenClient, QwenVLClient
from .schemas import (AffectedEvent, Conflict, EditBatch, EditOp, PreserveRequirement,
                      RepairRequirement, Scope, StatedConstraint, TimelineEvent,
                      Uncertainty, parse_timeline)

__all__ = ["EditDependencyAnalyzer", "QwenVLClient", "MockQwenClient", "LLMError",
           "EditOp", "Scope", "PreserveRequirement", "StatedConstraint",
           "TimelineEvent", "parse_timeline", "AffectedEvent", "RepairRequirement",
           "Conflict", "Uncertainty", "EditBatch"]
