"""时空规划器:把解析器 + 依赖器的结论展开为逐段编辑执行计划。

输入:编辑依赖器输出(主通道,含 direct_edits/edit_order/repairs/conflicts)
     或解析器输出(简单 arrive/stay 指令);
输出:ExecutionPlan —— 按批次的时间排程、关键帧、编辑段、修复方案、
     冲突跳过反馈与验证结论。VACE 物化在 Phase 2。
"""

from .planner import SpatioTemporalPlanner
from .schemas import (DEFAULT_SLOW_FACTOR, DEFAULT_SPEED_UP_FACTOR, ExecutionPlan,
                      Keyframe, PassPlan, PlanOp, RepairResolution, ScheduledItem,
                      SegmentSpec, TaskPackage, VerificationCheck)
from .task_package import build_task_package

__all__ = ["SpatioTemporalPlanner", "build_task_package", "TaskPackage", "PlanOp",
           "SegmentSpec", "Keyframe", "PassPlan", "ExecutionPlan", "ScheduledItem",
           "RepairResolution", "VerificationCheck",
           "DEFAULT_SLOW_FACTOR", "DEFAULT_SPEED_UP_FACTOR"]
