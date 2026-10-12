"""时空规划器输入/输出的数据结构。

输入侧 TaskPackage:统一任务包,合并指令解析器与编辑依赖器的结论
(依赖器输出为主通道,其 direct_edits/events 已与解析器同构)。
输出侧 ExecutionPlan:按依赖器 edit_order 批次展开的逐段执行计划,
Phase 1 输出时间排程、关键帧、编辑段与修复方案的规划级 spec;
VACE 物化(三元的 视频段/mask/指令)在 Phase 2 适配。

已确认的设计决策(2026-10-12):
- 冲突策略:跳过冲突编辑项,先执行无冲突部分,冲突项单独反馈用户;
- 修复默认风格:最小改动(动作删除+桥接、位置保持),LLM 替代动作为可选开关。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# 减速/加速的默认系数(指令未给数值时规划器取默认;给数值则用指令值)
DEFAULT_SLOW_FACTOR = 1.25
DEFAULT_SPEED_UP_FACTOR = 0.8


# ---------------------------------------------------------------- 输入侧

@dataclass
class PlanOp:
    """归一化编辑操作(来自依赖器 direct_edits 或解析器 events)。"""

    op_id: str
    action: str                      # delete/recolor/repath/delay/slow/speed_up/arrive/stay
    object_id: Optional[str] = None
    event_id: Optional[str] = None
    region_id: Optional[str] = None
    scope_kind: str = "all"          # all | window
    scope_start: Optional[float] = None
    scope_end: Optional[float] = None
    params: Dict[str, Any] = field(default_factory=dict)
    quote: str = ""
    skipped: bool = False            # 冲突策略:跳过不执行,单独反馈
    skip_reason: str = ""


@dataclass
class PlanRepair:
    """待物化的修复要求(来自依赖器 repair_requirements)。"""

    repair_id: str
    event_id: Optional[str]
    object_id: Optional[str]
    source_op_ids: List[str]
    requirement: str
    reason: str
    window: Optional[Tuple[float, float]] = None


@dataclass
class PlanEvent:
    """timeline 既有事件的排程副本(排程会改写窗口)。"""

    event_id: str
    verb: str
    participants: List[str]
    name: str = ""
    window: Optional[Tuple[float, float]] = None
    conditions: List[dict] = field(default_factory=list)   # {kind, refs}


@dataclass
class TaskPackage:
    """统一任务包:解析器 + 依赖器结论的合并体。"""

    ops: List[PlanOp]
    events: List[PlanEvent]                    # 既有事件(排程工作副本)
    repairs: List[PlanRepair]
    batches: List[List[str]]                   # 依赖器 edit_order 的 op_ids
    preserve_object_ids: List[str]
    new_events: List[dict] = field(default_factory=list)   # 解析器同构 arrive/stay
    conflicts: List[dict] = field(default_factory=list)    # 依赖器 conflicts 原样
    uncertainties: List[dict] = field(default_factory=list)   # 依赖器 uncertainties 原样
    video_duration: Optional[float] = None
    fps: Optional[float] = None
    escalations: List[EscalatedConflict] = field(default_factory=list)   # 冲突跳过反馈


# ---------------------------------------------------------------- 输出侧

@dataclass
class ScheduledItem:
    """一个事件/操作的最终时间安排。"""

    ref: str                         # event_id 或 op_id
    kind: str                        # event | op
    start: Optional[float] = None
    end: Optional[float] = None
    note: str = ""


@dataclass
class Keyframe:
    """轨迹关键帧(规划级;bbox 缺失时仅给出时间锚点,几何在 Phase 2 补)。"""

    t: float
    object_id: str
    region_id: Optional[str] = None
    bbox: Optional[List[float]] = None
    note: str = ""


@dataclass
class SegmentSpec:
    """最小重生成段的规划级 spec(VACE 三元组物化在 Phase 2)。"""

    object_id: Optional[str]
    start: float
    end: float
    spec_type: str     # delete_mask | recolor_mask | remove_action_bridge | hold_position |
                       # new_motion | spatial_pending
    params: Dict[str, Any] = field(default_factory=dict)
    source: str = ""   # 来源 op_id / repair_id


@dataclass
class RepairResolution:
    """修复要求的落地方案(最小改动模板)。"""

    repair_id: str
    resolved_as: str                 # remove_action_bridge | hold_position | rescheduled | ...
    detail: str
    segments: List[SegmentSpec] = field(default_factory=list)


@dataclass
class EscalatedConflict:
    """规划器解决不了、反馈给用户的冲突项。"""

    conflict_id: str
    op_ids: List[str]
    kind: str
    message: str
    user_feedback: str               # 给用户的一句话反馈与建议


@dataclass
class PassPlan:
    """一个编辑批次(对应依赖器 edit_order 的一遍)。"""

    pass_no: int
    op_ids: List[str]
    scheduled: List[ScheduledItem] = field(default_factory=list)
    segments: List[SegmentSpec] = field(default_factory=list)
    keyframes: List[Keyframe] = field(default_factory=list)
    repair_resolutions: List[RepairResolution] = field(default_factory=list)


@dataclass
class VerificationCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ExecutionPlan:
    """时空规划器最终输出。"""

    status: str                                  # ok | error
    passes: List[PassPlan] = field(default_factory=list)
    edit_passes: int = 0
    skipped_ops: List[PlanOp] = field(default_factory=list)
    escalated_conflicts: List[EscalatedConflict] = field(default_factory=list)
    resolved_uncertainties: List[dict] = field(default_factory=list)
    pending_uncertainties: List[dict] = field(default_factory=list)
    violations: List[dict] = field(default_factory=list)   # 排程不可满足项
    verification: List[VerificationCheck] = field(default_factory=list)
    warnings: List[dict] = field(default_factory=list)
    provenance: Dict[str, str] = field(default_factory=dict)

    def all_checks_passed(self) -> bool:
        return all(c.passed for c in self.verification)
