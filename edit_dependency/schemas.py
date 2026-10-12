"""编辑依赖器输入/输出的数据结构。

输入与指令解析器保持一致:(instruction, scene)。
scene 在解析器使用的 objects/regions/video 之外,可选附带 timeline 字段:
既有视频事件(由检测跟踪模块提供),每个事件带参与对象、时间窗与成立条件,
依赖传播的"状态门控"依据这些条件在事件发生时刻的覆盖情况判断。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------- 输入侧:场景扩展

# 事件成立条件的种类;refs 引用对象 ID 或(对 after_end_of)事件 ID。
# 传播矩阵(action × condition)见 propagation.py。
CONDITION_KINDS = (
    "requires_present",   # 事件成立需要 refs 中对象在场
    "supported_by",       # 事件成立需要 refs 中对象提供支撑
    "held_by",            # 事件成立需要 refs 中对象被持有
    "spatial_contact",    # 事件成立需要与 refs 中对象保持空间接触/相交
    "after_end_of",       # 事件成立需要 refs 中事件已结束(时序约束)
)


@dataclass
class TimelineCondition:
    kind: str
    refs: List[str] = field(default_factory=list)
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "TimelineCondition":
        kind = data.get("kind", "")
        if kind not in CONDITION_KINDS:
            raise ValueError("未知条件种类「%s」,支持:%s" % (kind, "/".join(CONDITION_KINDS)))
        refs = list(data.get("refs", []))
        if not isinstance(refs, list) or not all(isinstance(r, str) for r in refs):
            raise ValueError("条件 refs 需为对象/事件 ID 列表:%s" % refs)
        return cls(kind=kind, refs=refs, note=data.get("note", ""))

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "refs": list(self.refs)}
        if self.note:
            d["note"] = self.note
        return d


@dataclass
class TimelineEvent:
    """既有视频事件:依赖分析的传播对象。window 为 [start, end] 秒,缺失视为未知。"""

    event_id: str
    verb: str                                # appear / rest_on / pick_up / drink / walk / collide ...
    participants: List[str] = field(default_factory=list)   # 参与对象 ID
    window: Optional[Tuple[float, float]] = None
    conditions: List[TimelineCondition] = field(default_factory=list)
    name: str = ""                           # 可读名(如「A」「人拿起杯子」)
    text: str = ""                           # 检测跟踪模块给的自然语言描述

    @classmethod
    def from_dict(cls, data: dict) -> "TimelineEvent":
        window = data.get("window")
        if window is not None:
            if (not isinstance(window, (list, tuple)) or len(window) != 2
                    or not all(isinstance(v, (int, float)) for v in window)):
                raise ValueError("事件 %s 的 window 需为 [start, end] 秒" % data.get("event_id"))
            if not window[0] < window[1]:
                raise ValueError("事件 %s 的 window 需要 start < end" % data.get("event_id"))
            window = (float(window[0]), float(window[1]))
        return cls(
            event_id=data["event_id"],
            verb=data.get("verb", ""),
            participants=list(data.get("participants", [])),
            window=window,
            conditions=[TimelineCondition.from_dict(c) for c in data.get("conditions", [])],
            name=data.get("name", ""),
            text=data.get("text", ""),
        )

    def to_dict(self) -> dict:
        d = {"event_id": self.event_id, "verb": self.verb,
             "participants": list(self.participants)}
        if self.window is not None:
            d["window"] = [self.window[0], self.window[1]]
        if self.conditions:
            d["conditions"] = [c.to_dict() for c in self.conditions]
        if self.name:
            d["name"] = self.name
        if self.text:
            d["text"] = self.text
        return d

    def conditions_referencing(self, object_id: str) -> List[TimelineCondition]:
        """成立条件中引用了某对象的条件(时序约束 after_end_of 引用事件,不算)。"""
        return [c for c in self.conditions
                if c.kind != "after_end_of" and object_id in c.refs]


def parse_timeline(data: Optional[list]) -> List[TimelineEvent]:
    """解析并校验 scene.timeline:ID 唯一、参与对象/条件引用可查(由调用方查场景表)。"""
    if not data:
        return []
    if not isinstance(data, list):
        raise ValueError("timeline 需为事件列表")
    events = [TimelineEvent.from_dict(d) for d in data]
    ids = [e.event_id for e in events]
    if len(ids) != len(set(ids)):
        raise ValueError("timeline 中存在重复的事件 ID:%s" % ids)
    return events


# ---------------------------------------------------------------- 输入侧:LLM 提取产物

# 依赖器支持的编辑操作;arrive/stay 与解析器的事件结构保持一致
SUPPORTED_ACTIONS = (
    "delete",    # 删除对象(整段或指定窗口)
    "recolor",   # 改变外观颜色
    "repath",    # 更换对象路径
    "delay",     # 事件延迟(作用于既有事件)
    "slow",      # 减慢(对象速度或既有事件)
    "speed_up",  # 加快
    "arrive",    # 到达区域(与解析器同构)
    "stay",      # 停留(与解析器同构)
)


@dataclass
class Scope:
    """编辑作用范围:整段(all)或时间窗(window)。"""

    kind: str = "all"                        # "all" | "window"
    start_seconds: Optional[float] = None
    end_seconds: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "Scope":
        if not data or data.get("kind", "all") == "all":
            return cls(kind="all")
        start, end = data.get("start_seconds"), data.get("end_seconds")
        if start is None or end is None or not (start < end):
            raise ValueError("window 范围需要 start_seconds < end_seconds:%s" % data)
        return cls(kind="window", start_seconds=float(start), end_seconds=float(end))

    def to_dict(self) -> dict:
        if self.kind == "all":
            return {"kind": "all"}
        return {"kind": "window",
                "start_seconds": self.start_seconds, "end_seconds": self.end_seconds}

    def windows(self, duration: Optional[float]) -> List[Tuple[float, float]]:
        """展开为具体时间窗列表;all 且时长未知时返回空(由受影响区间补齐)。"""
        if self.kind == "window":
            return [(self.start_seconds, self.end_seconds)]
        if duration is not None:
            return [(0.0, duration)]
        return []

    def overlaps_window(self, window: Optional[Tuple[float, float]],
                        duration: Optional[float]) -> bool:
        """范围是否与某时间窗相交;范围或事件窗未知时保守返回 True(交给判定)。"""
        if window is None:
            return True
        mine = self.windows(duration)
        if not mine:
            return True
        return any(w[0] < window[1] and window[0] < w[1] for w in mine)

    def contains_scope(self, other: "Scope", duration: Optional[float]) -> bool:
        """本范围是否覆盖另一范围(用于「整段删除 vs 后半段变色」类冲突)。"""
        if self.kind == "all":
            return True
        if other.kind == "all":
            return duration is not None and self.start_seconds <= 0 and self.end_seconds >= duration
        return self.start_seconds <= other.start_seconds and self.end_seconds >= other.end_seconds


@dataclass
class EditOp:
    """一条用户直接编辑指令。对象类操作填 object_id,事件类操作(delay 等)填 event_id。"""

    op_id: str
    action: str                              # SUPPORTED_ACTIONS 之一
    object_id: Optional[str] = None
    event_id: Optional[str] = None           # 目标既有事件(timeline 中的 event_id)
    region_id: Optional[str] = None          # arrive 目标区域
    scope: Scope = field(default_factory=Scope)
    params: Dict[str, Any] = field(default_factory=dict)  # seconds/color/duration_seconds 等
    quote: str = ""                          # 对应的原文子句

    def to_dict(self) -> dict:
        d = {"op_id": self.op_id, "action": self.action}
        if self.object_id is not None:
            d["object_id"] = self.object_id
        if self.event_id is not None:
            d["event_id"] = self.event_id
        if self.region_id is not None:
            d["region_id"] = self.region_id
        d["scope"] = self.scope.to_dict()
        if self.params:
            d["params"] = dict(self.params)
        if self.quote:
            d["quote"] = self.quote
        return d


@dataclass
class PreserveRequirement:
    """保持要求:「X 保持不变」;aspects 为 appearance(外观)/ behavior(行为/轨迹)。"""

    object_id: str
    aspects: List[str] = field(default_factory=lambda: ["behavior", "appearance"])
    quote: str = ""

    def to_dict(self) -> dict:
        d = {"object_id": self.object_id, "aspects": list(self.aspects)}
        if self.quote:
            d["quote"] = self.quote
        return d


@dataclass
class StatedConstraint:
    """用户指令中声明的时序/空间约束,如「B 必须在 A 结束后开始」。"""

    kind: str                                # starts_after_end_of | starts_with ...
    subject_event_id: str
    ref_event_id: str
    quote: str = ""

    def to_dict(self) -> dict:
        d = {"kind": self.kind,
             "subject_event_id": self.subject_event_id,
             "ref_event_id": self.ref_event_id}
        if self.quote:
            d["quote"] = self.quote
        return d


# ---------------------------------------------------------------- 输出侧:分析产物

@dataclass
class AffectedEvent:
    """受影响事件:因某条直接编辑需要重新检查的既有事件。"""

    event_id: str
    source_op_ids: List[str]
    relation: str        # same_object | condition_reference | temporal_constraint | interaction | chained
    judgment: str        # propagated | stopped | uncertain
    evidence: List[dict] = field(default_factory=list)
    note: str = ""
    decided_by: str = "rule"                  # rule | llm

    def to_dict(self) -> dict:
        return {"event_id": self.event_id,
                "source_op_ids": list(self.source_op_ids),
                "relation": self.relation,
                "judgment": self.judgment,
                "evidence": list(self.evidence),
                "note": self.note,
                "decided_by": self.decided_by}


@dataclass
class RepairRequirement:
    """修复要求:某依赖需要调整,具体替代方案由规划器决定。"""

    repair_id: str
    event_id: Optional[str]                  # 需修复的既有事件;None 表示对象级修复
    object_id: Optional[str]                 # 需修复涉及的对象(如失去支撑的杯子)
    source_op_ids: List[str]
    requirement: str                         # 人读的修复要求
    reason: str                              # 触发原因(传播依据)
    decision_owner: str = "planner"
    window: Optional[Tuple[float, float]] = None   # 仅在该时间窗内需要修复

    def to_dict(self) -> dict:
        d = {"repair_id": self.repair_id,
             "source_op_ids": list(self.source_op_ids),
             "requirement": self.requirement,
             "reason": self.reason,
             "decision_owner": self.decision_owner}
        if self.event_id is not None:
            d["event_id"] = self.event_id
        if self.object_id is not None:
            d["object_id"] = self.object_id
        if self.window is not None:
            d["window"] = [self.window[0], self.window[1]]
        return d


@dataclass
class Conflict:
    """指令间冲突:无法仅靠编辑排序解决,需用户或规划器裁决。"""

    conflict_id: str
    op_ids: List[str]
    kind: str           # scope_overlap | preserve_violation | opposite_intent | edit_order_cyclic
    message: str
    resolvable_by_ordering: bool = False

    def to_dict(self) -> dict:
        return {"conflict_id": self.conflict_id,
                "op_ids": list(self.op_ids),
                "kind": self.kind,
                "message": self.message,
                "resolvable_by_ordering": self.resolvable_by_ordering}


@dataclass
class Uncertainty:
    """不确定项:关系证据不足或依赖规划器结果,无法当前判定。"""

    uncertainty_id: str
    subject: str
    question: str
    missing_evidence: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"uncertainty_id": self.uncertainty_id,
                "subject": self.subject,
                "question": self.question,
                "missing_evidence": list(self.missing_evidence)}


@dataclass
class EditBatch:
    """编辑排序的一个批次:同批操作互不依赖,可一次编辑遍内完成。"""

    batch: int
    op_ids: List[str]
    repair_ids: List[str] = field(default_factory=list)
    parallel: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {"batch": self.batch,
                "op_ids": list(self.op_ids),
                "repair_ids": list(self.repair_ids),
                "parallel": self.parallel,
                "reason": self.reason}
