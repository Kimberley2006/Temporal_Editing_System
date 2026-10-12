"""依赖传播规则:候选查找、状态门控、冲突检测与编辑排序。

设计要点(对应规格表):
- 候选查找只看四类联系:同一对象、事件成立条件引用、明确的时序约束、空间交互。
  「时间上发生在后面」本身不构成候选 —— 这是"不默认所有后续事件都延迟"的实现。
- 时间门控:候选事件的窗口必须与编辑作用范围相交;删除支撑类关系只在关系成立
  的时间窗内传播。杯子 8 秒后被人拿着,其在 [8,12] 的事件条件不引用桌子,
  天然不命中"删除桌子"的候选 —— 即"根据事件发生时的状态决定传播还是停止"。
- 规则无法判定的候选(事件时间窗未知、动词语义不明、依赖规划器新路径)分别
  送 Qwen3-VL 判定或标记 uncertain;LLM 调用失败降级为 uncertain,不中断分析。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .llm import LLMError
from .prompts import PROPAGATION_SYSTEM, PROPAGATION_USER
from .schemas import (AffectedEvent, Conflict, EditBatch, EditOp, PreserveRequirement,
                      RepairRequirement, Scope, StatedConstraint, TimelineEvent,
                      Uncertainty)

# 状态门控的动作 × 条件矩阵:该动作会破坏/影响的条件种类
BROKEN_CONDITIONS = {
    "delete": ("requires_present", "supported_by", "held_by", "spatial_contact"),
    "repath": ("spatial_contact",),
}
_MOVE_HINTS = ("走", "跑", "滚", "驶", "移", "飞", "游", "骑")
_STATIC_HINTS = ("静", "停", "放", "躺", "靠", "坐")
_APPEARANCE_HINTS = ("颜色", "外观", "特写", "展示", "蓝色", "红色", "绿色", "黄色", "白色", "黑色")

_JUDGMENT_RANK = {"stopped": 0, "uncertain": 1, "propagated": 2}


def _verb_text(event: TimelineEvent) -> str:
    return "".join(x for x in (event.verb, event.name, event.text) if x)


def _overlap_window(scope: Scope, window: Optional[Tuple[float, float]],
                    duration: Optional[float]) -> Optional[Tuple[float, float]]:
    """编辑范围与事件时间窗的交集;事件窗未知时返回 None(无法门控,需判定)。"""
    if window is None:
        return None
    result: Optional[Tuple[float, float]] = None
    for w in scope.windows(duration):
        lo, hi = max(w[0], window[0]), min(w[1], window[1])
        if lo < hi:
            if result is None:
                result = (lo, hi)
            else:
                result = (min(result[0], lo), max(result[1], hi))
    return result


def _scopes_overlap(a: Scope, b: Scope, duration: Optional[float]) -> bool:
    """两个编辑范围是否相交;任一范围无法展开为具体窗口时保守返回 True。"""
    wa, wb = a.windows(duration), b.windows(duration)
    if not wa or not wb:
        return True
    return any(x[0] < y[1] and y[0] < x[1] for x in wa for y in wb)


@dataclass
class _Candidate:
    event: TimelineEvent
    op: EditOp
    relation: str        # same_object | condition_reference | temporal_constraint | interaction | chained
    reason: str
    preset: Optional[str] = None    # 规则直接给出的判定(propagated/uncertain);None 表示默认 propagated


@dataclass
class PropagationContext:
    ops: List[EditOp]
    preserves: List[PreserveRequirement]
    constraints: List[StatedConstraint]
    timeline: List[TimelineEvent]
    duration: Optional[float] = None
    client: Optional[object] = None          # QwenVLClient / MockQwenClient;可为 None
    warnings: List[dict] = field(default_factory=list)


# ---------------------------------------------------------------- 候选查找

def _find_candidates(ctx: PropagationContext) -> List[_Candidate]:
    candidates: List[_Candidate] = []

    for op in ctx.ops:
        if op.event_id is not None:
            # 事件类操作(delay 等):只检查与该事件有明确约束联系的事件,不外溢;
            # 用户声明约束与 timeline 内 after_end_of 可能重复命中,按事件去重
            seen: set = set()
            for ev, why in _constraint_linked(ctx, op.event_id):
                if ev.event_id not in seen:
                    seen.add(ev.event_id)
                    candidates.append(_Candidate(ev, op, "temporal_constraint", why))
            continue
        if op.object_id is None:
            continue
        oid = op.object_id

        for ev in ctx.timeline:
            if not op.scope.overlaps_window(ev.window, ctx.duration):
                continue    # 时间门控:事件窗与编辑范围不相交,后面发生≠受影响
            hit_interaction = any(c.kind == "spatial_contact" and oid in c.refs
                                  for c in ev.conditions)
            hit_condition = any(
                c.kind in BROKEN_CONDITIONS.get(op.action, ()) and oid in c.refs
                for c in ev.conditions)
            hit_same = oid in ev.participants

            if op.action == "repath":
                if hit_interaction:
                    candidates.append(_Candidate(
                        ev, op, "interaction",
                        "事件与被改路径对象存在空间接触/碰撞关系,新路径由规划器决定",
                        preset="uncertain"))
                elif hit_same:
                    if _is_move_like(ev):
                        candidates.append(_Candidate(ev, op, "same_object",
                                                     "对象将改走新路径,其移动事件需重算"))
                    elif not _is_static_like(ev):
                        candidates.append(_Candidate(ev, op, "same_object",
                                                     "事件是否涉及对象移动证据不足",
                                                     preset="uncertain"))
                continue

            if hit_condition:
                kinds = "/".join(sorted({c.kind for c in ev.conditions
                                         if oid in c.refs and c.kind != "after_end_of"}))
                candidates.append(_Candidate(
                    ev, op, "condition_reference",
                    "事件成立条件(%s)引用了被修改对象 %s" % (kinds, oid)))
            elif hit_same:
                if op.action == "delete":
                    candidates.append(_Candidate(ev, op, "same_object",
                                                 "对象被删除,其全部出现区间一并失效"))
                elif op.action == "recolor":
                    if any(h in _verb_text(ev) for h in _APPEARANCE_HINTS):
                        candidates.append(_Candidate(ev, op, "same_object",
                                                     "事件内容涉及对象外观"))
                    # 其余事件不以外关为成立条件:外观修改不破坏动作可行性,不产生候选
                elif op.action in ("slow", "speed_up"):
                    candidates.append(_Candidate(ev, op, "same_object",
                                                 "对象节奏变化,相关事件时间安排需重检"))
            # arrive/stay 为新增事件,除显式约束外不影响既有事件

    # 级联一跳:直接受影响事件之后,时序上依赖它的后续事件一并重检
    direct_ids = {c.event.event_id for c in candidates}
    for c in list(candidates):
        for ev in ctx.timeline:
            if ev.event_id in direct_ids:
                continue
            if any(cc.kind == "after_end_of" and c.event.event_id in cc.refs
                   for cc in ev.conditions):
                candidates.append(_Candidate(
                    ev, c.op, "chained",
                    "事件在受影响事件 %s 结束后开始,存在时序依赖" % c.event.event_id))
                direct_ids.add(ev.event_id)
    return candidates


def _constraint_linked(ctx: PropagationContext, event_id: str):
    """与某事件有明确时序约束联系的事件(用户声明的约束 + timeline 内的 after_end_of)。"""
    linked = []
    for con in ctx.constraints:
        if con.ref_event_id == event_id:
            ev = _find_event(ctx, con.subject_event_id)
            if ev:
                linked.append((ev, "用户约束要求其在 %s 结束后开始" % event_id))
        elif con.subject_event_id == event_id:
            ev = _find_event(ctx, con.ref_event_id)
            if ev:
                linked.append((ev, "用户约束将其与 %s 关联" % event_id))
    for ev in ctx.timeline:
        if any(c.kind == "after_end_of" and event_id in c.refs for c in ev.conditions):
            linked.append((ev, "事件成立要求 %s 已结束" % event_id))
    return linked


def _find_event(ctx: PropagationContext, event_id: str) -> Optional[TimelineEvent]:
    for ev in ctx.timeline:
        if ev.event_id == event_id:
            return ev
    return None


def _is_move_like(ev: TimelineEvent) -> bool:
    return any(h in _verb_text(ev) for h in _MOVE_HINTS)


def _is_static_like(ev: TimelineEvent) -> bool:
    return any(h in _verb_text(ev) for h in _STATIC_HINTS)


# ---------------------------------------------------------------- 判定与产出

def analyze_propagation(ctx: PropagationContext) -> dict:
    """返回 {affected, repairs, uncertainties};修复方案由规划器决定。"""
    candidates = _find_candidates(ctx)
    judged = _resolve_judgments(ctx, candidates)

    affected: Dict[str, AffectedEvent] = {}
    repairs: List[RepairRequirement] = []
    uncertainties: List[Uncertainty] = []
    counters = {"repair": 0, "u": 0}

    def next_repair_id():
        counters["repair"] += 1
        return "repair_%02d" % counters["repair"]

    def next_uncertainty_id():
        counters["u"] += 1
        return "u_%02d" % counters["u"]

    for cand in candidates:
        judgment, decided_by, reason = judged[(cand.op.op_id, cand.event.event_id)]
        overlap = _overlap_window(cand.op.scope, cand.event.window, ctx.duration)

        entry = affected.setdefault(cand.event.event_id, AffectedEvent(
            event_id=cand.event.event_id, source_op_ids=[], relation=cand.relation,
            judgment="stopped", evidence=[], decided_by=decided_by))
        entry.source_op_ids.append(cand.op.op_id)
        if decided_by == "llm":
            entry.decided_by = "llm"
        entry.evidence.append({
            "op_id": cand.op.op_id, "relation": cand.relation,
            "judgment": judgment, "reason": reason,
            "window": list(overlap) if overlap else None})
        if _JUDGMENT_RANK[judgment] > _JUDGMENT_RANK[entry.judgment]:
            entry.judgment = judgment

        if judgment == "propagated":
            before = counters["repair"]
            specific = _build_repairs(cand, overlap, next_repair_id)
            repairs.extend(specific)
            if counters["repair"] == before:
                repairs.append(RepairRequirement(
                    repair_id=next_repair_id(), event_id=cand.event.event_id,
                    object_id=cand.op.object_id, source_op_ids=[cand.op.op_id],
                    requirement="事件需按「%s」编辑后的状态重新检查" % cand.op.action,
                    reason=cand.reason, window=overlap))
        elif judgment == "uncertain":
            uncertainties.append(Uncertainty(
                uncertainty_id=next_uncertainty_id(),
                subject="%s × %s" % (cand.op.op_id, cand.event.event_id),
                question="编辑「%s」对事件 %s 的影响待定" % (
                    cand.op.quote or cand.op.action, cand.event.event_id),
                missing_evidence=[reason]))

    return {"affected": list(affected.values()), "repairs": repairs,
            "uncertainties": uncertainties}


def _resolve_judgments(ctx: PropagationContext, candidates: List[_Candidate]) -> dict:
    """确定性判定为主;preset=uncertain 或证据不足(时间窗未知)的候选交 LLM 判定。"""
    judged: Dict[Tuple[str, str], tuple] = {}
    pending: List[_Candidate] = []
    for cand in candidates:
        if cand.preset == "uncertain":
            judged[(cand.op.op_id, cand.event.event_id)] = (
                "uncertain", "rule", cand.reason)
        elif cand.event.window is None:
            pending.append(cand)     # 时间未知无法门控,送判定
        else:
            judged[(cand.op.op_id, cand.event.event_id)] = (
                "propagated", "rule", cand.reason)

    if not pending:
        return judged
    if ctx.client is None:
        ctx.warnings.append({"code": "judgment_skipped",
                             "message": "存在待判定候选但未配置 LLM,已按不确定处理"})
        for cand in pending:
            judged[(cand.op.op_id, cand.event.event_id)] = (
                "uncertain", "rule", "事件时间窗未知且无 LLM 可判定")
        return judged

    op_brief = "; ".join("%s: %s %s" % (o.op_id, o.action, o.object_id or o.event_id or "")
                         for o in ctx.ops)
    cand_brief = "; ".join("%s(%s)" % (c.event.event_id, _verb_text(c.event)) for c in pending)
    evidence = []
    for ev in ctx.timeline:
        win = "[%g,%g]s" % ev.window if ev.window else "未知"
        evidence.append("- %s %s 参与者%s 条件%s" % (
            ev.event_id, win, ev.participants,
            [(c.kind, c.refs) for c in ev.conditions]))
    preserve_brief = "; ".join("%s 保持%s" % (p.object_id, "/".join(p.aspects))
                               for p in ctx.preserves) or "(无)"
    system = PROPAGATION_SYSTEM.format(
        op_brief=op_brief, candidate_brief=cand_brief,
        evidence_brief="\n".join(evidence), preserve_brief=preserve_brief)
    try:
        answer = ctx.client.complete_json("judge_propagation", system, PROPAGATION_USER)
    except LLMError as exc:
        ctx.warnings.append({"code": "judgment_llm_failed",
                             "message": "传播判定调用失败,候选按不确定处理:%s" % exc.message})
        for cand in pending:
            judged[(cand.op.op_id, cand.event.event_id)] = (
                "uncertain", "rule", "LLM 调用失败:%s" % exc.message)
        return judged

    answers = {j.get("event_id"): j for j in answer.get("judgments", [])
               if isinstance(j, dict) and j.get("event_id")}
    for cand in pending:
        j = answers.get(cand.event.event_id)
        if j and j.get("judgment") in ("propagated", "stopped", "uncertain"):
            judged[(cand.op.op_id, cand.event.event_id)] = (
                j["judgment"], "llm", j.get("reason", ""))
        else:
            judged[(cand.op.op_id, cand.event.event_id)] = (
                "uncertain", "llm", "模型未给出该事件的判定")
    return judged


def _build_repairs(cand: _Candidate, overlap: Optional[Tuple[float, float]],
                   next_id) -> List[RepairRequirement]:
    repairs: List[RepairRequirement] = []
    op, ev = cand.op, cand.event

    def add(**kwargs):
        repairs.append(RepairRequirement(repair_id=next_id(), **kwargs))

    if op.action == "delete" and op.object_id is not None:
        for c in ev.conditions:
            if c.kind == "supported_by" and op.object_id in c.refs:
                # 支撑被删:依赖对象(桌上的杯子)在该区间需要重新安置
                for dep in [p for p in ev.participants if p != op.object_id]:
                    add(event_id=ev.event_id, object_id=dep,
                        source_op_ids=[op.op_id],
                        requirement="对象 %s 在该区间失去来自 %s 的支撑,需重新安置"
                                    % (dep, op.object_id),
                        reason="支撑关系在编辑区间内被删除破坏", window=overlap)
            elif c.kind in ("requires_present", "held_by", "spatial_contact") \
                    and op.object_id in c.refs:
                # 引用被删对象为成立条件:动作需修复,替代方案由规划器决定
                add(event_id=ev.event_id,
                    object_id=op.object_id if op.object_id in ev.participants else None,
                    source_op_ids=[op.op_id],
                    requirement="事件动作需重新设计(%s 已删除),替代动作由规划器决定"
                                % op.object_id,
                    reason="成立条件 %s 引用了被删除对象" % c.kind, window=overlap)

    if op.event_id is not None and (cand.relation == "temporal_constraint" or any(
            c.kind == "after_end_of" and op.event_id in c.refs for c in ev.conditions)):
        add(event_id=ev.event_id, object_id=None, source_op_ids=[op.op_id],
            requirement="事件时间需重新排布,保持与 %s 的时序约束" % op.event_id,
            reason="被修改事件是本事件时序约束的参照", window=overlap)
    return repairs


# ---------------------------------------------------------------- 冲突检测

def detect_conflicts(ops: List[EditOp], preserves: List[PreserveRequirement],
                     duration: Optional[float]) -> List[Conflict]:
    conflicts: List[Conflict] = []
    counter = 0

    def next_id():
        nonlocal counter
        counter += 1
        return "c_%02d" % counter

    touched_aspects = {
        "delete": {"appearance", "behavior"}, "recolor": {"appearance"},
        "repath": {"behavior"}, "delay": {"behavior"}, "slow": {"behavior"},
        "speed_up": {"behavior"}, "arrive": {"behavior"}, "stay": {"behavior"},
    }

    for op in ops:
        if op.object_id is None:
            continue
        for p in preserves:
            if p.object_id == op.object_id and \
                    touched_aspects.get(op.action, set()) & set(p.aspects):
                conflicts.append(Conflict(
                    conflict_id=next_id(), op_ids=[op.op_id], kind="preserve_violation",
                    message="对象 %s 被要求保持不变(%s),同时又要求「%s」"
                            % (op.object_id, "/".join(p.aspects), op.quote or op.action)))

    for i in range(len(ops)):
        for j in range(i + 1, len(ops)):
            a, b = ops[i], ops[j]
            same_target = (a.object_id is not None and a.object_id == b.object_id) or (
                a.event_id is not None and a.event_id == b.event_id)
            if not same_target:
                continue
            overlap = _scopes_overlap(a.scope, b.scope, duration)
            if a.action == "delete" and b.action == "delete" and overlap:
                conflicts.append(Conflict(
                    conflict_id=next_id(), op_ids=[a.op_id, b.op_id],
                    kind="scope_overlap", message="两条删除指令作用范围重叠"))
            elif "delete" in (a.action, b.action) and overlap:
                other = b if a.action == "delete" else a
                conflicts.append(Conflict(
                    conflict_id=next_id(), op_ids=[a.op_id, b.op_id], kind="scope_overlap",
                    message="删除与「%s」作用于同一目标的重叠区间,不能仅靠编辑排序解决"
                            % (other.quote or other.action)))
            elif {a.action, b.action} == {"slow", "speed_up"}:
                conflicts.append(Conflict(
                    conflict_id=next_id(), op_ids=[a.op_id, b.op_id],
                    kind="opposite_intent", message="同一目标同时被要求减慢与加快"))
    return conflicts


# ---------------------------------------------------------------- 编辑排序

def build_edit_order(ops: List[EditOp], affected: List[AffectedEvent],
                     repairs: List[RepairRequirement],
                     duration: Optional[float]) -> dict:
    """基于依赖边的拓扑分批:同批互不依赖可一次编辑遍完成,批数即编辑遍数。

    排序原则(最少改动、最少遍数):
    - 删除等改变存在前提的操作先做,后续编辑基于删除后状态一次算对,避免返工;
    - 同对象上大范围修改先于小范围;同事件上存在性 > 几何/时序 > 外观;
    - 互不依赖的编辑并入同批,减少对视频的编辑遍数。
    """
    priority = {"delete": 0, "repath": 1, "delay": 2, "slow": 2, "speed_up": 2,
                "arrive": 2, "stay": 2, "recolor": 3}
    affected_events_by_op: Dict[str, set] = {}
    for ae in affected:
        if ae.judgment == "stopped":
            continue
        for op_id in ae.source_op_ids:
            affected_events_by_op.setdefault(op_id, set()).add(ae.event_id)

    edges: Dict[str, set] = {op.op_id: set() for op in ops}
    reasons: Dict[Tuple[str, str], str] = {}
    for a in ops:
        for b in ops:
            if a.op_id == b.op_id:
                continue
            reason = None
            b_target_events = {b.event_id} if b.event_id else set()
            if a.action == "delete" and (
                    (a.object_id is not None and a.object_id == b.object_id
                     and _scopes_overlap(a.scope, b.scope, duration))
                    or (b_target_events & affected_events_by_op.get(a.op_id, set()))):
                reason = "删除改变存在前提,「%s」应基于删除后状态计算" % (b.quote or b.action)
            elif (a.object_id is not None and a.object_id == b.object_id
                    and a.scope.contains_scope(b.scope, duration)
                    and not b.scope.contains_scope(a.scope, duration)):
                reason = "大范围修改先执行,避免小范围改动被覆盖后返工"
            elif (a.event_id is not None and a.event_id == b.event_id
                    and priority.get(a.action, 9) < priority.get(b.action, 9)):
                reason = "同一事件上,高优先级修改先于低优先级修改"
            elif (affected_events_by_op.get(a.op_id, set())
                    & affected_events_by_op.get(b.op_id, set())
                    and priority.get(a.action, 9) < priority.get(b.action, 9)):
                reason = "两处修改作用于同一事件,存在性/几何/时序类修改先做,避免返工"
            if reason:
                edges[a.op_id].add(b.op_id)
                reasons[(a.op_id, b.op_id)] = reason

    indeg = {op_id: 0 for op_id in edges}
    for dsts in edges.values():
        for d in dsts:
            indeg[d] += 1
    queue = sorted(op_id for op_id, d in indeg.items() if d == 0)
    levels: List[List[str]] = []
    placed: set = set()
    while queue:
        levels.append(queue)
        placed.update(queue)
        nxt: set = set()
        for op_id in queue:
            for d in edges[op_id]:
                indeg[d] -= 1
                if indeg[d] == 0 and d not in placed:
                    nxt.add(d)
        queue = sorted(nxt)
    cyclic = len(placed) < len(ops)

    repairs_by_op: Dict[str, List[str]] = {}
    for r in repairs:
        for op_id in r.source_op_ids:
            lst = repairs_by_op.setdefault(op_id, [])
            if r.repair_id not in lst:
                lst.append(r.repair_id)

    op_by_id = {o.op_id: o for o in ops}
    batches: List[EditBatch] = []
    for idx, level in enumerate(levels, start=1):
        repair_ids: List[str] = []
        for op_id in level:
            repair_ids.extend(repairs_by_op.get(op_id, []))
        if len(level) > 1:
            reason = "同批操作互不依赖,可并行执行以减少编辑遍数"
        else:
            op = op_by_id[level[0]]
            reason = ";".join(sorted(reasons[(op.op_id, b)] for b in edges[op.op_id])) \
                or "无前置依赖"
        batches.append(EditBatch(batch=idx, op_ids=list(level), repair_ids=repair_ids,
                                 parallel=len(level) > 1, reason=reason))
    leftover = [op.op_id for op in ops if op.op_id not in placed]
    if leftover:
        batches.append(EditBatch(
            batch=len(batches) + 1, op_ids=leftover, parallel=False,
            reason="存在循环依赖,按指令顺序兜底,需人工核对"))
    return {"batches": batches, "cyclic": cyclic}
