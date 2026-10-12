"""统一任务包构建:合并解析器/依赖器结论为 TaskPackage。

主通道是编辑依赖器的输出(其 direct_edits/events 已与解析器同构);
解析器单独使用时(简单指令)由 parser 适配路径归一化为 arrive/stay 操作。
冲突策略在此落地:涉冲突的 op 标记 skipped,批次剔除后重排,冲突项转
EscalatedConflict 反馈用户 —— 规划器不吞冲突,也不因冲突阻塞无冲突部分。
"""

from typing import List, Optional

from .schemas import (EscalatedConflict, PlanEvent, PlanOp, PlanRepair, TaskPackage)

_CONFLICT_KIND_ADVICE = {
    "scope_overlap": "两条编辑作用于同一目标的重叠区间,请保留其一或错开时间范围",
    "preserve_violation": "该对象被要求保持不变,请撤销对其的修改或取消保持要求",
    "opposite_intent": "同一目标同时被要求减速与加速,请保留其一",
}


def build_task_package(dependency_output: Optional[dict] = None,
                       parser_output: Optional[dict] = None,
                       scene: Optional[dict] = None,
                       timeline: Optional[list] = None) -> dict:
    """返回 {status, package} 或 {status:"error", payload}。"""
    if dependency_output is None and parser_output is None:
        return {"status": "error",
                "payload": {"status": "error",
                            "error": {"code": "empty_input",
                                      "message": "需要提供 dependency_output 或 parser_output"}}}
    src = dependency_output or parser_output
    if src.get("status") != "ok":
        return {"status": "error",
                "payload": {"status": "error",
                            "error": {"code": "upstream_not_ok",
                                      "message": "上游输出 status 非 ok,规划器拒绝接管",
                                      "upstream_error": src.get("error", {}).get("code", "unknown")}}}

    timeline = timeline if timeline is not None else (scene or {}).get("timeline", [])
    events = [_to_plan_event(d) for d in timeline]

    if dependency_output is not None:
        ops = [_op_from_dependency(e) for e in dependency_output.get("direct_edits", [])]
        batches = [list(b.get("op_ids", [])) for b in dependency_output.get("edit_order", [])]
    else:
        ops = _ops_from_parser(parser_output)
        batches = [[o.op_id for o in ops]]

    repairs = [PlanRepair(
        repair_id=r.get("repair_id", ""),
        event_id=r.get("event_id"), object_id=r.get("object_id"),
        source_op_ids=list(r.get("source_op_ids", [])),
        requirement=r.get("requirement", ""), reason=r.get("reason", ""),
        window=tuple(r["window"]) if r.get("window") else None)
        for r in dependency_output.get("repair_requirements", [])] \
        if dependency_output is not None else []

    video = (scene or {}).get("video", {})
    package = TaskPackage(
        ops=ops, events=events,
        new_events=list(src.get("events", [])),
        repairs=repairs, batches=batches,
        preserve_object_ids=list(src.get("preserve", {}).get("object_behavior_ids", [])),
        conflicts=list(dependency_output.get("conflicts", [])) if dependency_output is not None else [],
        uncertainties=list(dependency_output.get("uncertainties", [])) if dependency_output is not None else [],
        video_duration=video.get("duration_seconds"),
        fps=video.get("fps"))

    # 冲突策略:跳过涉冲突 op(用户已裁决的冲突不在此列),批次重排,冲突项反馈用户
    result = _apply_conflict_policy(package)
    return {"status": "ok", "package": result}


def _to_plan_event(d: dict) -> PlanEvent:
    win = d.get("window")
    return PlanEvent(
        event_id=d.get("event_id", ""), verb=d.get("verb", ""),
        participants=list(d.get("participants", [])),
        name=d.get("name", ""),
        window=(float(win[0]), float(win[1])) if win else None,
        conditions=list(d.get("conditions", [])))


def _op_from_dependency(e: dict) -> PlanOp:
    scope = e.get("scope", {})
    return PlanOp(
        op_id=e.get("op_id", ""), action=e.get("action", ""),
        object_id=e.get("object_id"), event_id=e.get("event_id"),
        region_id=e.get("region_id"),
        scope_kind=scope.get("kind", "all"),
        scope_start=scope.get("start_seconds"), scope_end=scope.get("end_seconds"),
        params=dict(e.get("params", {})), quote=e.get("quote", ""))


def _ops_from_parser(out: dict) -> List[PlanOp]:
    """解析器输出 → arrive/stay 操作(单目标)。"""
    ops: List[PlanOp] = []
    for idx, ev in enumerate(out.get("events", []), start=1):
        if ev.get("type") == "arrive":
            ops.append(PlanOp(
                op_id="op_%02d" % idx, action="arrive",
                object_id=out.get("target_object_id"),
                region_id=ev.get("region_id"),
                params={"time_seconds": ev.get("time_seconds")}))
        else:
            base = next((o for o in reversed(ops) if o.action == "arrive"), None)
            ops.append(PlanOp(
                op_id="op_%02d" % idx, action="stay",
                object_id=base.object_id if base else out.get("target_object_id"),
                region_id=base.region_id if base else None,
                params={"duration_seconds": ev.get("duration_seconds"),
                        "after_op_id": base.op_id if base else None}))
    return ops


def _apply_conflict_policy(package: TaskPackage) -> TaskPackage:
    """跳过冲突项:涉冲突 op 标记 skipped,从批次剔除;空批次删除;冲突项转用户反馈。"""
    op_by_id = {o.op_id: o for o in package.ops}
    conflict_ops: set = set()
    escalations: List[EscalatedConflict] = []

    for c in package.conflicts:
        involved = [oid for oid in c.get("op_ids", []) if oid in op_by_id]
        advice = _CONFLICT_KIND_ADVICE.get(c.get("kind", ""), "请调整指令后重试")
        quote = ";".join(filter(None, (op_by_id[oid].quote for oid in involved)))
        escalations.append(EscalatedConflict(
            conflict_id=c.get("conflict_id", ""),
            op_ids=involved, kind=c.get("kind", ""),
            message=c.get("message", ""),
            user_feedback="已跳过编辑[%s]:%s。%s。" % (
                ",".join(involved), c.get("message", ""), advice)
            if quote is not None else ""))
        conflict_ops.update(involved)

    for oid in conflict_ops:
        op = op_by_id[oid]
        op.skipped = True
        op.skip_reason = "conflict"

    # 批次剔除被跳过的 op;修复要求若全部来源 op 被跳过则一并挂起
    new_batches: List[List[str]] = []
    for batch in package.batches:
        kept = [oid for oid in batch if oid not in conflict_ops]
        if kept:
            new_batches.append(kept)
    package.batches = new_batches

    # 冲突反馈的 user_feedback 重建(quote 用于提示)
    for esc in escalations:
        quotes = ";".join(filter(None, (op_by_id[oid].quote for oid in esc.op_ids)))
        advice = _CONFLICT_KIND_ADVICE.get(esc.kind, "请调整指令后重试")
        esc.user_feedback = "已跳过「%s」:%s。建议:%s。" % (quotes or ",".join(esc.op_ids),
                                                       esc.message, advice)

    package.ops = [o for o in package.ops if not o.skipped] + \
                  [o for o in package.ops if o.skipped]
    package.escalations = escalations
    return package
