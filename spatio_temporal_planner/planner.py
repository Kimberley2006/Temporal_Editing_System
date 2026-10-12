"""时空规划器主编排:统一任务包 → 时间排程 → 修复落地 → 批次展开 → 静态验证。

对上游的契约:
- 输入主通道为编辑依赖器输出(与解析器同构字段已合并);
- 顺序决策(edit_order)来自依赖器,规划器不重排,只展开成逐段计划;
- 冲突策略:跳过冲突项、执行无冲突部分、冲突单独反馈用户(task_package 落地);
- 修复默认最小改动模板(repair.py);
- "运动能否实现"由本模块判断:排程越界/约束不可满足记入 violations 并使验证不通过。
"""

import dataclasses
from typing import Dict, List, Optional

from .repair import resolve_repairs
from .schemas import (EscalatedConflict, ExecutionPlan, Keyframe, PassPlan,
                      RepairResolution, ScheduledItem, SegmentSpec, TaskPackage)
from .task_package import build_task_package
from .temporal import schedule
from .verifier import verify


class SpatioTemporalPlanner:
    def __init__(self, scene: Optional[dict] = None):
        self.scene = scene or {}

    def plan(self, dependency_output: Optional[dict] = None,
             parser_output: Optional[dict] = None,
             scene: Optional[dict] = None) -> dict:
        scene = scene or self.scene
        built = build_task_package(dependency_output=dependency_output,
                                   parser_output=parser_output, scene=scene)
        if built["status"] == "error":
            return built["payload"]
        package: TaskPackage = built["package"]

        # 时间排程
        sched = schedule(package)
        event_windows = sched["event_windows"]

        # 修复落地(最小改动模板)
        resolutions = resolve_repairs(package, event_windows)
        res_by_repair = {r.repair_id: r for r in resolutions}
        op_batch = {oid: i for i, batch in enumerate(package.batches) for oid in batch}

        # 依赖器受影响证据:删除/换路径编辑的出现区间
        affected_windows: Dict[str, List] = {}
        if dependency_output:
            for ae in dependency_output.get("affected_events", []):
                if ae.get("judgment") == "stopped":
                    continue
                for ev in ae.get("evidence", []):
                    if ev.get("window"):
                        affected_windows.setdefault(ev["op_id"], []).append(ev["window"])

        duration = package.video_duration
        regions = {r.get("id"): r for r in self._regions(scene)}
        passes: List[PassPlan] = []
        pending_uncertainties: List[dict] = []
        resolved_uncertainties: List[dict] = []
        uncertainty_subjects = {u.get("subject", ""): u for u in package.uncertainties}

        op_by_id = {o.op_id: o for o in package.ops}
        for i, batch in enumerate(package.batches, start=1):
            p = PassPlan(pass_no=i, op_ids=list(batch))
            for oid in batch:
                op = op_by_id.get(oid)
                if op is None:
                    continue
                p.scheduled.extend(
                    s for s in sched["scheduled"]
                    if s.kind == "op" and s.ref == oid)
                p.segments.extend(_op_segments(op, affected_windows.get(oid, []),
                                               duration, regions,
                                               pending_uncertainties, uncertainty_subjects))
                p.keyframes.extend(_op_keyframes(op, sched["op_windows"].get(oid), regions))
            # 修复方案挂到来源 op 的批次(修复段并入该批次执行)
            for r in package.repairs:
                if r.repair_id in res_by_repair:
                    hosts = [oid for oid in r.source_op_ids if oid in op_batch]
                    if hosts and op_batch[hosts[0]] == i - 1:
                        resolution = res_by_repair[r.repair_id]
                        p.repair_resolutions.append(resolution)
                        p.segments.extend(resolution.segments)
            if p.scheduled or p.segments or p.op_ids:
                passes.append(p)

        # 事件级排程结果挂到第一批(全局信息)
        if passes:
            event_items = [s for s in sched["scheduled"] if s.kind == "event"]
            passes[0].scheduled = event_items + passes[0].scheduled

        plan = ExecutionPlan(
            status="ok", passes=passes, edit_passes=len(passes),
            skipped_ops=[o for o in package.ops if o.skipped],
            escalated_conflicts=package.escalations,
            resolved_uncertainties=resolved_uncertainties,
            pending_uncertainties=pending_uncertainties,
            violations=sched["violations"],
            provenance={
                "edit_order": "dependency:edit_order",
                "schedule": "rule:interval_propagation",
                "repairs": "template:minimal_change",
                "conflicts": "policy:skip_and_escalate",
                "segments": "rule:affected_windows",
            })
        plan.verification = verify(package, plan, event_windows)
        plan.warnings.extend(
            {"code": "schedule_violation", "message": v["message"]}
            for v in sched["violations"])
        return _to_dict(plan)

    # ---------------------------------------------------------------- 工具

    def _regions(self, scene: dict) -> List[dict]:
        return scene.get("regions", []) if isinstance(scene, dict) else []


def _op_segments(op, windows: List, duration, regions,
                 pending_uncertainties, uncertainty_subjects) -> List[SegmentSpec]:
    segments: List[SegmentSpec] = []
    if op.action == "delete":
        spans = windows or ([(0.0, duration)] if duration is not None else [])
        for w in spans:
            segments.append(SegmentSpec(object_id=op.object_id, start=w[0], end=w[1],
                                        spec_type="delete_mask", source=op.op_id))
    elif op.action == "recolor":
        if op.scope_kind == "window":
            w = (op.scope_start, op.scope_end)
        else:
            w = (0.0, duration) if duration is not None else (0.0, 0.0)
        segments.append(SegmentSpec(object_id=op.object_id, start=w[0], end=w[1],
                                    spec_type="recolor_mask",
                                    params={"color": op.params.get("color")},
                                    source=op.op_id))
    elif op.action == "repath":
        spans = windows or ([(0.0, duration)] if duration is not None else [])
        for w in spans:
            segments.append(SegmentSpec(object_id=op.object_id, start=w[0], end=w[1],
                                        spec_type="spatial_pending",
                                        params={"reason": "路径与避碰求解在 Phase 2 空间规划器"},
                                        source=op.op_id))
        for subject in list(uncertainty_subjects):
            if op.op_id in subject:
                u = uncertainty_subjects.pop(subject)
                pending_uncertainties.append(
                    {"uncertainty_id": u.get("uncertainty_id"), "subject": subject,
                     "question": u.get("question"),
                     "handling": "挂起至 Phase 2 空间规划器(避碰求解后消解)"})
    elif op.action in ("delay", "slow", "speed_up"):
        pass    # 效果已体现在排程(scheduled),无独立渲染段
    return segments


def _op_keyframes(op, window, regions) -> List[Keyframe]:
    kfs: List[Keyframe] = []
    if window is None:
        return kfs
    if op.action == "arrive":
        region = regions.get(op.region_id, {}) if op.region_id else {}
        bbox = region.get("localization", {}).get("bbox")
        kfs.append(Keyframe(t=window[0], object_id=op.object_id,
                            region_id=op.region_id, bbox=bbox,
                            note="" if bbox else "区域无 bbox,几何在 Phase 2 补齐"))
    elif op.action == "stay":
        region = regions.get(op.region_id, {}) if op.region_id else {}
        bbox = region.get("localization", {}).get("bbox")
        note = "停留窗口 [%g, %g]" % window
        if not bbox:
            note += ";无 bbox,几何在 Phase 2 补齐"
        kfs.append(Keyframe(t=window[0], object_id=op.object_id,
                            region_id=op.region_id, bbox=bbox, note=note))
        kfs.append(Keyframe(t=window[1], object_id=op.object_id,
                            region_id=op.region_id, bbox=bbox))
    return kfs


def _to_dict(plan: ExecutionPlan) -> dict:
    return dataclasses.asdict(plan)
