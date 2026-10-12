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

        # 空间规划(Phase 2):repath/arrive 的路径与避碰,消解不确定项
        spatial_out = self._plan_spatial(package, scene, sched["op_windows"])
        spatial_keyframes = spatial_out["keyframes"]
        resolved_uncertainties = spatial_out["resolved"]
        spatial_escalations = spatial_out["escalations"]
        spatial_warnings = spatial_out["warnings"]

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
                p.keyframes.extend(spatial_keyframes.get(oid, []))
                p.keyframes.extend(_op_keyframes(op, sched["op_windows"].get(oid), regions)
                                   if oid not in spatial_keyframes else [])
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
            escalated_conflicts=package.escalations + spatial_escalations,
            resolved_uncertainties=resolved_uncertainties,
            pending_uncertainties=pending_uncertainties,
            violations=sched["violations"],
            provenance={
                "edit_order": "dependency:edit_order",
                "schedule": "rule:interval_propagation",
                "spatial": "rule:path_planning_with_avoidance",
                "repairs": "template:minimal_change",
                "conflicts": "policy:skip_and_escalate",
                "segments": "rule:affected_windows",
            })
        plan.verification = verify(package, plan, event_windows)
        plan.warnings.extend(spatial_warnings)
        plan.warnings.extend(
            {"code": "schedule_violation", "message": v["message"]}
            for v in sched["violations"])
        return _to_dict(plan)

    # ---------------------------------------------------------------- 工具

    def _regions(self, scene: dict) -> List[dict]:
        return scene.get("regions", []) if isinstance(scene, dict) else []

    def _plan_spatial(self, package: TaskPackage, scene: dict, op_windows: dict) -> dict:
        """repath/arrive 的路径规划与避碰;不可行则跳过该编辑并升级反馈。"""
        from .schemas import EscalatedConflict, Keyframe
        from .spatial import bbox_from_center, plan_path, sample_keyframes

        out = {"keyframes": {}, "resolved": [], "escalations": [], "warnings": []}
        if not isinstance(scene, dict):
            return out
        objects = {o.get("id"): o for o in scene.get("objects", [])}
        regions = {r.get("id"): r for r in scene.get("regions", [])}
        video = scene.get("video", {})
        W = float(video.get("width", 832))
        H = float(video.get("height", 480))
        fps = float(video.get("fps", package.fps or 16))

        def bbox_of(entry):
            loc = (entry or {}).get("localization", {})
            b = loc.get("bbox")
            return tuple(b) if b else None

        obstacles = [bbox_of(objects[pid]) for pid in package.preserve_object_ids
                     if pid in objects]
        obstacles = [b for b in obstacles if b]

        uncertainty_subjects = {u.get("subject", ""): u for u in package.uncertainties}
        op_by_id = {o.op_id: o for o in package.ops}
        infeasible: set = set()

        for oid, batch in [(oid, b) for b in package.batches for oid in b]:
            op = op_by_id.get(oid)
            if op is None or op.action not in ("repath", "arrive"):
                continue
            start = bbox_of(objects.get(op.object_id))
            if op.action == "arrive" and op.region_id:
                goal = bbox_of(regions.get(op.region_id))
            else:
                # repath(或 arrive 无区域)无显式终点:取水平镜像位置作为新终点
                # (演示语义;检测跟踪提供真实轨迹后可改为"原起终点换路线")
                if start is None:
                    goal = None
                else:
                    W_, H_ = start[2], start[3]
                    goal = (W - start[0] - start[2], start[1], W_, H_)
            result = plan_path(start, goal, obstacles, (W, H))
            window = op_windows.get(oid)
            n_frames = max(int(round((window[1] - window[0]) * fps)), 2) if window else 2

            matched = [s for s in uncertainty_subjects if oid in s]
            if result.feasible:
                kfs = []
                if window:
                    for (px, py), t in zip(sample_keyframes(result.waypoints, n_frames),
                                           [window[0] + (window[1] - window[0]) * i /
                                            max(n_frames - 1, 1) for i in range(n_frames)]):
                        w, h = (start or (0, 0, 0, 0))[2], (start or (0, 0, 0, 0))[3]
                        kfs.append(Keyframe(t=round(t, 3), object_id=op.object_id,
                                            region_id=op.region_id,
                                            bbox=list(bbox_from_center((px, py), w, h))))
                # 关键帧抽稀(每 0.5 秒一个,计划保持可读)
                step = max(int(fps // 2), 1)
                out["keyframes"][oid] = kfs[::step] or kfs[:1]
                for s in matched:
                    u = uncertainty_subjects.pop(s)
                    out["resolved"].append({
                        "uncertainty_id": u.get("uncertainty_id"), "subject": s,
                        "question": u.get("question"),
                        "resolution": "空间规划器已求解:%s;最小间隙 %gpx"
                                      % (result.reason, result.min_clearance)})
            else:
                if op.action == "repath":
                    infeasible.add(oid)
                    op.skipped = True
                    op.skip_reason = "no_feasible_path"
                    out["escalations"].append(EscalatedConflict(
                        conflict_id="spatial_%s" % oid, op_ids=[oid],
                        kind="spatial_infeasible",
                        message=result.reason,
                        user_feedback="已跳过「%s」:%s。建议放宽保持要求或指定其他路径。"
                                      % (op.quote or oid, result.reason)))
                else:
                    out["warnings"].append({
                        "code": "path_infeasible", "message":
                        "arrive %s 路径不可行(%s),保留时间锚点关键帧" % (oid, result.reason)})
                for s in matched:
                    u = uncertainty_subjects.pop(s)
                    out["escalations"].append(EscalatedConflict(
                        conflict_id="spatial_%s" % oid, op_ids=[oid],
                        kind="spatial_infeasible",
                        message="%s(不确定项 %s 无法消解)" % (result.reason, s),
                        user_feedback="路径求解失败,请检查几何输入或放宽约束:%s" % s))

        if infeasible:
            package.batches = [[oid for oid in b if oid not in infeasible]
                               for b in package.batches]
            package.batches = [b for b in package.batches if b]
        return out


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
        # 路径已由空间规划器求解(不可行的已被跳过),按出现区间给出运动编辑段
        spans = windows or ([(0.0, duration)] if duration is not None else [])
        for w in spans:
            segments.append(SegmentSpec(object_id=op.object_id, start=w[0], end=w[1],
                                        spec_type="new_motion",
                                        params={"note": "轨迹见本批 keyframes;VACE 以 mask+prompt 引导"},
                                        source=op.op_id))
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
