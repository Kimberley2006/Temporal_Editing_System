"""修复方案生成:最小改动模板(已确认的默认风格)。

依赖器只标记"需要修复",这里给出规划级落地方案:
- 引用被删对象的动作   → remove_action_bridge(动作删除 + 前后帧桥接)
- 失去支撑的对象       → hold_position(窗口内位置保持,不引入新轨迹)
- 时序约束破坏         → rescheduled(由时间规划器重排落地)
- 删除已覆盖的出现区间 → covered_by_delete(无需额外动作)
LLM 替代动作提议为可选开关(llm_propose),Phase 3 接入;模板输出始终经过
时间/空间可行性校验后才进入执行计划。
"""

from typing import List, Optional

from .schemas import PlanEvent, PlanOp, RepairResolution, SegmentSpec, TaskPackage


def resolve_repairs(package: TaskPackage, event_windows: dict) -> List[RepairResolution]:
    resolutions: List[RepairResolution] = []
    delete_ops = {o.object_id: o for o in package.ops
                  if o.action == "delete" and not o.skipped and o.object_id}
    event_by_id = {e.event_id: e for e in package.events}

    for r in package.repairs:
        if r.source_op_ids and all(
                _op_skipped(package, oid) for oid in r.source_op_ids):
            continue    # 来源编辑全部被跳过(冲突),修复随之作废

        window = r.window
        if window is None and r.event_id and r.event_id in event_windows:
            window = event_windows[r.event_id]

        if "重新设计" in r.requirement and "已删除" in r.requirement:
            resolutions.append(RepairResolution(
                repair_id=r.repair_id, resolved_as="remove_action_bridge",
                detail="动作删除 + 前后帧桥接(最小改动,不引入替代动作)",
                segments=[_segment(r.object_id, window, "remove_action_bridge",
                                   source=r.repair_id)] if window else []))
        elif "失去" in r.requirement and "支撑" in r.requirement:
            resolutions.append(RepairResolution(
                repair_id=r.repair_id, resolved_as="hold_position",
                detail="窗口内保持原位(不引入新轨迹),支撑物删除后不做位移",
                segments=[_segment(r.object_id, window, "hold_position",
                                   source=r.repair_id)] if window else []))
        elif "时序约束" in r.requirement or "重新排布" in r.requirement:
            final = event_windows.get(r.event_id) if r.event_id else None
            resolutions.append(RepairResolution(
                repair_id=r.repair_id, resolved_as="rescheduled",
                detail="由时间规划器重排,事件最终窗口 %s"
                       % (list(final) if final else "未变/未知")))
        else:
            covered = None
            if r.object_id and r.object_id in delete_ops and window:
                covered = delete_ops[r.object_id]
            if covered is not None:
                resolutions.append(RepairResolution(
                    repair_id=r.repair_id, resolved_as="covered_by_delete",
                    detail="该区间已并入 %s 的删除覆盖,无需额外修复" % covered.op_id))
            else:
                resolutions.append(RepairResolution(
                    repair_id=r.repair_id, resolved_as="recheck_in_pass",
                    detail="随来源编辑所在批次重检(最小改动兜底)"))
    return resolutions


def _op_skipped(package: TaskPackage, op_id: str) -> bool:
    for o in package.ops:
        if o.op_id == op_id:
            return o.skipped
    return True


def _segment(object_id: Optional[str], window, spec_type: str, source: str) -> SegmentSpec:
    return SegmentSpec(object_id=object_id, start=window[0], end=window[1],
                       spec_type=spec_type, params={}, source=source)
