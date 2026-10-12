"""时间规划器:事件排程与约束传播。

变量:每个既有事件的时间窗(start/end)与新增事件(arrive/stay)的窗口。
约束:after_end_of 时序约束、视频总时长边界。
解法:按依赖器批次顺序应用操作效果(delay 平移/slow 拉伸/arrive 落点/stay 锚定),
然后对 after_end_of 约束做定点迭代传播(后继事件 start 不足则整体后移,保持时长),
不可满足(越出视频末尾)记为 violation 上报 —— 即"运动能否实现由规划器判断"的时间部分。
"""

from typing import Dict, List, Optional, Tuple

from .schemas import (DEFAULT_SLOW_FACTOR, DEFAULT_SPEED_UP_FACTOR, PlanOp,
                      ScheduledItem, TaskPackage)

_MAX_PROPAGATION_ROUNDS = 16


def schedule(package: TaskPackage) -> dict:
    """返回 {scheduled, violations, event_windows, op_windows}。"""
    events = {e.event_id: e for e in package.events}
    duration = package.video_duration
    op_by_id = {o.op_id: o for o in package.ops if not o.skipped}
    scheduled: List[ScheduledItem] = []
    violations: List[dict] = []
    op_windows: Dict[str, Tuple[float, float]] = {}

    def events_of_object(oid: str) -> List:
        return [e for e in events.values() if oid in e.participants]

    def object_first_start(oid: str) -> float:
        starts = [e.window[0] for e in events_of_object(oid) if e.window]
        return min(starts) if starts else 0.0

    # 1) 按批次顺序应用操作效果
    ordered_ops: List[PlanOp] = []
    for batch in package.batches:
        for oid in batch:
            if oid in op_by_id:
                ordered_ops.append(op_by_id[oid])

    for op in ordered_ops:
        if op.action == "delay":
            seconds = float(op.params.get("seconds", 0))
            ev = events.get(op.event_id) if op.event_id else None
            if ev and ev.window:
                ev.window = (ev.window[0] + seconds, ev.window[1] + seconds)
                op_windows[op.op_id] = ev.window
                scheduled.append(ScheduledItem(
                    ref=ev.event_id, kind="event", start=ev.window[0], end=ev.window[1],
                    note="delay +%gs 由 op %s 应用" % (seconds, op.op_id)))
        elif op.action in ("slow", "speed_up"):
            default = DEFAULT_SLOW_FACTOR if op.action == "slow" else DEFAULT_SPEED_UP_FACTOR
            factor = float(op.params.get("factor", default))
            targets = []
            if op.event_id and op.event_id in events:
                targets = [events[op.event_id]]
            elif op.object_id:
                targets = events_of_object(op.object_id)
            for ev in targets:
                if not ev.window:
                    continue
                s, e = ev.window
                ev.window = (s, s + (e - s) * factor)
            op_windows[op.op_id] = (min(t.window[0] for t in targets if t.window),
                                    max(t.window[1] for t in targets if t.window)) \
                if targets and all(t.window for t in targets) else None
            scheduled.append(ScheduledItem(
                ref=op.object_id or op.event_id, kind="op",
                note="%s 系数 %g 应用于 %d 个事件" % (op.action, factor, len(targets))))
        elif op.action == "arrive":
            t = op.params.get("time_seconds")
            if t is None:
                t = _after_last_event(events_of_object(op.object_id)) if op.object_id else 0.0
                scheduled.append(ScheduledItem(
                    ref=op.op_id, kind="op", start=t, end=t,
                    note="arrive 未指定时间,规划器默认排在同对象最后事件之后"))
            else:
                scheduled.append(ScheduledItem(ref=op.op_id, kind="op",
                                               start=float(t), end=float(t)))
            op_windows[op.op_id] = (float(t), float(t))
        elif op.action == "stay":
            d = float(op.params.get("duration_seconds", 0))
            after = op.params.get("after_op_id")
            if after and after in op_windows:
                start = op_windows[after][1]
            else:
                start = object_first_start(op.object_id) if op.object_id else 0.0
                scheduled.append(ScheduledItem(
                    ref=op.op_id, kind="op", start=start, end=start + d,
                    note="独立停留锚定在对象首次出现时刻"))
            op_windows[op.op_id] = (start, start + d)
            if not (after and after in op_windows):
                scheduled.append(ScheduledItem(ref=op.op_id, kind="op",
                                               start=start, end=start + d))
        else:
            # delete/recolor/repath:窗口取编辑范围(整段则由时长界定)
            if op.scope_kind == "window":
                op_windows[op.op_id] = (op.scope_start, op.scope_end)
            elif duration is not None:
                op_windows[op.op_id] = (0.0, duration)
            else:
                op_windows[op.op_id] = None
            if op_windows[op.op_id]:
                w = op_windows[op.op_id]
                scheduled.append(ScheduledItem(ref=op.op_id, kind="op",
                                               start=w[0], end=w[1]))

    # 2) after_end_of 约束定点传播
    for _ in range(_MAX_PROPAGATION_ROUNDS):
        changed = False
        for ev in events.values():
            if ev.window is None:
                continue
            for cond in ev.conditions:
                if cond.get("kind") != "after_end_of":
                    continue
                for ref in cond.get("refs", []):
                    base = events.get(ref)
                    if base is None or base.window is None:
                        continue
                    if ev.window[0] < base.window[1]:
                        delta = base.window[1] - ev.window[0]
                        dur = ev.window[1] - ev.window[0]
                        ev.window = (base.window[1], base.window[1] + dur)
                        changed = True
                        scheduled.append(ScheduledItem(
                            ref=ev.event_id, kind="event",
                            start=ev.window[0], end=ev.window[1],
                            note="时序约束(%s 之后)后移 %gs" % (ref, delta)))
        if not changed:
            break

    # 3) 边界校验
    for ev in events.values():
        if ev.window is None:
            continue
        if duration is not None and ev.window[1] > duration + 1e-9:
            violations.append({
                "kind": "time_out_of_range", "ref": ev.event_id,
                "window": list(ev.window), "video_duration": duration,
                "message": "事件 %s 排程后结束于 %g 秒,超出视频时长 %g 秒"
                           % (ev.event_id, ev.window[1], duration)})
        if ev.window[0] < -1e-9:
            violations.append({
                "kind": "negative_start", "ref": ev.event_id,
                "window": list(ev.window),
                "message": "事件 %s 排程后起点为负" % ev.event_id})
    for op_id, w in op_windows.items():
        if w is not None and duration is not None and w[1] > duration + 1e-9:
            violations.append({
                "kind": "op_out_of_range", "ref": op_id, "window": list(w),
                "video_duration": duration,
                "message": "编辑 %s 的窗口结束于 %g 秒,超出视频时长" % (op_id, w[1])})

    # 汇总事件最终窗口
    final = []
    for ev in package.events:
        cur = events[ev.event_id]
        if cur.window and (ev.window is None or cur.window != ev.window):
            final.append(ScheduledItem(ref=ev.event_id, kind="event",
                                       start=cur.window[0], end=cur.window[1],
                                       note="排程调整(原 %s)" % (list(ev.window)
                                                                 if ev.window else "未知")))
        elif cur.window:
            final.append(ScheduledItem(ref=ev.event_id, kind="event",
                                       start=cur.window[0], end=cur.window[1]))
    return {"scheduled": final + scheduled, "violations": violations,
            "event_windows": {eid: e.window for eid, e in events.items()},
            "op_windows": op_windows}


def _after_last_event(events: List) -> float:
    ends = [e.window[1] for e in events if e.window]
    return max(ends) if ends else 0.0
