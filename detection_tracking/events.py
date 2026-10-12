"""几何事件推导:由跟踪轨迹推导 timeline 事件(纯规则,无 LLM)。

推导范围(第一版,均可在轨迹上确定性判定):
  - 出现区间事件:每个对象一个主事件(verb = move / stationary);
  - 空间接触(spatial_contact):两对象 IoU 超阈值的连续时段;
  - 支撑(supported_by):A 的底边落在 B 的上沿带内且水平方向重叠的连续时段。
语义级事件(拿起/喝水/持握)不在几何推导范围内 —— 由后续版本用 Qwen3-VL
对关键帧做关系问答补充;当前输出已满足依赖器传播所需的条件种类。
"""

from typing import List

from .schemas import TrackedObject

IOU_CONTACT = 0.05          # 接触判定的 IoU 阈值
SUPPORT_BAND = 0.08         # 支撑判定:A 底边距 B 上沿不超过帧高的该比例
HORIZONTAL_MIN = 0.3        # 支撑判定:水平重叠占较小者宽度的最小比例


def iou(a, b) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _h_overlap_ratio(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    return ix / min(a[2], b[2]) if min(a[2], b[2]) > 0 else 0.0


def _runs(flags: List[bool]) -> List:
    """True 的连续段 [(start, end)],end 为开区间下标。"""
    runs, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def derive_timeline(tracked: List[TrackedObject], n_frames: int,
                    fps: float = 16.0, sample_every: int = 4,
                    frame_h: float = 480.0) -> List[dict]:
    events: List[dict] = []
    by_id = {t.object_id: t for t in tracked if t.keyframes}
    counter = 0

    def next_id() -> str:
        nonlocal counter
        counter += 1
        return "t_%02d" % counter

    times = [i / fps for i in range(0, n_frames, sample_every)]

    # 1) 每对象的主事件(出现区间 + 移动/静止)
    for t in tracked:
        if not t.keyframes:
            continue
        win = t.window()
        events.append({
            "event_id": next_id(), "verb": "stationary" if t.stationary else "move",
            "participants": [t.object_id], "window": [round(win[0], 2), round(win[1], 2)],
            "name": "%s%s" % (t.name, "静止" if t.stationary else "移动"),
            "conditions": [], "text": "%s在 [%g, %g] 秒出现,判定为%s"
            % (t.name, win[0], win[1], "静止" if t.stationary else "移动"),
        })

    # 2) 两两空间接触
    ids = [t.object_id for t in tracked if t.keyframes]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = by_id[ids[i]], by_id[ids[j]]
            flags = [iou(a.bbox_at(t), b.bbox_at(t)) > IOU_CONTACT for t in times]
            for s, e in _runs(flags):
                events.append({
                    "event_id": next_id(), "verb": "contact",
                    "participants": sorted([a.object_id, b.object_id]),
                    "window": [round(times[s], 2), round(times[min(e, len(times) - 1)], 2)],
                    "name": "%s与%s接触" % (a.name, b.name),
                    "conditions": [{"kind": "spatial_contact",
                                    "refs": sorted([a.object_id, b.object_id])}],
                    "text": "%s 与 %s 在该区间发生空间接触/重叠"
                    % (a.name, b.name),
                })

    # 3) 支撑关系:A 底边在 B 上沿带内且水平重叠充分
    for a in tracked:
        for b in tracked:
            if a.object_id == b.object_id or not a.keyframes or not b.keyframes:
                continue
            band = frame_h * SUPPORT_BAND

            def supported(t: float) -> bool:
                ba, bb = a.bbox_at(t), b.bbox_at(t)
                a_bottom = ba[1] + ba[3]
                return (abs(a_bottom - bb[1]) <= band
                        and _h_overlap_ratio(ba, bb) >= HORIZONTAL_MIN)
            flags = [supported(t) for t in times]
            for s, e in _runs(flags):
                events.append({
                    "event_id": next_id(), "verb": "rest_on",
                    "participants": [a.object_id, b.object_id],
                    "window": [round(times[s], 2), round(times[min(e, len(times) - 1)], 2)],
                    "name": "%s在%s上" % (a.name, b.name),
                    "conditions": [{"kind": "supported_by", "refs": [b.object_id]}],
                    "text": "%s 的底部位于 %s 上沿,判定为支撑关系"
                    % (a.name, b.name),
                })
    return events
