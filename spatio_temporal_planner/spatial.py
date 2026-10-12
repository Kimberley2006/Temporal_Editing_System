"""空间规划器(Phase 2):路径规划与避碰,消解 repath 类不确定项。

几何约定:bbox = [x, y, w, h](左上角 + 宽高),帧尺寸 (W, H)。
输入几何来自场景对象表的 localization.bbox 与 timeline 事件的定位
(检测跟踪模块提供;缺失时上报证据不足,不猜)。

策略(确定性、可解释):
- 起点/终点取对象当前 bbox 与目标区域 bbox 的中心;
- 直线路径采样,与"禁区"(保持对象 bbox 外扩 margin)相交则加一个绕行点
  (上/下两侧取余量更大的一侧);
- 绕行后仍相交或几何缺失 → 无可行路径,不确定项升级为冲突反馈;
- 输出关键帧(等速插值)与最小间隙(最近距离),作为消解证据。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

BBox = Tuple[float, float, float, float]   # (x, y, w, h)
DEFAULT_MARGIN = 40.0                      # 避碰安全边距(px)


def _center(b: BBox) -> Tuple[float, float]:
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def _inflate(b: BBox, margin: float) -> BBox:
    return (b[0] - margin, b[1] - margin, b[2] + 2 * margin, b[3] + 2 * margin)


def _point_in_bbox(p: Tuple[float, float], b: BBox) -> bool:
    return b[0] <= p[0] <= b[0] + b[2] and b[1] <= p[1] <= b[1] + b[3]


def _seg_hits_bbox(a: Tuple[float, float], c: Tuple[float, float], b: BBox,
                   samples: int = 64) -> bool:
    for i in range(samples + 1):
        t = i / samples
        if _point_in_bbox((a[0] + (c[0] - a[0]) * t, a[1] + (c[1] - a[1]) * t), b):
            return True
    return False


def _point_rect_distance(p, b: BBox) -> float:
    dx = max(b[0] - p[0], 0, p[0] - (b[0] + b[2]))
    dy = max(b[1] - p[1], 0, p[1] - (b[1] + b[3]))
    return (dx * dx + dy * dy) ** 0.5


@dataclass
class PathResult:
    feasible: bool
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    keyframes: List[Tuple[float, float]] = field(default_factory=list)   # 逐帧中心
    min_clearance: float = 0.0          # 与禁区的最小间隙(px)
    reason: str = ""


def plan_path(start_bbox: Optional[BBox], goal_bbox: Optional[BBox],
              obstacle_bboxes: List[BBox], frame_size: Tuple[float, float],
              margin: float = DEFAULT_MARGIN) -> PathResult:
    """规划从对象当前位置到目标区域中心的避碰路径。"""
    if start_bbox is None or goal_bbox is None:
        return PathResult(feasible=False,
                          reason="起点/终点几何缺失(检测跟踪模块未提供),无法规划路径")
    W, H = frame_size
    start, goal = _center(start_bbox), _center(goal_bbox)
    forbidden = [_inflate(b, margin) for b in obstacle_bboxes if b is not None]

    def clearance(pts) -> float:
        return min((_point_rect_distance(p, b) for p in pts for b in forbidden),
                   default=float("inf"))

    def blocked(pts) -> bool:
        for i in range(len(pts) - 1):
            for b in forbidden:
                if _seg_hits_bbox(pts[i], pts[i + 1], b):
                    return True
        return False

    direct = [start, goal]
    if not blocked(direct):
        return PathResult(True, waypoints=direct, min_clearance=clearance(direct),
                          reason="直线路径不经过任何禁区")

    # 绕行:对每个挡在途中的禁区,在其上/下两侧余量更大的一侧加一对绕行点
    # (禁区左外沿 + 右外沿),构成"门"形通道,避免斜线切到禁区边角
    waypoints = [start]
    current = start
    for b in forbidden:
        if not _seg_hits_bbox(current, goal, b):
            continue
        above_y = b[1]                      # 禁区上缘
        below_y = b[1] + b[3]               # 禁区下缘
        left_x = b[0] - margin              # 禁区左外沿
        right_x = b[0] + b[2] + margin      # 禁区右外沿
        up_room = above_y                   # 到画面顶的距离
        down_room = H - below_y             # 到画面底的距离
        if up_room >= down_room:
            side_y = max(above_y - margin, margin / 2.0)
        else:
            side_y = min(below_y + margin, H - margin / 2.0)
        if current[0] > right_x:            # 禁区在身后,不必绕
            continue
        waypoints.append((left_x, side_y))
        waypoints.append((right_x, side_y))
        current = (right_x, side_y)
    waypoints.append(goal)
    if blocked(waypoints):
        return PathResult(feasible=False, waypoints=waypoints,
                          reason="绕行后路径仍穿越禁区,无可行路径")
    return PathResult(True, waypoints=waypoints, min_clearance=clearance(waypoints),
                      reason="绕行路径避开全部禁区")


def sample_keyframes(waypoints: List[Tuple[float, float]], n_frames: int
                     ) -> List[Tuple[float, float]]:
    """等速折线插值:把路径均分为 n 帧(保持对象保持要求的关键帧形式)。"""
    if n_frames <= 0 or not waypoints:
        return []
    seg_len = []
    total = 0.0
    for i in range(len(waypoints) - 1):
        (x0, y0), (x1, y1) = waypoints[i], waypoints[i + 1]
        d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        seg_len.append(d)
        total += d
    if total == 0:
        return [waypoints[0]] * n_frames
    pts, acc = [], 0.0
    seg_i = 0
    for i in range(n_frames):
        target = total * i / max(n_frames - 1, 1)
        while seg_i < len(seg_len) - 1 and acc + seg_len[seg_i] < target:
            acc += seg_len[seg_i]
            seg_i += 1
        (x0, y0), (x1, y1) = waypoints[seg_i], waypoints[seg_i + 1]
        t = (target - acc) / seg_len[seg_i] if seg_len[seg_i] > 0 else 0.0
        pts.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return pts


def bbox_from_center(center: Tuple[float, float], w: float, h: float) -> BBox:
    return (center[0] - w / 2.0, center[1] - h / 2.0, w, h)


def bboxes_overlap(a: BBox, b: BBox) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and \
        a[1] < b[1] + b[3] and b[1] < a[1] + a[3]
