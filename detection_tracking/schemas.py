"""检测与跟踪模块的数据结构。"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

BBox = List[float]   # [x, y, w, h],视频像素坐标


@dataclass
class Detection:
    """一帧上的一个检测框。"""

    name: str                 # 对象显示名(用户指定的开放词表)
    bbox: BBox                # 像素坐标 [x, y, w, h]
    confidence: float = 1.0
    frame_idx: int = 0

    @property
    def center(self) -> Tuple[float, float]:
        return (self.bbox[0] + self.bbox[2] / 2, self.bbox[1] + self.bbox[3] / 2)

    def to_dict(self) -> dict:
        return {"name": self.name, "bbox": [round(v, 1) for v in self.bbox],
                "confidence": round(self.confidence, 3), "frame_idx": self.frame_idx}


@dataclass
class TrackedObject:
    """一个对象的跟踪结果:关键帧轨迹 + 类别(移动/静止)。"""

    object_id: str
    name: str
    keyframes: List[Tuple[float, BBox]] = field(default_factory=list)   # [(t 秒, bbox)]
    stationary: bool = False

    def bbox_at(self, t: float) -> BBox:
        """关键帧间线性插值;越界取端点。"""
        if not self.keyframes:
            return [0, 0, 0, 0]
        if t <= self.keyframes[0][0]:
            return list(self.keyframes[0][1])
        if t >= self.keyframes[-1][0]:
            return list(self.keyframes[-1][1])
        for i in range(len(self.keyframes) - 1):
            t0, b0 = self.keyframes[i]
            t1, b1 = self.keyframes[i + 1]
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return [b0[k] + (b1[k] - b0[k]) * a for k in range(4)]
        return list(self.keyframes[-1][1])

    def trajectory(self, fps: float, n_frames: int,
                   every: int = 4) -> List[List]:
        """逐帧(抽稀)轨迹 [[t, bbox], ...],与执行器/场景表格式一致。"""
        return [[round(i / fps, 3), [round(v, 1) for v in self.bbox_at(i / fps)]]
                for i in range(0, n_frames, every)]

    def window(self) -> Optional[Tuple[float, float]]:
        if not self.keyframes:
            return None
        return (self.keyframes[0][0], self.keyframes[-1][0])

    def max_speed(self) -> float:
        """相邻关键帧中心最大位移(像素/秒)。"""
        if len(self.keyframes) < 2:
            return 0.0
        speeds = []
        for i in range(len(self.keyframes) - 1):
            (t0, b0), (t1, b1) = self.keyframes[i], self.keyframes[i + 1]
            dx = (b0[0] + b0[2] / 2) - (b1[0] + b1[2] / 2)
            dy = (b0[1] + b0[3] / 2) - (b1[1] + b1[3] / 2)
            dt = t1 - t0
            speeds.append(((dx * dx + dy * dy) ** 0.5) / dt if dt > 0 else 0.0)
        return max(speeds)
