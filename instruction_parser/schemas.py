"""输入/输出的数据结构:场景对象表与编辑事件。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class VideoInfo:
    duration_seconds: Optional[float] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "VideoInfo":
        if not data:
            return cls()
        return cls(
            duration_seconds=data.get("duration_seconds"),
            fps=data.get("fps"),
            frame_count=data.get("frame_count"),
        )


@dataclass
class SceneObject:
    """场景对象。bbox/mask/trajectory 等定位信息由检测跟踪模块提供,解析器只引用 ID。"""

    id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    localization: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneObject":
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            aliases=list(data.get("aliases", [])),
            localization=dict(data.get("localization", {})),
        )


@dataclass
class Region:
    id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    localization: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Region":
        return cls(
            id=data["id"],
            name=data.get("name", ""),
            aliases=list(data.get("aliases", [])),
            localization=dict(data.get("localization", {})),
        )


@dataclass
class SceneInfo:
    """场景对象表:第一版由人工建立,检测与跟踪由独立模块负责。"""

    objects: List[SceneObject] = field(default_factory=list)
    regions: List[Region] = field(default_factory=list)
    video: VideoInfo = field(default_factory=VideoInfo)

    @classmethod
    def from_dict(cls, data: dict) -> "SceneInfo":
        objects = [SceneObject.from_dict(d) for d in data.get("objects", [])]
        regions = [Region.from_dict(d) for d in data.get("regions", [])]
        obj_ids = [o.id for o in objects]
        if len(obj_ids) != len(set(obj_ids)):
            raise ValueError("场景对象表中存在重复的对象 ID: %s" % obj_ids)
        region_ids = [r.id for r in regions]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("场景对象表中存在重复的区域 ID: %s" % region_ids)
        return cls(objects=objects, regions=regions,
                   video=VideoInfo.from_dict(data.get("video")))


@dataclass
class Event:
    """结构化编辑事件,arrive 表示到达、stay 表示停留(stay 通过 after_event_id 关联 arrive)。"""

    event_id: str
    type: str  # "arrive" | "stay"
    region_id: Optional[str] = None
    time_seconds: Optional[float] = None
    after_event_id: Optional[str] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> dict:
        d = {"event_id": self.event_id, "type": self.type}
        if self.region_id is not None:
            d["region_id"] = self.region_id
        if self.type == "arrive":
            # 到达时间未指定时显式输出 null(字段为空,不自行补值),供规划器判断
            d["time_seconds"] = self.time_seconds
        else:
            if self.after_event_id is not None:
                d["after_event_id"] = self.after_event_id
            if self.duration_seconds is not None:
                d["duration_seconds"] = self.duration_seconds
        return d
