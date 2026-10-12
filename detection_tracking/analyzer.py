"""检测与跟踪主流水线:视频 → 关键帧检测 → 跟踪 → 事件推导 → 场景表。

检测用 Qwen3-VL-8B-Instruct 的多模态开放词表能力(输入帧 + 目标名列表,
输出归一化框);跟踪是确定性规则(同名关联 + 置信度择优 + 关键帧插值);
事件推导是纯几何规则(出现/移动/静止/接触/支撑)。LLM 只承担"看图给框",
其余全部可离线测试 —— 与系统其余部分的"提取归 LLM、规则归几何"分工一致。
"""

import base64
import io
import json
from typing import List, Optional

from .events import derive_timeline
from .schemas import Detection, TrackedObject

DETECT_SYSTEM = """你是视频帧的目标检测器。在图像中检测指定的对象,输出严格 JSON:
{"objects": [{"name": "对象名(必须与给定名单一致)", "bbox": [x, y, w, h], "confidence": 0~1}]}

要求:
- bbox 为相对坐标,四个数都在 0~1 之间(x,y 为左上角,w,h 为宽高),图像中找不到的对象不要输出;
- 每个名字最多一个框;确信度低于 0.3 的不要输出;
- 只输出 JSON,不要任何其他文字。"""

# 移动/静止判定:最大速度低于帧宽的该比例视为静止
STATIONARY_SPEED_RATIO = 0.02


def _error(code: str, message: str, **details) -> dict:
    err = {"code": code, "message": message}
    err.update(details)
    return {"status": "error", "error": err}


class DetectionPipeline:
    def __init__(self, client, sample_every: int = 8,
                 stationary_speed_ratio: float = STATIONARY_SPEED_RATIO):
        """client: QwenVLClient(真实)/ MockQwenClient(离线测试)。"""
        self.client = client
        self.sample_every = max(int(sample_every), 1)
        self.stationary_speed_ratio = stationary_speed_ratio

    # ---------------------------------------------------------------- 对外入口

    def run(self, video_path: str, object_names: List[str]) -> dict:
        """返回 {status, scene, detections}(scene 即系统通用的场景表+timeline)。"""
        names = [n.strip() for n in object_names if n.strip()]
        if not names:
            return _error("empty_object_list", "目标对象名单为空")
        meta = self._video_meta(video_path)
        if meta is None:
            return _error("video_unreadable", "无法读取视频:%s" % video_path)
        n_frames, fps = meta["n_frames"], meta["fps"]

        frame_indices = list(range(0, n_frames, self.sample_every))
        if frame_indices[-1] != n_frames - 1:
            frame_indices.append(n_frames - 1)

        detections: List[Detection] = []
        for fi in frame_indices:
            dets = self._detect_frame(video_path, fi, n_frames, names, meta)
            if dets["status"] == "error":
                return dets["payload"] if "payload" in dets else dets
            detections.extend(dets["detections"])

        tracked = self._track(detections, names, fps, n_frames)
        missing = [n for n, t in tracked.items() if not t.keyframes]
        timeline = derive_timeline(list(tracked.values()), n_frames=n_frames)
        scene = self._build_scene(tracked, names, meta, timeline)
        result = {"status": "ok", "scene": scene,
                  "detections": [d.to_dict() for d in detections]}
        if missing:
            result["warnings"] = [{"code": "object_not_found",
                                   "message": "未检测到对象:%s" % ", ".join(missing),
                                   "objects": missing}]
        return result

    # ---------------------------------------------------------------- 步骤 1:检测

    def _detect_frame(self, video_path: str, frame_idx: int, n_frames: int,
                      names: List[str], meta: dict) -> dict:
        b64 = self._frame_b64(video_path, frame_idx, meta)
        if b64 is None:
            return {"status": "error",
                    "payload": _error("frame_unreadable",
                                      "无法读取第 %d 帧" % frame_idx)}
        user = ("第 %d/%d 帧。检测以下对象:%s" % (frame_idx, n_frames, "、".join(names)))
        try:
            answer = self.client.complete_json("detect_objects", DETECT_SYSTEM,
                                               user, images=[b64])
        except Exception as exc:  # LLMError 或网络异常
            return {"status": "error",
                    "payload": _error("llm_detect_failed",
                                      "检测调用失败(帧 %d):%s" % (frame_idx, exc))}
        dets: List[Detection] = []
        for raw in answer.get("objects", []):
            name = raw.get("name", "")
            if name not in names:
                continue
            bbox = self._to_pixel_bbox(raw.get("bbox"), meta)
            if bbox is None:
                continue
            conf = raw.get("confidence", 1.0)
            if not isinstance(conf, (int, float)) or conf < 0.3:
                continue
            dets.append(Detection(name=name, bbox=bbox,
                                  confidence=float(conf), frame_idx=frame_idx))
        return {"status": "ok", "detections": dets}

    @staticmethod
    def _to_pixel_bbox(norm, meta: dict) -> Optional[list]:
        if not isinstance(norm, (list, tuple)) or len(norm) != 4:
            return None
        if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in norm):
            return None
        x, y, w, h = (float(v) for v in norm)
        if w <= 0 or h <= 0:
            return None
        return [x * meta["width"], y * meta["height"],
                w * meta["width"], h * meta["height"]]

    # ---------------------------------------------------------------- 步骤 2:跟踪

    def _track(self, detections: List[Detection], names: List[str],
               fps: float, n_frames: int) -> dict:
        """v1:同名关联(每个名字单实例),同帧多检选择信度最高者;
        关键帧轨迹 = 各采样帧的检测框,stationary 由最大速度判定。"""
        tracked = {}
        width = None
        for idx, name in enumerate(names):
            obj = TrackedObject(object_id="object_%02d" % (idx + 1), name=name)
            per_frame = {}
            for d in detections:
                if d.name == name:
                    cur = per_frame.get(d.frame_idx)
                    if cur is None or d.confidence > cur.confidence:
                        per_frame[d.frame_idx] = d
            obj.keyframes = sorted(
                (fi / fps, [round(v, 1) for v in d.bbox])
                for fi, d in per_frame.items())
            tracked[name] = obj
            if obj.keyframes and width is None:
                width = obj.keyframes[0][1][2]
        self._postprocess(tracked, width, fps, n_frames)
        return tracked

    def _postprocess(self, tracked: dict, width, fps: float, n_frames: int):
        """检测后处理:尺寸中值平滑 + 首尾位移静止判定 + 画面内匀速外推补尾。

        真实模型对小目标的框位置较准但宽高抖动明显,尺寸抖动会虚高中心位移,
        导致静止对象被误判为移动;刚体尺寸近似恒定,用关键帧中位数替代。
        """
        for obj in tracked.values():
            if len(obj.keyframes) >= 3:
                ws = sorted(k[1][2] for k in obj.keyframes)
                hs = sorted(k[1][3] for k in obj.keyframes)
                mw, mh = ws[len(ws) // 2], hs[len(hs) // 2]
                obj.keyframes = [(t, [b[0], b[1], mw, mh])
                                 for t, b in obj.keyframes]
        speed_th = (width or 100) * self.stationary_speed_ratio
        for obj in tracked.values():
            if len(obj.keyframes) >= 2:
                (t0, b0), (t1, b1) = obj.keyframes[0], obj.keyframes[-1]
                disp = (((b0[0] + b0[2] / 2) - (b1[0] + b1[2] / 2)) ** 2
                        + ((b0[1] + b0[3] / 2) - (b1[1] + b1[3] / 2)) ** 2) ** 0.5
                obj.stationary = disp / max(t1 - t0, 1e-6) <= speed_th
            else:
                obj.stationary = True
        end_t = (n_frames - 1) / fps
        sample_t = self.sample_every / fps
        for obj in tracked.values():
            if not obj.keyframes or obj.stationary or len(obj.keyframes) < 2:
                continue
            tl, bl = obj.keyframes[-1]
            if end_t - tl <= sample_t * 0.25:   # 几乎无缺口才不补
                continue
            tp, bp = obj.keyframes[-2]
            dt = tl - tp
            vx = (bl[0] - bp[0]) / dt if dt > 0 else 0.0
            vy = (bl[1] - bp[1]) / dt if dt > 0 else 0.0
            ext = end_t - tl
            x, y = bl[0] + vx * ext, bl[1] + vy * ext
            w, h = bl[2], bl[3]
            if -w * 0.5 <= x <= 832 - w * 0.5 and -h * 0.5 <= y <= 480 - h * 0.5:
                obj.keyframes.append((end_t, [round(x, 1), round(y, 1), w, h]))

    # ---------------------------------------------------------------- 步骤 4:场景组装

    def _build_scene(self, tracked: dict, names: List[str],
                     meta: dict, timeline: list) -> dict:
        objects = []
        for name in names:
            obj = tracked.get(name)
            entry = {"id": obj.object_id if obj else None, "name": name,
                     "aliases": []}
            if obj and obj.keyframes:
                entry["localization"] = {
                    "bbox": [round(v, 1) for v in obj.keyframes[0][1]],
                    "trajectory": obj.trajectory(meta["fps"], meta["n_frames"]),
                    "stationary": obj.stationary,
                }
            else:
                entry["localization"] = {}
            objects.append(entry)
        return {
            "objects": objects,
            "regions": [],
            "video": {"duration_seconds": round(meta["n_frames"] / meta["fps"], 3),
                      "fps": meta["fps"], "frame_count": meta["n_frames"],
                      "width": meta["width"], "height": meta["height"]},
            "timeline": timeline,
        }

    # ---------------------------------------------------------------- 视频工具

    @staticmethod
    def _video_meta(video_path: str) -> Optional[dict]:
        try:
            import math

            import imageio.v2 as imageio
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            fps = float(meta.get("fps") or 16)
            size = meta.get("size") or meta.get("source_size") or (832, 480)
            raw_n = meta.get("nframes")
            n = int(raw_n) if isinstance(raw_n, (int, float)) and math.isfinite(raw_n) \
                and raw_n > 0 else 0
            if n <= 0:
                duration = meta.get("duration")
                n = int(round(float(duration) * fps)) if duration else reader.count_frames()
            reader.close()
            return {"n_frames": max(n, 1), "fps": fps,
                    "width": int(size[0]), "height": int(size[1])}
        except Exception:
            return None

    @staticmethod
    def _frame_b64(video_path: str, frame_idx: int, meta: dict) -> Optional[str]:
        try:
            import imageio.v2 as imageio
            from PIL import Image
            reader = imageio.get_reader(video_path)
            frame = reader.get_data(frame_idx)
            reader.close()
            img = Image.fromarray(frame).convert("RGB")
            img = img.resize((min(img.width, 832),
                              int(img.height * min(img.width, 832) / img.width)))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=88)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:
            return None


def build_scene_from_video(video_path: str, object_names: List[str],
                           client, sample_every: int = 8) -> dict:
    """便捷入口:视频 + 对象名单 → scene JSON(含 timeline)。"""
    return DetectionPipeline(client=client, sample_every=sample_every).run(
        video_path, object_names)
