"""VACE 物化适配器:ExecutionPlan 的 segments/keyframes → Wan2.1-VACE 任务。

VACE 的条件形式(已在 diffsynth WanVideoUnit_VACE 源码确认):
  vace_video         输入视频帧
  vace_video_mask    逐帧 mask,白(1)=重生成区域,黑(0)=保持区域
  vace_reference_image / vace_scale / 文本 prompt
本模块把规划级 spec 翻译成 (帧区间, mask 规格, prompt) 三元组:
  delete_mask         → 对象出现区间的椭圆/矩形 mask + 移除指令
  recolor_mask        → 窗口内对象 mask + 外观指令(带颜色)
  remove_action_bridge→ 人物区域 mask + 无物动作指令
  hold_position       → 对象 mask + 原位保持指令
  repath/arrive/stay  → 关键帧 + 运动指令(VACE 无原生轨迹条件,以 mask+prompt
                        引导,几何参考由关键帧 mask 序列提供 —— 已知局限,如实标注)
几何来自场景对象表 localization.bbox 与空间规划器关键帧;缺失时该段降级为
prompt-only(mask 全黑,靠指令编辑,标记 warning)。
"""

from typing import Dict, List, Optional, Tuple

PROMPT_TEMPLATES = {
    "delete_mask": "remove the {name} completely, {qualifier}",
    "recolor_mask": "the {name} becomes {color}, {qualifier}",
    "remove_action_bridge": "the {name} performs the motion with empty hands, "
                            "holding nothing, {qualifier}",
    "hold_position": "the {name} stays still at the same position, {qualifier}",
    "new_motion": "the {name} moves smoothly to the target area, {qualifier}",
}
_QUALIFIER = "keep everything else unchanged, high quality, consistent motion"

# 常用中文颜色 -> 英文(指令里的颜色进英文 prompt 模板)
COLOR_MAP = {
    "蓝色": "blue", "红色": "red", "绿色": "green", "黄色": "yellow",
    "白色": "white", "黑色": "black", "紫色": "purple", "橙色": "orange",
    "粉色": "pink", "灰色": "gray", "棕色": "brown",
}

# 演示对象的中文 -> 英文名(Wan 双语可懂,但纯英文 prompt 更稳)
NAME_MAP = {
    "红球": "red ball", "蓝球": "blue ball", "杯子": "cup", "水杯": "cup",
    "玻璃杯": "glass", "桌子": "table", "行人": "pedestrian", "人": "person",
    "汽车": "car", "远处的汽车": "car in the distance",
}


def build_vace_jobs(plan: dict, scene: dict, frame_size: Tuple[float, float],
                    fps: float, geometry: Optional[dict] = None) -> dict:
    """把 ExecutionPlan 物化为按 pass 分组的 VACE 任务。

    geometry: {object_id: {"shape": "ellipse|rect", "bbox": [x,y,w,h],
                           "trajectory": [(t, bbox), ...]  可选}}
    返回 {status, passes: [{pass_no, op_ids, jobs: [VACEJob-dict]}], warnings}
    """
    if plan.get("status") != "ok":
        return {"status": "error",
                "error": {"code": "plan_not_ok", "message": "执行计划 status 非 ok"}}
    geometry = dict(geometry or {}) if geometry else {}
    names = {o.get("id"): o.get("name", o.get("id"))
             for o in scene.get("objects", [])}
    # 缺省几何:从场景对象表 localization.bbox 取,球形对象用椭圆 mask
    for o in scene.get("objects", []):
        oid = o.get("id")
        if oid in geometry:
            continue
        bbox = (o.get("localization") or {}).get("bbox")
        if bbox:
            shape = "ellipse" if any(w in (o.get("name") or "") for w in ("球", "球体")) \
                else "rect"
            geometry[oid] = {"shape": shape, "bbox": list(bbox)}
    warnings: List[dict] = []
    passes_out = []

    for p in plan.get("passes", []):
        jobs = []
        for seg in p.get("segments", []):
            obj = seg.get("object_id")
            geo = geometry.get(obj) if obj else None
            mask_spec = _mask_spec(geo, seg, frame_size, warnings)
            jobs.append({
                "job_id": "job_%s_%s" % (p["pass_no"], len(jobs) + 1),
                "source": seg.get("source", ""),
                "spec_type": seg["spec_type"],
                "object_id": obj,
                "object_name": names.get(obj, obj or "?"),
                "frame_range": _frame_range(seg["start"], seg["end"], fps),
                "mask": mask_spec,
                "prompt": _prompt(seg, names.get(obj, obj or "object")),
                "vace_scale": 1.0,
            })
        for kf in p.get("keyframes", []):
            geo = geometry.get(kf.get("object_id"))
            jobs.append({
                "job_id": "job_%s_kf%s" % (p["pass_no"], len(jobs) + 1),
                "source": "keyframe:%s" % kf.get("region_id") or "keyframe",
                "spec_type": "new_motion",
                "object_id": kf.get("object_id"),
                "object_name": names.get(kf.get("object_id"), "?"),
                "frame_range": None,
                "keyframe": {"t": kf.get("t"),
                             "bbox": kf.get("bbox") or (geo or {}).get("bbox")},
                "mask": None,
                "prompt": PROMPT_TEMPLATES["new_motion"].format(
                    name=names.get(kf.get("object_id"), "object"),
                    qualifier=_QUALIFIER),
                "vace_scale": 0.8,
            })
        passes_out.append({"pass_no": p["pass_no"], "op_ids": p.get("op_ids", []),
                           "jobs": jobs})
    return {"status": "ok", "passes": passes_out, "warnings": warnings}


def _frame_range(start: float, end: float, fps: float) -> Tuple[int, int]:
    f0 = max(int(round(start * fps)), 0)
    f1 = max(int(round(end * fps)), f0)
    return (f0, f1)


def _mask_spec(geo: Optional[dict], seg: dict, frame_size, warnings) -> Optional[dict]:
    if not geo or not geo.get("bbox"):
        warnings.append({
            "code": "mask_geometry_missing",
            "message": "段 %s(%s) 缺少对象几何,mask 降级为 prompt-only" % (
                seg.get("source", "?"), seg.get("spec_type"))})
        return None
    return {
        "shape": geo.get("shape", "rect"),
        "bbox": geo.get("bbox"),
        "trajectory": geo.get("trajectory"),
        "frame_size": list(frame_size),
    }


def _prompt(seg: dict, name: str) -> str:
    name = NAME_MAP.get(name, name)
    color = (seg.get("params") or {}).get("color")
    color = COLOR_MAP.get(color, color) if color else None
    template = PROMPT_TEMPLATES.get(seg["spec_type"], PROMPT_TEMPLATES["new_motion"])
    return template.format(name=name, color=color or "a different color",
                           qualifier=_QUALIFIER)
