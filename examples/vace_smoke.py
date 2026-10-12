"""VACE 端到端冒烟:合成视频「删除红球」→ 规划器 → 物化 → Wan2.1-VACE 真实推理。

前置:python3 examples/make_smoke_video.py;GPU 需空闲(GPU0)。
依赖器输出用手工最小构造(冒烟只验证 规划→物化→执行 这一段,不走 LLM)。

    python3 examples/vace_smoke.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatio_temporal_planner import SpatioTemporalPlanner  # noqa: E402
from spatio_temporal_planner.executor import VACEExecutor  # noqa: E402
from spatio_temporal_planner.materialize import build_vace_jobs  # noqa: E402

EX_DIR = os.path.dirname(os.path.abspath(__file__))
SCENE_PATH = os.path.join(EX_DIR, "scene_smoke.json")
VIDEO_IN = os.path.join(EX_DIR, "smoke_video.mp4")
VIDEO_OUT = os.path.join(EX_DIR, "smoke_edited.mp4")

DEP_OUTPUT = {
    "status": "ok",
    "direct_edits": [
        {"op_id": "op_01", "action": "delete", "object_id": "object_01",
         "event_id": None, "region_id": None, "scope": {"kind": "all"},
         "params": {}, "quote": "删除红球"},
    ],
    "edit_order": [{"batch": 1, "op_ids": ["op_01"]}],
    "repair_requirements": [
        {"repair_id": "repair_01", "event_id": "t_01", "object_id": "object_01",
         "source_op_ids": ["op_01"],
         "requirement": "事件需按「delete」编辑后的状态重新检查",
         "reason": "对象被删除,其全部出现区间一并失效", "window": [0.0, 2.0]},
    ],
    "conflicts": [], "uncertainties": [],
    "preserve": {"object_behavior_ids": ["object_02"],
                 "target_identity": True, "background": True},
    "events": [],
}


def main():
    with open(SCENE_PATH, encoding="utf-8") as f:
        scene = json.load(f)

    plan = SpatioTemporalPlanner(scene=scene).plan(dependency_output=DEP_OUTPUT,
                                                   scene=scene)
    assert plan["status"] == "ok", plan
    assert all(c["passed"] for c in plan["verification"]), plan["verification"]

    red = scene["objects"][0]["localization"]
    geometry = {"object_01": {"shape": "ellipse", "bbox": red["bbox"],
                              "trajectory": red["trajectory"]},
                "object_02": {"shape": "ellipse",
                              "bbox": scene["objects"][1]["localization"]["bbox"]}}
    video = scene["video"]
    jobs_out = build_vace_jobs(plan, scene,
                               (video["width"], video["height"]), video["fps"],
                               geometry=geometry)
    assert jobs_out["status"] == "ok", jobs_out
    jobs = jobs_out["passes"][0]["jobs"]
    print("VACE 任务:")
    for j in jobs:
        print("  %s %s frames=%s mask_bbox=%s" % (
            j["job_id"], j["spec_type"], j["frame_range"],
            (j["mask"] or {}).get("bbox")))

    executor = VACEExecutor(device="cuda:0")
    result = executor.run_pass(VIDEO_IN, jobs, VIDEO_OUT)
    print("输出:", result)
    assert os.path.exists(VIDEO_OUT)
    size = os.path.getsize(VIDEO_OUT)
    print("完成: %s (%d KB, %d 帧)" % (VIDEO_OUT, size // 1024, result["num_frames"]))


if __name__ == "__main__":
    main()
