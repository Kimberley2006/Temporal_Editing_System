"""真实视频全链路:检测跟踪 → 依赖分析 → 规划 → VACE 物化与执行。

两阶段运行(GPU 守护编排,阶段间释放显存):
  --stage analyze  需要 Qwen3-VL 端点:检测跟踪(行人)→ 依赖器(删除行人)→
                   规划器 → 物化;产物存 examples/real_artifacts.json
  --stage execute  需要 ≥60GB 空闲显存:VACE 执行 + 输出验证

    QWEN_VL_BASE_URL=http://127.0.0.1:8001/v1 python3 examples/real_pipeline.py --stage analyze
    python3 examples/real_pipeline.py --stage execute
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EX_DIR = os.path.dirname(os.path.abspath(__file__))
CLIP = os.path.join(EX_DIR, "real_clip.mp4")
ARTIFACTS = os.path.join(EX_DIR, "real_artifacts.json")
EDITED = os.path.join(EX_DIR, "real_edited.mp4")
INSTRUCTION = "删除行人。"
OBJECTS = ["行人"]


def stage_analyze():
    from detection_tracking import build_scene_from_video
    from edit_dependency import EditDependencyAnalyzer, QwenVLClient
    from spatio_temporal_planner import SpatioTemporalPlanner
    from spatio_temporal_planner.materialize import build_vace_jobs

    client = QwenVLClient(timeout=180)
    det = build_scene_from_video(CLIP, OBJECTS, client, sample_every=4)
    assert det["status"] == "ok", det
    scene = det["scene"]
    print("[检测跟踪] timeline:",
          [(e["verb"], e["window"]) for e in scene["timeline"]])
    for o in scene["objects"]:
        loc = o["localization"]
        print("[检测跟踪] %s bbox=%s stationary=%s 轨迹点数=%d" % (
            o["name"], loc.get("bbox"), loc.get("stationary"),
            len(loc.get("trajectory", []))))
    if any(not o["localization"].get("bbox") for o in scene["objects"]):
        print("FAIL: 有对象未被检测到", det.get("warnings"))
        sys.exit(1)

    dep = EditDependencyAnalyzer(client=client).analyze(INSTRUCTION, scene)
    print("[依赖器] status=%s edits=%s" % (
        dep.get("status"), [(e["action"], e.get("object_id"))
                            for e in dep.get("direct_edits", [])]))
    assert dep.get("status") == "ok", dep

    plan = SpatioTemporalPlanner(scene=scene).plan(dependency_output=dep, scene=scene)
    print("[规划器] status=%s passes=%s 全检=%s" % (
        plan.get("status"), plan.get("edit_passes"),
        all(c["passed"] for c in plan.get("verification", []))))
    assert plan.get("status") == "ok", plan

    geometry = {}
    for o in scene["objects"]:
        loc = o["localization"]
        if loc.get("bbox"):
            geometry[o["id"]] = {"shape": "rect", "bbox": loc["bbox"],
                                 "trajectory": loc.get("trajectory")}
    video = scene["video"]
    jobs = build_vace_jobs(plan, scene, (video["width"], video["height"]),
                           video["fps"], geometry=geometry)
    assert jobs["status"] == "ok", jobs
    for w in jobs.get("warnings", []):
        print("[物化] warning:", w["message"])
    n_jobs = sum(len(p["jobs"]) for p in jobs["passes"])
    print("[物化] VACE 任务 %d 个" % n_jobs)
    for p in jobs["passes"]:
        for j in p["jobs"]:
            print("   %s %s %s frames=%s" % (j["job_id"], j["spec_type"],
                                             j["object_name"], j["frame_range"]))

    with open(ARTIFACTS, "w", encoding="utf-8") as f:
        json.dump({"scene": scene, "dependency": dep, "plan": plan,
                   "jobs": jobs, "instruction": INSTRUCTION}, f,
                  ensure_ascii=False, indent=2)
    print("artifacts ->", ARTIFACTS)


def stage_execute():
    from spatio_temporal_planner.executor import VACEExecutor

    with open(ARTIFACTS, encoding="utf-8") as f:
        art = json.load(f)
    jobs = art["jobs"]["passes"][0]["jobs"]
    executor = VACEExecutor(device="cuda:0")
    result = executor.run_pass(CLIP, jobs, EDITED)
    print("[VACE] 输出:", result)
    verify(art)


def verify(art):
    import imageio.v2 as imageio
    import numpy as np
    from PIL import Image, ImageDraw

    traj = art["scene"]["objects"][0]["localization"]["trajectory"]
    s = imageio.get_reader(CLIP)
    o = imageio.get_reader(EDITED)
    changes_out, changes_in = [], []
    keys = {round(t * 15): b for t, b in traj}
    for i in range(0, 33, 4):
        sf, of = np.array(s.get_data(i)).astype(int), np.array(o.get_data(i)).astype(int)
        diff = np.abs(sf - of).mean()
        bbox = keys.get(i) or keys.get(min(keys, key=lambda k: abs(k - i)))
        x, y, w, h = [int(v) for v in bbox]
        m = Image.new("L", (832, 480), 0)
        ImageDraw.Draw(m).rectangle([x, y, x + w, y + h], fill=255)
        m = np.array(m, dtype=float) / 255.0
        inside = (np.abs(sf - of).mean(axis=2) * m).sum() / max(m.sum(), 1)
        changes_out.append(diff)
        changes_in.append(inside)
        print("帧%2d 全图平均差 %.2f | 检测区内平均差 %.2f" % (i, diff, inside))
    s.close(), o.close()
    ok = max(changes_in) > 15 and np.mean(changes_out) < max(changes_in)
    print("VERIFY:", "PASS" if ok else "CHECK_MANUALLY",
          "(检测区内应显著变化,全区平均差应小于区内变化)")


if __name__ == "__main__":
    args = " ".join(sys.argv[1:]).replace("--stage", "").strip()
    if args == "analyze":
        stage_analyze()
    else:
        stage_execute()
