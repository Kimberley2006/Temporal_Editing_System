"""检测与跟踪模块测试:离线运行,Mock 检测器按合成视频的已知几何生成应答。

验证:检测解析/过滤、跟踪插值、移动/静止判定、接触与支撑事件推导、
场景表与系统既有 schema 的兼容性(edit_dependency.parse_timeline 可解析、
解析器 SceneInfo 可加载)、以及"自动场景表 → 依赖器 → 规划器"闭环。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_tracking import DetectionPipeline, build_scene_from_video  # noqa: E402
from edit_dependency import MockQwenClient  # noqa: E402

VIDEO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "examples", "smoke_video.mp4")

# 合成视频真值:红球 [60,210,56,56] -> [700,210,56,56] 线性;蓝球 [420,200,64,64] 静止
N_FRAMES, FPS, W, H = 33, 16, 832, 480


def gt_red(f):
    return [60 + (700 - 60) * f / (N_FRAMES - 1), 210, 56, 56]


def make_mock_handler(noise: float = 2.0, drop_frames=(), miss_blue=False):
    """模拟 Qwen3-VL:按真值 + 确定性噪声返回归一化框。"""

    def _noisy(v, k, f):
        return v + ((f * 7 + k * 13) % 5 - 2) * noise / 5

    def handler(task, user):
        assert task == "detect_objects"
        f = int(user.split("第 ")[1].split("/")[0])
        objs = []
        if f not in drop_frames:
            red = [_noisy(v, k, f) for k, v in enumerate(gt_red(f))]
            objs.append({"name": "红球",
                         "bbox": [round(v / W, 4) if k % 2 == 0 else round(v / W, 4)
                                  for k, v in enumerate(red[:2])]
                                     + [round(red[2] / W, 4), round(red[3] / H, 4)],
                         "confidence": 0.9})
            objs[0]["bbox"] = [round(red[0] / W, 4), round(red[1] / H, 4),
                               round(red[2] / W, 4), round(red[3] / H, 4)]
        if not miss_blue:
            objs.append({"name": "蓝球",
                         "bbox": [420 / W, 200 / H, 64 / W, 64 / H],
                         "confidence": 0.95})
        return {"objects": objs}
    return handler


def run_pipeline(handler=None, sample_every=4):
    client = MockQwenClient(handler or make_mock_handler())
    return DetectionPipeline(client=client, sample_every=sample_every).run(
        VIDEO, ["红球", "蓝球"])


# ---------------------------------------------------------------- 全链路

def test_full_pipeline_mock():
    out = run_pipeline()
    assert out["status"] == "ok", out
    scene = out["scene"]
    # ffmpeg 封装后实际可读 32 帧(写入 33,末帧被复用器丢弃)
    assert scene["video"]["frame_count"] == N_FRAMES - 1
    red = scene["objects"][0]
    assert red["name"] == "红球" and red["localization"]["bbox"]
    # 轨迹首点应接近真值 [60,210,56,56]
    t0, b0 = red["localization"]["trajectory"][0]
    assert t0 == 0.0
    assert all(abs(b0[k] - gt_red(0)[k]) < 6 for k in range(4))
    # 依赖器的 timeline 解析器接受自动生成的 timeline(schema 兼容)
    from edit_dependency.schemas import parse_timeline
    events = parse_timeline(scene["timeline"])
    assert events, "timeline 不应为空"


def test_stationary_vs_moving():
    out = run_pipeline()
    objs = {o["name"]: o for o in out["scene"]["objects"]}
    assert objs["蓝球"]["localization"]["stationary"] is True
    assert objs["红球"]["localization"]["stationary"] is False
    verbs = {e["verb"] for e in out["scene"]["timeline"]}
    assert "move" in verbs and "stationary" in verbs


def test_contact_event_derived():
    """红球轨迹在 ~1.0-1.3s 穿过蓝球 → 接触事件(spatial_contact 条件)。"""
    out = run_pipeline()
    contacts = [e for e in out["scene"]["timeline"] if e["verb"] == "contact"]
    assert contacts, "红球穿过蓝球应推导出接触事件"
    c = contacts[0]
    assert c["conditions"][0]["kind"] == "spatial_contact"
    assert sorted(c["conditions"][0]["refs"]) == ["object_01", "object_02"]
    lo, hi = c["window"]
    assert 0.7 <= lo <= 1.3 and 1.0 <= hi <= 1.8


def test_support_event_derived():
    from detection_tracking.events import derive_timeline
    from detection_tracking.schemas import TrackedObject
    cup = TrackedObject(object_id="object_03", name="杯子",
                        keyframes=[(0.0, [140, 330, 48, 70]), (8.0, [140, 330, 48, 70])])
    table = TrackedObject(object_id="object_04", name="桌子",
                          keyframes=[(0.0, [80, 396, 200, 26]), (8.0, [80, 396, 200, 26])])
    events = derive_timeline([cup, table], n_frames=129, fps=16, frame_h=480)
    supports = [e for e in events if e["verb"] == "rest_on"]
    assert supports and supports[0]["conditions"][0]["kind"] == "supported_by"
    assert supports[0]["conditions"][0]["refs"] == ["object_04"]


def test_detection_filtering():
    """无效名称/越界框/低信度被过滤,不影响有效项。"""
    def handler(task, user):
        return {"objects": [
            {"name": "不存在的对象", "bbox": [0.1, 0.1, 0.1, 0.1], "confidence": 0.9},
            {"name": "蓝球", "bbox": [1.5, -0.2, 0.1, 0.1], "confidence": 0.9},   # 越界
            {"name": "蓝球", "bbox": [0.5, 0.4, 0.08, 0.13], "confidence": 0.1},  # 低信度
            {"name": "蓝球", "bbox": [420 / 832, 200 / 480, 64 / 832, 64 / 480],
             "confidence": 0.95},
        ]}
    out = run_pipeline(handler=handler)
    assert out["status"] == "ok"
    blue = out["scene"]["objects"][1]
    assert blue["localization"]["bbox"] == [420.0, 200.0, 64.0, 64.0]


def test_missing_object_warning():
    out = run_pipeline(handler=make_mock_handler(miss_blue=True))
    assert out["status"] == "ok"
    assert any(w["code"] == "object_not_found" and "蓝球" in w["message"]
               for w in out.get("warnings", []))
    assert out["scene"]["objects"][1]["localization"] == {}


def test_interpolation_across_dropped_frames():
    """部分关键帧检测缺失时,轨迹仍应连续(插值补齐)。"""
    out = run_pipeline(handler=make_mock_handler(drop_frames={8, 12}))
    assert out["status"] == "ok"
    traj = out["scene"]["objects"][0]["localization"]["trajectory"]
    # t=0.5s(帧 8)被丢弃,由相邻关键帧插值,位置误差应很小
    approx = dict((round(t, 3), b) for t, b in traj)
    assert any(abs(t - 0.5) <= 0.26 for t in approx)


def test_errors():
    from edit_dependency import QwenVLClient
    out = DetectionPipeline(client=QwenVLClient(base_url="http://127.0.0.1:1/v1")).run(
        VIDEO, ["红球"])
    assert out["status"] == "error"
    assert out["error"]["code"] in ("llm_detect_failed", "llm_unavailable",
                                    "llm_http_error")
    out2 = run_pipeline()
    assert out2["status"] == "ok"


# ---------------------------------------------------------------- 闭环:自动场景 → 依赖器 → 规划器

def test_closed_loop_with_downstream():
    """自动生成的场景表能被依赖器/规划器直接消费(离线 Mock 全链路)。"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "examples"))
    from edit_dependency import EditDependencyAnalyzer
    from spatio_temporal_planner import SpatioTemporalPlanner

    scene = run_pipeline()["scene"]

    # 依赖器:手工最小输出(删除红球),验证 timeline 条件被自动场景喂进去
    dep = {"status": "ok",
           "direct_edits": [{"op_id": "op_01", "action": "delete",
                             "object_id": "object_01", "event_id": None,
                             "region_id": None, "scope": {"kind": "all"},
                             "params": {}, "quote": "删除红球"}],
           "edit_order": [{"batch": 1, "op_ids": ["op_01"]}],
           "repair_requirements": [], "conflicts": [], "uncertainties": [],
           "preserve": {"object_behavior_ids": ["object_02"],
                        "target_identity": True, "background": True},
           "events": []}
    plan = SpatioTemporalPlanner(scene=scene).plan(dependency_output=dep, scene=scene)
    assert plan["status"] == "ok", plan
    # 删除红球的编辑段应基于自动推导的出现区间 [0, 2]
    segs = [s for p in plan["passes"] for s in p["segments"]
            if s["spec_type"] == "delete_mask"]
    assert segs and segs[0]["start"] == 0.0 and abs(segs[0]["end"] - 2.0) < 0.1
    assert all(c["passed"] for c in plan["verification"])


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS %s" % name)
            except AssertionError as exc:
                failures += 1
                print("FAIL %s: %s" % (name, exc))
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print("ERROR %s: %r" % (name, exc))
    print("\n%d failed" % failures if failures else "\nall tests passed")
    sys.exit(1 if failures else 0)
