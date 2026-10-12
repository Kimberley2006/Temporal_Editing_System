"""检测与跟踪 live 验证:真实 Qwen3-VL 检测合成视频,对照已知真值判定。

前置:QWEN_VL_BASE_URL 指向已服务的 Qwen3-VL 端点(如 http://127.0.0.1:8001/v1)。
判定标准(合成视频真值已知):
  - 红球轨迹与真值线性轨迹的 IoU >= 0.5(在 0/0.5/1/1.5/2.0 秒抽检);
  - 蓝球判定为静止;
  - 推导出红×蓝接触事件(spatial_contact)。
输出:examples/scene_detected.json + PASS/FAIL。

    QWEN_VL_BASE_URL=http://127.0.0.1:8001/v1 python3 examples/detect_live.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_tracking import build_scene_from_video  # noqa: E402
from edit_dependency import QwenVLClient  # noqa: E402

VIDEO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "smoke_video.mp4")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_detected.json")


def gt_red(t: float):
    return [60 + 320 * t, 210, 56, 56]   # 2 秒从 60 到 700


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    out = build_scene_from_video(VIDEO, ["红球", "蓝球"],
                                 QwenVLClient(timeout=120), sample_every=8)
    print(json.dumps({"status": out.get("status"),
                      "warnings": out.get("warnings", []),
                      "timeline_summary": [
                          {"verb": e["verb"], "window": e["window"], "name": e["name"]}
                          for e in out.get("scene", {}).get("timeline", [])]},
                     ensure_ascii=False, indent=2))
    if out.get("status") != "ok":
        print("FAIL: pipeline error", out.get("error"))
        sys.exit(1)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out["scene"], f, ensure_ascii=False, indent=2)
    print("scene ->", OUT)

    objs = {o["name"]: o for o in out["scene"]["objects"]}
    failures = []
    red = objs.get("红球", {}).get("localization", {})
    if not red.get("bbox"):
        failures.append("红球未被检测到")
    else:
        for t in (0.0, 0.5, 1.0, 1.5, 2.0):
            got = red["bbox"] if t == 0 else _bbox_at(red["trajectory"], t)
            v = iou(got, gt_red(t))
            print("t=%.1fs 红球 IoU=%.2f" % (t, v))
            if v < 0.5:
                failures.append("t=%.1fs 红色轨迹 IoU %.2f < 0.5" % (t, v))
    blue = objs.get("蓝球", {}).get("localization", {})
    if not blue.get("bbox"):
        failures.append("蓝球未被检测到")
    elif blue.get("stationary") is not True:
        failures.append("蓝球未判定为静止(stationary=%s)" % blue.get("stationary"))
    if not any(e["verb"] == "contact" for e in out["scene"]["timeline"]):
        failures.append("未推导出红×蓝接触事件")

    if failures:
        print("FAIL:", "; ".join(failures))
        sys.exit(1)
    print("PASS: 检测跟踪与真值一致")


def _bbox_at(trajectory, t):
    """由轨迹关键帧插值出 t 时刻 bbox。"""
    pts = sorted((p[0], p[1]) for p in trajectory)
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for i in range(len(pts) - 1):
        (t0, b0), (t1, b1) = pts[i], pts[i + 1]
        if t0 <= t <= t1:
            a = (t - t0) / (t1 - t0)
            return [b0[k] + (b1[k] - b0[k]) * a for k in range(4)]
    return pts[-1][1]


if __name__ == "__main__":
    main()
