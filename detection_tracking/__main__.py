"""命令行:视频 + 对象名单 → 自动场景表(含 timeline)。

离线(--mock,需要预先准备的脚本化应答,见 tests):
在线(默认,需要 Qwen3-VL 端点):
    python3 -m detection_tracking --video examples/smoke_video.mp4 \
      --objects "红球,蓝球" --out examples/scene_from_video.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detection_tracking import build_scene_from_video  # noqa: E402
from edit_dependency import MockQwenClient, QwenVLClient  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="检测与跟踪:视频 → 场景对象表 + timeline")
    ap.add_argument("--video", "-v", required=True)
    ap.add_argument("--objects", "-o", required=True, help="逗号分隔的对象名单")
    ap.add_argument("--out", default=None, help="输出 scene JSON 路径(缺省打印)")
    ap.add_argument("--sample-every", type=int, default=8, help="关键帧采样间隔(帧)")
    ap.add_argument("--mock-handler", default=None,
                    help="离线模式:Python 表达式,输入(task,user) 返回应答 dict")
    args = ap.parse_args()

    client = QwenVLClient(timeout=120)
    if args.mock_handler:
        client = MockQwenClient(eval(args.mock_handler))  # noqa: S307 仅测试用
    result = build_scene_from_video(args.video, [s.strip() for s in args.objects.split(",")],
                                    client, sample_every=args.sample_every)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print("scene ->", args.out)
    else:
        print(text)
    sys.exit(0 if result.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
