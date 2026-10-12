"""合成冒烟视频:红球向右滚动、蓝球静止 —— 几何与 mini 冺烟场景表完全对齐,
供 VACE 真实推理冒烟使用(mask 由同一几何画出,天然吻合)。

    python3 examples/make_smoke_video.py            # 生成 examples/smoke_video.mp4
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

W, H, FPS, DURATION = 832, 480, 16, 2.0
N = int(FPS * DURATION)   # 33 帧 -> (33-1)/4+1 = 9 latent 帧,符合 Wan VAE 压缩

RED_BALL = {"bbox0": [60, 210, 56, 56], "bbox1": [700, 210, 56, 56], "color": (200, 40, 40)}
BLUE_BALL = {"bbox": [420, 200, 64, 64], "color": (50, 90, 210)}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_OUT = os.path.join(OUT_DIR, "smoke_video.mp4")
SCENE_OUT = os.path.join(OUT_DIR, "scene_smoke.json")


def red_bbox(i: int):
    t = i / (N - 1)
    b0, b1 = RED_BALL["bbox0"], RED_BALL["bbox1"]
    return [b0[k] + (b1[k] - b0[k]) * t for k in range(4)]


def make_frame(i: int) -> np.ndarray:
    img = Image.new("RGB", (W, H), (235, 231, 220))
    d = ImageDraw.Draw(img)
    # 地面与淡格线(给 VACE 一点结构可参照)
    d.rectangle([0, 380, W, H], fill=(190, 185, 175))
    for x in range(0, W, 80):
        d.line([(x, 380), (x, H)], fill=(175, 170, 160), width=1)
    x, y, w, h = red_bbox(i)
    d.ellipse([x, y, x + w, y + h], fill=RED_BALL["color"])
    bx, by, bw, bh = BLUE_BALL["bbox"]
    d.ellipse([bx, by, bx + bw, by + bh], fill=BLUE_BALL["color"])
    return np.array(img)


def main():
    writer = imageio.get_writer(VIDEO_OUT, fps=FPS, macro_block_size=1)
    for i in range(N):
        writer.append_data(make_frame(i))
    writer.close()
    trajectory = [[round(i / FPS, 3), [round(v, 1) for v in red_bbox(i)]]
                  for i in range(0, N, 4)]   # 每 4 帧一个定位点
    scene = {
        "objects": [
            {"id": "object_01", "name": "红球", "aliases": ["红色球"],
             "localization": {"bbox": RED_BALL["bbox0"], "trajectory": trajectory}},
            {"id": "object_02", "name": "蓝球", "aliases": ["蓝色球"],
             "localization": {"bbox": BLUE_BALL["bbox"]}},
        ],
        "regions": [],
        "video": {"duration_seconds": DURATION, "fps": FPS, "frame_count": N,
                  "width": W, "height": H},
        "timeline": [
            {"event_id": "t_01", "verb": "roll", "participants": ["object_01"],
             "window": [0.0, DURATION], "name": "红球滚动", "conditions": [],
             "text": "红球从左向右滚动"},
            {"event_id": "t_02", "verb": "stationary", "participants": ["object_02"],
             "window": [0.0, DURATION], "name": "蓝球静止", "conditions": [],
             "text": "蓝球全程保持静止"},
        ],
    }
    with open(SCENE_OUT, "w", encoding="utf-8") as f:
        json.dump(scene, f, ensure_ascii=False, indent=2)
    print("video ->", VIDEO_OUT)
    print("scene ->", SCENE_OUT)


if __name__ == "__main__":
    main()
