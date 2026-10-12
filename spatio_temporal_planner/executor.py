"""Wan2.1-VACE 执行器:消费物化后的 VACE 任务,逐 pass 生成编辑视频。

重依赖(diffsynth/torch)全部惰性导入,保持规划器包其余部分纯标准库可测。
权重直载本地目录(/cache/wuhan/Wan2.1-VACE-1.3B),redirect_common_files=False
避免把 .pth 重定向到远端 safetensors。
"""

import os
from typing import List, Optional

DEFAULT_MODEL_DIR = "/cache/wuhan/Wan2.1-VACE-1.3B"


class VACEExecutor:
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, device: str = "cuda:0",
                 torch_dtype_name: str = "bfloat16"):
        self.model_dir = model_dir
        self.device = device
        self.torch_dtype_name = torch_dtype_name
        self._pipe = None

    # ---------------------------------------------------------------- 模型加载

    def load(self):
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffsynth.core.loader.config import ModelConfig
        from diffsynth.pipelines.wan_video import WanVideoPipeline

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "float32": torch.float32}[self.torch_dtype_name]
        md = self.model_dir
        model_configs = [
            ModelConfig(path=os.path.join(md, "models_t5_umt5-xxl-enc-bf16.pth")),
            ModelConfig(path=os.path.join(md, "diffusion_pytorch_model.safetensors")),
            ModelConfig(path=os.path.join(md, "Wan2.1_VAE.pth")),
        ]
        pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=dtype, device=self.device,
            model_configs=model_configs,
            tokenizer_config=ModelConfig(path=os.path.join(md, "google", "umt5-xxl")),
            redirect_common_files=False)
        self._pipe = pipe
        return pipe

    # ---------------------------------------------------------------- 执行

    def run_pass(self, video_path: str, jobs: List[dict], out_path: str,
                 prompt_override: Optional[str] = None,
                 num_frames_limit: Optional[int] = None,
                 vace_scale: float = 1.0, seed: int = 0,
                 height: Optional[int] = None, width: Optional[int] = None):
        """执行一个编辑遍:输入视频 + 任务 mask → 输出编辑后视频。"""
        from PIL import Image

        pipe = self.load()
        frames, meta = _read_video(video_path)
        H, W = meta["height"], meta["width"]
        height = height or H
        width = width or W
        if num_frames_limit and len(frames) > num_frames_limit:
            step = len(frames) / num_frames_limit
            frames = [frames[int(i * step)] for i in range(num_frames_limit)]

        masks = _build_masks(jobs, len(frames), meta)
        prompts = [j["prompt"] for j in jobs if j.get("prompt")]
        prompt = prompt_override or "; ".join(dict.fromkeys(prompts)) or \
            "high quality video"

        vace_frames = [Image.fromarray(f).convert("RGB") for f in frames]
        # VACE 的 mask 走与视频相同的预处理,需三通道;白=重生成区,黑=保持区
        vace_masks = [Image.fromarray(m).convert("RGB") for m in masks]
        output = pipe(
            prompt=prompt,
            vace_video=vace_frames,
            vace_video_mask=vace_masks,
            vace_scale=vace_scale,
            height=height, width=width, num_frames=len(vace_frames),
            seed=seed, cfg_scale=5.0,
        )
        generated = _to_frames(output, H, W)
        # 像素级合成:保持区直接用原帧(确定性保持,不受模型漂移影响),
        # 编辑区用生成结果,边界羽化过渡 —— "最少编辑改动"在执行层的落地
        frames_out = _composite(frames, generated, masks)
        _write_video(out_path, frames_out, meta["fps"])
        return {"out_path": out_path, "num_frames": len(frames_out),
                "prompt": prompt, "jobs": len(jobs)}


# ---------------------------------------------------------------- 视频/mask 工具(标准库 + numpy/PIL,执行时才导入)

def _read_video(path: str):
    import numpy as np
    from PIL import Image
    import imageio.v2 as imageio
    reader = imageio.get_reader(path)
    frames = []
    meta = reader.get_meta_data()
    fps = meta.get("fps", 16) or 16
    for f in reader:
        img = Image.fromarray(np.array(f)).convert("RGB")
        frames.append(np.array(img))
    reader.close()
    size = Image.fromarray(frames[0]).size
    return frames, {"fps": fps, "width": size[0], "height": size[1]}


def _build_masks(jobs: List[dict], n_frames: int, meta: dict) -> "numpy.ndarray":
    """按任务几何画逐帧 mask:白=重生成区,黑=保持区;无几何的任务不占 mask。

    轨迹关键帧之间做线性插值,保证逐帧 bbox 连续无间隙。
    """
    import numpy as np
    from PIL import Image, ImageDraw
    W, H = meta["width"], meta["height"]
    masks = np.zeros((n_frames, H, W), dtype=np.uint8)
    for job in jobs:
        spec = job.get("mask")
        if not spec or not spec.get("bbox"):
            continue
        f0, f1 = job.get("frame_range") or (0, n_frames)
        f0, f1 = max(0, f0), min(n_frames, f1)
        if f1 <= f0:
            continue
        base = list(spec["bbox"])
        shape = spec.get("shape", "rect")
        per_frame = _interp_trajectory(spec.get("trajectory"), meta["fps"], f0, f1, base)
        for i in range(f0, f1):
            bbox = per_frame[i - f0]
            img = Image.new("L", (W, H), 0)
            d = ImageDraw.Draw(img)
            x, y, w, h = bbox
            if shape == "ellipse":
                d.ellipse([x, y, x + w, y + h], fill=255)
            else:
                d.rectangle([x, y, x + w, y + h], fill=255)
            masks[i] = np.maximum(masks[i], np.array(img))
    return masks


def _interp_trajectory(traj, fps: float, f0: int, f1: int, base):
    """轨迹关键帧 [(t, bbox)] 线性插值为逐帧 bbox;无轨迹时全程用 base。"""
    n = f1 - f0
    if not traj:
        return [base] * n
    keys = sorted((t * fps, [float(v) for v in b]) for t, b in traj)

    def at(fi: float):
        t = fi
        if t <= keys[0][0]:
            return keys[0][1]
        if t >= keys[-1][0]:
            return keys[-1][1]
        for k in range(len(keys) - 1):
            t0, b0 = keys[k]
            t1, b1 = keys[k + 1]
            if t0 <= t <= t1:
                a = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                return [b0[j] + (b1[j] - b0[j]) * a for j in range(4)]
        return keys[-1][1]
    return [at(float(i)) for i in range(f0, f1)]


def _composite(src_frames, gen_frames, masks, feather_px: int = 5):
    """编辑区取生成帧、保持区取原始帧;mask 边缘羽化过渡。"""
    import numpy as np
    from PIL import Image, ImageFilter
    n = min(len(src_frames), len(gen_frames), len(masks))
    out = []
    for i in range(n):
        m = Image.fromarray(masks[i]).filter(
            ImageFilter.GaussianBlur(feather_px))
        m = np.asarray(m, dtype=np.float32) / 255.0
        src = src_frames[i].astype(np.float32)
        gen = gen_frames[i].astype(np.float32)
        if gen.shape[:2] != m.shape:
            gen = np.array(Image.fromarray(
                gen.astype(np.uint8)).resize((m.shape[1], m.shape[0])))
            gen = gen.astype(np.float32)
        out.append((src * (1.0 - m[..., None]) + gen * m[..., None])
                   .astype(np.uint8))
    return out


def _to_frames(output, H: int, W: int):
    """兼容不同版本的管线输出(list[PIL] / list[tensor] / 带 frames 属性)。"""
    import numpy as np
    from PIL import Image
    items = output
    for attr in ("frames", "images", "videos"):
        items = getattr(items, attr, None) or items
        if items is not output and items:
            break
    if items and isinstance(items[0], (list, tuple)):
        items = items[0]
    frames = []
    for f in items:
        if isinstance(f, Image.Image):
            frames.append(np.array(f.convert("RGB").resize((W, H))))
        else:
            arr = f.detach().cpu().float().numpy()
            if arr.ndim == 3 and arr.shape[0] in (1, 3):
                arr = arr.transpose(1, 2, 0)
            arr = (arr + 1.0) * 127.5 if arr.max() <= 1.5 else arr
            frames.append(np.clip(arr, 0, 255).astype("uint8"))
    return frames


def _write_video(path: str, frames, fps: float):
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()
