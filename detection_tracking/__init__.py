"""检测与跟踪模块:真实视频 → 场景对象表 + timeline(自动生成,替代人工构建)。

流水线:
  1. 关键帧采样(等间隔) → Qwen3-VL-8B-Instruct 开放词表检测(对象名 + 归一化框);
  2. 跟踪器:同名 + IoU 关联,关键帧间线性插值 → 逐帧轨迹(localization.trajectory);
  3. 事件推导(纯几何规则):出现区间 / 移动 / 静止 / 空间接触 / 支撑关系
     → timeline 事件(带成立条件,与依赖器/规划器的输入格式一致);
  4. scene_builder 组装为 scene JSON(objects/regions/video/timeline)。

接口约定:client 复用 edit_dependency.llm 的 QwenVLClient/MockQwenClient
(complete_json 接口);视觉消息由本模块组装(base64 图像帧)。
"""

from .analyzer import DetectionPipeline, build_scene_from_video
from .schemas import Detection, TrackedObject

__all__ = ["DetectionPipeline", "build_scene_from_video", "Detection", "TrackedObject"]
