"""用户指令解析器:把时间/空间编辑指令解析为结构化编辑任务(供规划器消费)。"""

from .defaults import TASK_DEFAULTS
from .parser import InstructionParser
from .schemas import Event, Region, SceneInfo, SceneObject, VideoInfo

__all__ = ["InstructionParser", "SceneInfo", "SceneObject", "Region",
           "VideoInfo", "Event", "TASK_DEFAULTS"]
