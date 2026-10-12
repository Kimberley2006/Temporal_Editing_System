"""代表性验证数据集:编辑依赖器的 8 个用例。

前 7 个对应规格表(同一对象跨事件/事件依赖被修改对象/时间修改影响后续安排/
空间修改影响交互/多指令冲突/级联影响遇到其他编辑/修改彼此独立),
第 8 个是「删除桌子后桌上的杯子」状态门控样例(杯子 8 秒后被人拿着,传播停止)。

EXTRACTIONS 是 Qwen3-VL-8B-Instruct 对各指令应返回的结构化拆解(脚本化应答,
与 prompts.py 定义的输出契约一致);make_mock_client() 用它构造离线客户端,
供测试与演示使用。接入真实模型时无需本文件,直接用 QwenVLClient。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edit_dependency import LLMError, MockQwenClient  # noqa: E402

SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "scene_dependency_demo.json")

CASES = [
    ("整段删除杯子。",
     "同一对象跨事件:删除要求关联到杯子的全部出现区间"),
    ("删除杯子。",
     "事件依赖被修改对象:后面的拿取、喝水动作需要修复,替代动作由规划器决定"),
    ("A 延迟 1 秒，B 必须在 A 结束后开始。",
     "时间修改影响后续安排:只检查有约束联系的 B,不默认延迟所有后续事件"),
    ("让红球换条路径，蓝球保持不变。",
     "空间修改影响交互:碰撞事件待定(新路径由规划器决定),保留蓝球保持要求"),
    ("整段删除杯子，同时把后半段杯子改成蓝色。",
     "多条指令相互冲突:删除与变色区间重叠,标记冲突,不能靠排序解决"),
    ("删除杯子，同时让持杯行走的人慢一点。",
     "级联影响遇到其他编辑:持杯动作需要修复,同时保留人物减速要求"),
    ("红球停留 1 秒，远处汽车变成蓝色。",
     "修改彼此独立:无关联则不触发额外修改,可同批编辑"),
    ("删除桌子。",
     "状态门控:桌上的杯子需重新安置;杯子被拿着之后的动作不因桌子被删而受影响"),
]

EXTRACTIONS = {
    "整段删除杯子，同时把后半段杯子改成蓝色。": {
        "edits": [
            {"action": "delete", "object_id": "object_03", "event_id": None,
             "region_id": None, "scope": {"kind": "all"}, "params": {},
             "quote": "整段删除杯子"},
            {"action": "recolor", "object_id": "object_03", "event_id": None,
             "region_id": None, "scope": {"kind": "window", "start_seconds": 7.5,
                                          "end_seconds": 15.0},
             "params": {"color": "蓝色"}, "quote": "把后半段杯子改成蓝色"},
        ],
        "preserves": [], "constraints": [], "unmatched": [],
    },
    "删除杯子，同时让持杯行走的人慢一点。": {
        "edits": [
            {"action": "delete", "object_id": "object_03", "event_id": None,
             "region_id": None, "scope": {"kind": "all"}, "params": {},
             "quote": "删除杯子"},
            {"action": "slow", "object_id": "object_05", "event_id": None,
             "region_id": None, "scope": {"kind": "all"}, "params": {},
             "quote": "让持杯行走的人慢一点"},
        ],
        "preserves": [], "constraints": [], "unmatched": [],
    },
    "A 延迟 1 秒，B 必须在 A 结束后开始。": {
        "edits": [
            {"action": "delay", "object_id": None, "event_id": "t_09",
             "region_id": None, "scope": {"kind": "all"},
             "params": {"seconds": 1}, "quote": "A 延迟 1 秒"},
        ],
        "preserves": [],
        "constraints": [
            {"kind": "starts_after_end_of", "subject_event_id": "t_10",
             "ref_event_id": "t_09", "quote": "B 必须在 A 结束后开始"},
        ],
        "unmatched": [],
    },
    "让红球换条路径，蓝球保持不变。": {
        "edits": [
            {"action": "repath", "object_id": "object_01", "event_id": None,
             "region_id": None, "scope": {"kind": "all"}, "params": {},
             "quote": "让红球换条路径"},
        ],
        "preserves": [
            {"object_id": "object_02", "aspects": ["behavior", "appearance"],
             "quote": "蓝球保持不变"},
        ],
        "constraints": [], "unmatched": [],
    },
    "红球停留 1 秒，远处汽车变成蓝色。": {
        "edits": [
            {"action": "stay", "object_id": "object_01", "event_id": None,
             "region_id": None, "scope": {"kind": "all"},
             "params": {"duration_seconds": 1}, "quote": "红球停留 1 秒"},
            {"action": "recolor", "object_id": "object_06", "event_id": None,
             "region_id": None, "scope": {"kind": "all"},
             "params": {"color": "蓝色"}, "quote": "远处汽车变成蓝色"},
        ],
        "preserves": [], "constraints": [], "unmatched": [],
    },
    "整段删除杯子。": {
        "edits": [
            {"action": "delete", "object_id": "object_03", "event_id": None,
             "region_id": None, "scope": {"kind": "all"}, "params": {},
             "quote": "整段删除杯子"},
        ],
        "preserves": [], "constraints": [], "unmatched": [],
    },
    "删除杯子。": {
        "edits": [
            {"action": "delete", "object_id": "object_03", "event_id": None,
             "region_id": None, "scope": {"kind": "all"}, "params": {},
             "quote": "删除杯子"},
        ],
        "preserves": [], "constraints": [], "unmatched": [],
    },
    "删除桌子。": {
        "edits": [
            {"action": "delete", "object_id": "object_04", "event_id": None,
             "region_id": None, "scope": {"kind": "all"}, "params": {},
             "quote": "删除桌子"},
        ],
        "preserves": [], "constraints": [], "unmatched": [],
    },
}

# 时间窗未知(t_11 放下杯子)的候选由 LLM 按事件时刻状态判定
JUDGE_ANSWER = {
    "judgments": [
        {"event_id": "t_11", "judgment": "propagated",
         "reason": "杯子被整段删除,「放下杯子」失去作用对象,动作需修复"},
    ]
}


def load_scene():
    import json
    with open(SCENE_PATH, encoding="utf-8") as f:
        return json.load(f)


def make_mock_client():
    """离线客户端:按指令子串返回脚本化拆解;传播判定返回固定应答。"""
    def handler(task: str, user: str) -> dict:
        if task == "extract_edits":
            for key, resp in EXTRACTIONS.items():
                if key in user:
                    return resp
            raise LLMError("llm_bad_response", "无该指令的脚本化拆解")
        if task == "judge_propagation":
            return JUDGE_ANSWER
        raise LLMError("llm_bad_response", "未知任务 %s" % task)
    return MockQwenClient(handler)
