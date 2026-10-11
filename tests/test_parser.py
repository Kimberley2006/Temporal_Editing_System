"""指令解析器测试:三种指令句式、对象绑定、事件结构与异常输出。

可直接 `python tests/test_parser.py` 运行,也可 `pytest tests/` 运行。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instruction_parser import InstructionParser  # noqa: E402

SCENE = {
    "objects": [
        {"id": "object_01", "name": "红球", "aliases": ["红色球"]},
        {"id": "object_02", "name": "蓝球", "aliases": ["蓝色球"]},
    ],
    "regions": [
        {"id": "region_01", "name": "蓝色区域", "aliases": ["蓝区"]},
    ],
    "video": {"duration_seconds": 5.0, "fps": 16, "frame_count": 80},
}

P = InstructionParser()


# ---------------------------------------------------------------- 正常解析

def test_arrive_stay_preserve():
    """规格主例:到达 + 停留 + 蓝球保持不变。"""
    out = P.parse("红球在第 3 秒到达蓝色区域，并停留 1 秒；蓝球保持不变。", SCENE)
    assert out["status"] == "ok", out
    assert out["target_object_id"] == "object_01"
    assert out["events"] == [
        {"event_id": "event_01", "type": "arrive",
         "region_id": "region_01", "time_seconds": 3.0},
        {"event_id": "event_02", "type": "stay",
         "region_id": "region_01", "after_event_id": "event_01",
         "duration_seconds": 1.0},
    ]
    assert out["preserve"] == {
        "object_behavior_ids": ["object_02"],
        "target_identity": True,
        "background": True,
    }
    assert out["editable"] == ["target_path", "target_speed"]
    assert out["warnings"] == []
    assert out["provenance"]["preserve.object_behavior_ids.object_02"] == "instruction"
    assert out["provenance"]["preserve.target_identity"] == "task_default"
    assert out["provenance"]["editable"] == "task_default"


def test_arrive_only_chinese_numeral():
    out = P.parse("红球在第三秒的时候到达蓝色区域", SCENE)
    assert out["status"] == "ok"
    assert out["events"][0]["time_seconds"] == 3.0
    assert out["events"][0]["type"] == "arrive"


def test_stay_only_combined_clause():
    """类型 B:进入区域后停留,到达时间为空,不自行补上。"""
    out = P.parse("红球进入蓝色区域后停留 1 秒", SCENE)
    assert out["status"] == "ok"
    assert len(out["events"]) == 2
    arrive, stay = out["events"]
    assert arrive["type"] == "arrive" and arrive["time_seconds"] is None
    assert stay["type"] == "stay" and stay["duration_seconds"] == 1.0
    assert stay["after_event_id"] == arrive["event_id"]
    codes = [w["code"] for w in out["warnings"]]
    assert "arrive_time_unspecified" in codes


def test_decimal_and_alias():
    out = P.parse("红色球在第 3.5 秒到达蓝区", SCENE)
    assert out["status"] == "ok"
    assert out["events"][0]["time_seconds"] == 3.5
    assert out["events"][0]["region_id"] == "region_01"


def test_arrive_without_time():
    """未指定到达时间:字段为空并给出警告,不算错误。"""
    out = P.parse("红球到达蓝色区域", SCENE)
    assert out["status"] == "ok"
    assert out["events"][0]["time_seconds"] is None


def test_preserve_all_others():
    out = P.parse("红球在第 3 秒到达蓝色区域，其他对象保持不变", SCENE)
    assert out["status"] == "ok"
    assert out["preserve"]["object_behavior_ids"] == ["object_02"]
    assert out["provenance"]["preserve.object_behavior_ids.object_02"] == "instruction"


def test_background_preserve():
    out = P.parse("红球在第 3 秒到达蓝色区域，背景保持不变", SCENE)
    assert out["status"] == "ok"
    assert out["preserve"]["background"] is True
    assert out["provenance"]["preserve.background"] == "instruction"


# ---------------------------------------------------------------- 异常输出

def test_ambiguous_object():
    scene = json.loads(json.dumps(SCENE, ensure_ascii=False))
    scene["objects"].append({"id": "object_03", "name": "红球"})
    out = P.parse("红球在第 3 秒到达蓝色区域", scene)
    assert out["status"] == "error"
    assert out["error"]["code"] == "ambiguous_object"
    assert sorted(out["error"]["candidates"]) == ["object_01", "object_03"]


def test_unknown_object():
    out = P.parse("绿球在第 3 秒到达蓝色区域", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "unknown_object"
    assert out["error"]["name"] == "绿球"


def test_unknown_region():
    out = P.parse("红球在第 3 秒到达红色区域", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "unknown_region"


def test_time_out_of_range():
    out = P.parse("红球在第 8 秒到达蓝色区域", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "time_out_of_range"
    assert out["error"]["video_duration"] == 5.0


def test_stay_window_out_of_range():
    out = P.parse("红球在第 4.8 秒到达蓝色区域，并停留 1 秒", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "time_out_of_range"
    assert out["error"]["window_end"] == 5.8


def test_unsupported_operation():
    out = P.parse("红球旋转一圈", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "unsupported_operation"
    assert out["error"]["operation"] == "旋转"


def test_multi_target():
    out = P.parse("红球在第 3 秒到达蓝色区域，蓝球在第 4 秒到达蓝色区域", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "multi_target_unsupported"


def test_stay_without_arrive():
    out = P.parse("红球停留 1 秒", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "stay_without_arrive"


def test_preserve_conflict():
    out = P.parse("红球在第 3 秒到达蓝色区域，红球保持不变", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "preserve_conflict"


def test_unparseable():
    out = P.parse("今天天气不错", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "unparseable"


def test_empty_instruction():
    out = P.parse("  ", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "empty_instruction"


def test_invalid_scene():
    out = P.parse("红球在第 3 秒到达蓝色区域", {"objects": [
        {"id": "object_01", "name": "红球"},
        {"id": "object_01", "name": "另一个红球"}]})
    assert out["status"] == "error"
    assert out["error"]["code"] == "invalid_scene"


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
    print("\n%d failed" % failures if failures else "\nall tests passed")
    sys.exit(1 if failures else 0)
