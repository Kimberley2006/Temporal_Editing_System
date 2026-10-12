"""编辑依赖器测试:代表性数据集(规格表七种情况 + 状态门控样例)验证
拆解后的用户指令能否转换为正确的编辑先后顺序,以及受影响事件/修复要求/
冲突/不确定项是否符合规格。离线运行,LLM 应答由 Mock 客户端脚本化提供。

可直接 `python tests/test_dependency.py` 运行,也可 `pytest tests/` 运行。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "examples"))

from dependency_cases import load_scene, make_mock_client  # noqa: E402
from edit_dependency import EditDependencyAnalyzer, LLMError, MockQwenClient  # noqa: E402

SCENE = load_scene()
A = EditDependencyAnalyzer(client=make_mock_client())


def affected_ids(out):
    return {ae["event_id"] for ae in out["affected_events"]}


def batch_ops(out):
    return [[op for op in b["op_ids"]] for b in out["edit_order"]]


# ------------------------------------------------ 情况 1:同一对象跨事件出现

def test_case1_same_object_all_intervals():
    out = A.analyze("整段删除杯子。", SCENE)
    assert out["status"] == "ok", out
    # 删除要求关联到同一杯子的全部出现区间
    assert affected_ids(out) == {"t_01", "t_02", "t_03", "t_04", "t_11"}
    assert all(ae["judgment"] == "propagated" for ae in out["affected_events"])
    # 时间窗未知的 t_11 由 LLM 判定
    t11 = next(ae for ae in out["affected_events"] if ae["event_id"] == "t_11")
    assert t11["decided_by"] == "llm"
    assert out["edit_passes"] == 1
    assert batch_ops(out) == [["op_01"]]


# ------------------------------------------------ 情况 2:事件依赖被修改的对象

def test_case2_dependent_actions_repaired():
    out = A.analyze("删除杯子。", SCENE)
    assert out["status"] == "ok", out
    repair_events = {r["event_id"] for r in out["repair_requirements"]}
    assert {"t_02", "t_03"} <= repair_events    # 拿起、喝水需要修复
    plan_repair = [r for r in out["repair_requirements"]
                   if r["event_id"] in ("t_02", "t_03")]
    assert all(r["decision_owner"] == "planner" for r in plan_repair)
    assert all("规划器决定" in r["requirement"] for r in plan_repair)


# ------------------------------------------------ 情况 3:时间修改影响后续安排

def test_case3_delay_checks_only_constrained():
    out = A.analyze("A 延迟 1 秒，B 必须在 A 结束后开始。", SCENE)
    assert out["status"] == "ok", out
    # 只检查有约束联系的 B(t_10),不默认所有后续事件都延迟
    assert affected_ids(out) == {"t_10"}
    ae = out["affected_events"][0]
    assert ae["relation"] == "temporal_constraint"
    assert any("时序约束" in r["requirement"] for r in out["repair_requirements"])


# ------------------------------------------------ 情况 4:空间修改影响交互

def test_case4_repath_checks_interaction_keeps_preserve():
    out = A.analyze("让红球换条路径，蓝球保持不变。", SCENE)
    assert out["status"] == "ok", out
    # 碰撞事件待定(新路径由规划器决定);红球自己的移动事件需重算
    by_id = {ae["event_id"]: ae for ae in out["affected_events"]}
    assert by_id["t_05"]["judgment"] == "uncertain"
    assert by_id["t_06"]["judgment"] == "propagated"
    assert len(out["uncertainties"]) >= 1
    # 蓝球保持要求被保留,且蓝球未被当作受影响对象去修改
    assert out["preserve"]["object_behavior_ids"] == ["object_02"]
    assert "t_07" not in by_id
    assert out["conflicts"] == []


# ------------------------------------------------ 情况 5:多条指令相互冲突

def test_case5_conflict_flagged_not_reordered():
    out = A.analyze("整段删除杯子，同时把后半段杯子改成蓝色。", SCENE)
    assert out["status"] == "ok", out
    assert len(out["conflicts"]) == 1
    c = out["conflicts"][0]
    assert c["kind"] == "scope_overlap"
    assert sorted(c["op_ids"]) == ["op_01", "op_02"]
    assert c["resolvable_by_ordering"] is False    # 不能仅靠编辑排序解决


# ------------------------------------------------ 情况 6:级联影响遇到其他编辑

def test_case6_cascade_keeps_slowdown_requirement():
    out = A.analyze("删除杯子，同时让持杯行走的人慢一点。", SCENE)
    assert out["status"] == "ok", out
    by_id = {ae["event_id"]: ae for ae in out["affected_events"]}
    # 持杯行走(t_04)因杯子删除需要修复
    assert "t_04" in by_id and by_id["t_04"]["judgment"] == "propagated"
    t04_repairs = [r for r in out["repair_requirements"] if r["event_id"] == "t_04"]
    assert any("删除" in r["requirement"] or "重新设计" in r["requirement"]
               for r in t04_repairs)
    # 人物减速要求保留为直接编辑,未被取消
    assert [e["action"] for e in out["direct_edits"]] == ["delete", "slow"]
    # 删除(改变存在前提)先做,减速基于删除后状态计算
    assert batch_ops(out) == [["op_01"], ["op_02"]]


# ------------------------------------------------ 情况 7:修改彼此独立

def test_case7_independent_edits_single_batch():
    out = A.analyze("红球停留 1 秒，远处汽车变成蓝色。", SCENE)
    assert out["status"] == "ok", out
    assert out["affected_events"] == []      # 无关联不触发额外修改
    assert out["conflicts"] == []
    assert out["uncertainties"] == []
    # 互不依赖 -> 同批执行,一遍完成
    assert len(out["edit_order"]) == 1
    b = out["edit_order"][0]
    assert sorted(b["op_ids"]) == ["op_01", "op_02"]
    assert b["parallel"] is True
    assert out["edit_passes"] == 1


# ------------------------------------------------ 状态门控:删除桌子的例子

def test_case8_state_gated_propagation():
    out = A.analyze("删除桌子。", SCENE)
    assert out["status"] == "ok", out
    # 桌上的杯子(t_01 区间)失去支撑需要重新安置
    assert affected_ids(out) == {"t_01"}
    repair = out["repair_requirements"][0]
    assert repair["object_id"] == "object_03"
    assert "重新安置" in repair["requirement"]
    assert repair["window"] == [0.0, 8.0]
    # 杯子 8 秒后被人拿着:拿取/喝水/持杯行走不因桌子被删而受影响
    assert not ({"t_02", "t_03", "t_04"} & affected_ids(out))


# ------------------------------------------------ 接口与异常

def test_same_input_interface_as_parser():
    """输入与指令解析器一致:(instruction, scene);输出含解析器同构字段。"""
    out = A.analyze("让红球换条路径，蓝球保持不变。", SCENE)
    for key in ("status", "events", "preserve", "warnings", "provenance"):
        assert key in out
    assert out["provenance"]["direct_edits"] == "llm_extraction"


def test_arrive_stay_ops_keep_parser_event_shape():
    client = make_mock_client()
    orig = client.handler

    def handler(task, user):
        if task == "extract_edits" and "到达" in user:
            return {"edits": [
                {"action": "arrive", "object_id": "object_01", "event_id": None,
                 "region_id": "region_01", "scope": {"kind": "all"},
                 "params": {"time_seconds": 3}, "quote": "红球在第 3 秒到达蓝色区域"},
                {"action": "stay", "object_id": "object_01", "event_id": None,
                 "region_id": None, "scope": {"kind": "all"},
                 "params": {"duration_seconds": 1}, "quote": "停留 1 秒"},
            ], "preserves": [], "constraints": [], "unmatched": []}
        return orig(task, user)
    out = EditDependencyAnalyzer(client=MockQwenClient(handler)).analyze(
        "红球在第 3 秒到达蓝色区域，并停留 1 秒", SCENE)
    assert out["status"] == "ok", out
    assert out["events"] == [
        {"event_id": "event_01", "type": "arrive",
         "region_id": "region_01", "time_seconds": 3},
        {"event_id": "event_02", "type": "stay", "region_id": "region_01",
         "after_event_id": "event_01", "duration_seconds": 1},
    ]


def test_extraction_retry_on_invalid_then_valid():
    calls = {"n": 0}

    def handler(task, user):
        if task == "extract_edits":
            calls["n"] += 1
            if calls["n"] == 1:
                return {"edits": [{"action": "delete", "object_id": "不存在的对象",
                                   "scope": {"kind": "all"}}],
                        "preserves": [], "constraints": []}
            return {"edits": [
                {"action": "delete", "object_id": "object_03",
                 "scope": {"kind": "all"}, "quote": "删除杯子"}],
                "preserves": [], "constraints": [], "unmatched": []}
        return {"judgments": []}
    out = EditDependencyAnalyzer(client=MockQwenClient(handler)).analyze(
        "删除杯子", SCENE)
    assert out["status"] == "ok", out
    assert calls["n"] == 2    # 校验失败带反馈重试了一次


def test_extraction_invalid_after_retry():
    def handler(task, user):
        return {"edits": [{"action": "teleport", "object_id": "object_03"}],
                "preserves": [], "constraints": []}
    out = EditDependencyAnalyzer(client=MockQwenClient(handler)).analyze(
        "删除杯子", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "llm_extraction_invalid"


def test_llm_unavailable():
    def handler(task, user):
        raise LLMError("llm_unavailable", "无法连接模型端点")
    out = EditDependencyAnalyzer(client=MockQwenClient(handler)).analyze(
        "删除杯子", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "llm_unavailable"


def test_empty_instruction():
    out = A.analyze("  ", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "empty_instruction"


def test_invalid_scene():
    out = A.analyze("删除杯子", {"objects": [
        {"id": "object_01", "name": "红球"},
        {"id": "object_01", "name": "另一个红球"}]})
    assert out["status"] == "error"
    assert out["error"]["code"] == "invalid_scene"


def test_invalid_timeline():
    scene = {"objects": SCENE["objects"], "video": SCENE["video"],
             "timeline": [{"event_id": "t_01", "window": [5.0, 2.0]}]}
    out = A.analyze("删除杯子", scene)
    assert out["status"] == "error"
    assert out["error"]["code"] == "invalid_timeline"


def test_no_edit_operation():
    def handler(task, user):
        return {"edits": [], "preserves": [], "constraints": []}
    out = EditDependencyAnalyzer(client=MockQwenClient(handler)).analyze(
        "今天天气不错", SCENE)
    assert out["status"] == "error"
    assert out["error"]["code"] == "no_edit_operation"


def test_unmatched_fragment_warning():
    def handler(task, user):
        if "绿球" in user:
            return {"edits": [
                {"action": "delete", "object_id": "object_03",
                 "scope": {"kind": "all"}, "quote": "删除杯子"}],
                "preserves": [], "constraints": [], "unmatched": ["绿球"]}
        return {"edits": [], "preserves": [], "constraints": []}
    out = EditDependencyAnalyzer(client=MockQwenClient(handler)).analyze(
        "删除杯子，绿球消失", SCENE)
    assert out["status"] == "ok", out
    assert any(w["code"] == "unmatched_reference" for w in out["warnings"])


def test_preserve_violation_conflict():
    def handler(task, user):
        return {"edits": [
            {"action": "delete", "object_id": "object_03",
             "scope": {"kind": "all"}, "quote": "删除杯子"},
            {"action": "recolor", "object_id": "object_02",
             "scope": {"kind": "all"}, "params": {"color": "红色"},
             "quote": "蓝球变成红色"},
        ], "preserves": [
            {"object_id": "object_02", "aspects": ["behavior", "appearance"],
             "quote": "蓝球保持不变"},
        ], "constraints": [], "unmatched": []}
    out = EditDependencyAnalyzer(client=MockQwenClient(handler)).analyze(
        "删除杯子，蓝球保持不变，蓝球变成红色", SCENE)
    assert out["status"] == "ok", out
    kinds = {c["kind"] for c in out["conflicts"]}
    assert "preserve_violation" in kinds


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
