"""时空规划器独立测试:不依赖上游模块运行时,直接构造任务包/依赖器输出验证。

两组用例:
  A. 纯独立(手工任务包):排程/冲突跳过/时间不可行/修复模板/验证器;
  B. 链路(离线 Mock 依赖器输出 → 规划器):代表数据集用例的端到端结论。

可直接 `python tests/test_planner.py` 运行,也可 `pytest tests/` 运行。
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "examples"))

from dependency_cases import load_scene, make_mock_client  # noqa: E402
from edit_dependency import EditDependencyAnalyzer  # noqa: E402
from spatio_temporal_planner import SpatioTemporalPlanner  # noqa: E402

SCENE = load_scene()
TIMELINE = SCENE["timeline"]
PLANNER = SpatioTemporalPlanner(scene=SCENE)
DEP = EditDependencyAnalyzer(client=make_mock_client())


def dep_ok(edits, order=None, conflicts=None, uncertainties=None, repairs=None,
           preserve=None, events=None):
    """手工构造最小依赖器输出(独立测试不需要真的跑依赖器)。"""
    return {
        "status": "ok",
        "direct_edits": edits,
        "edit_order": order or [{"batch": i + 1, "op_ids": [e["op_id"]]}
                                for i, e in enumerate(edits)],
        "repair_requirements": repairs or [],
        "conflicts": conflicts or [],
        "uncertainties": uncertainties or [],
        "preserve": preserve or {"object_behavior_ids": [],
                                 "target_identity": True, "background": True},
        "events": events or [],
    }


def op(op_id, action, object_id=None, event_id=None, scope="all",
       start=None, end=None, params=None, quote=""):
    return {"op_id": op_id, "action": action, "object_id": object_id,
            "event_id": event_id, "region_id": None,
            "scope": {"kind": scope, "start_seconds": start, "end_seconds": end},
            "params": params or {}, "quote": quote}


def check_map(plan):
    return {c["name"]: c["passed"] for c in plan["verification"]}


# ================================================================ A. 纯独立测试

def test_parser_path_arrive_stay():
    """解析器输出(简单指令)直接进规划器:关键帧与停留窗口。"""
    parser_out = {
        "status": "ok", "target_object_id": "object_01",
        "events": [
            {"event_id": "event_01", "type": "arrive",
             "region_id": "region_01", "time_seconds": 3.0},
            {"event_id": "event_02", "type": "stay", "region_id": "region_01",
             "after_event_id": "event_01", "duration_seconds": 1.0},
        ],
        "preserve": {"object_behavior_ids": ["object_02"],
                     "target_identity": True, "background": True},
    }
    plan = PLANNER.plan(parser_output=parser_out, scene=SCENE)
    assert plan["status"] == "ok", plan
    assert plan["edit_passes"] == 1
    kf = plan["passes"][0]["keyframes"]
    assert kf[0]["t"] == 3.0 and kf[-1]["t"] == 4.0     # arrive 3s, stay 至 4s
    assert all(check_map(plan).values())


def test_delay_reschedules_downstream_only():
    """A 延迟 1 秒:仅 after_end_of 关联的 B 后移,其他事件窗口不动。"""
    out = dep_ok([op("op_01", "delay", event_id="t_09", params={"seconds": 1})])
    plan = PLANNER.plan(dependency_output=out, scene=SCENE)
    assert plan["status"] == "ok"
    # t_09 [2,5] -> [3,6];t_10 依赖 t_09 结束,后移到 [6,8];无关事件 t_01 保持 [0,8]
    sched = {(s["ref"], s.get("start"), s.get("end")) for s in plan["passes"][0]["scheduled"]}
    assert ("t_10", 6.0, 8.0) in sched
    assert ("t_01", 0.0, 8.0) in sched
    assert all(check_map(plan).values())


def test_slow_default_factor():
    """「慢一点」未给数值:默认系数 1.25 拉伸对象事件。"""
    out = dep_ok([op("op_01", "slow", object_id="object_05")])
    plan = PLANNER.plan(dependency_output=out, scene=SCENE)
    assert plan["status"] == "ok"
    sched = {s["ref"]: (s["start"], s["end"])
             for s in plan["passes"][0]["scheduled"] if s["kind"] == "event"}
    assert sched["t_09"] == (2.0, 5.75)     # [2,5] 时长 3s × 1.25 -> [2,5.75]
    assert all(check_map(plan).values())


def test_conflict_skip_policy():
    """冲突策略:涉冲突编辑全部跳过(反馈用户裁决),无冲突部分照常执行。"""
    out = dep_ok(
        [op("op_01", "delete", object_id="object_03", quote="整段删除杯子"),
         op("op_02", "recolor", object_id="object_03", quote="把后半段杯子改成蓝色"),
         op("op_03", "recolor", object_id="object_06", quote="远处汽车变成蓝色")],
        order=[{"batch": 1, "op_ids": ["op_01"]},
               {"batch": 2, "op_ids": ["op_02", "op_03"]}],
        conflicts=[{"conflict_id": "c_01", "op_ids": ["op_01", "op_02"],
                    "kind": "scope_overlap", "message": "删除与变色区间重叠",
                    "resolvable_by_ordering": False}])
    plan = PLANNER.plan(dependency_output=out, scene=SCENE)
    assert plan["status"] == "ok"
    # 涉冲突两项都跳过并反馈用户;无关的 op_03 照常执行
    assert sorted(o["op_id"] for o in plan["skipped_ops"]) == ["op_01", "op_02"]
    assert [p["op_ids"] for p in plan["passes"]] == [["op_03"]]
    assert plan["edit_passes"] == 1
    esc = plan["escalated_conflicts"][0]
    assert esc["kind"] == "scope_overlap"
    assert "已跳过" in esc["user_feedback"] and "建议" in esc["user_feedback"]
    assert all(check_map(plan).values())


def test_preserve_violation_skipped():
    """保持违例:编辑蓝球被跳过,蓝球保持不被触及。"""
    out = dep_ok(
        [op("op_01", "recolor", object_id="object_02", quote="蓝球变成红色")],
        conflicts=[{"conflict_id": "c_01", "op_ids": ["op_01"],
                    "kind": "preserve_violation", "message": "蓝球被要求保持不变",
                    "resolvable_by_ordering": False}])
    plan = PLANNER.plan(dependency_output=out, scene=SCENE)
    assert plan["status"] == "ok"
    assert plan["passes"] == [] or all(
        p["op_ids"] == [] for p in plan["passes"]) or plan["edit_passes"] == 0
    assert len(plan["skipped_ops"]) == 1
    assert check_map(plan)["preserve_intact"] is True


def test_time_infeasible_flagged():
    """时间不可行:延迟把依赖事件挤出视频末尾 → violation + 验证不通过,计划仍产出。"""
    tight = {"objects": SCENE["objects"], "regions": SCENE["regions"],
             "video": {"duration_seconds": 6.0},
             "timeline": [
                 {"event_id": "t_09", "verb": "walk", "participants": ["object_05"],
                  "window": [2.0, 5.0], "name": "A"},
                 {"event_id": "t_10", "verb": "wave", "participants": ["object_05"],
                  "window": [5.0, 7.0], "name": "B",
                  "conditions": [{"kind": "after_end_of", "refs": ["t_09"]}]}]}
    out = dep_ok([op("op_01", "delay", event_id="t_09", params={"seconds": 3})])
    plan = PLANNER.plan(dependency_output=out, scene=tight)
    assert plan["status"] == "ok"                       # 计划产出但不放行
    assert any(v["kind"] == "time_out_of_range" for v in plan["violations"])
    assert check_map(plan)["windows_in_range"] is False
    assert check_map(plan)["constraints_satisfied"] is True


def test_repair_minimal_templates():
    """修复模板:动作修复=删除+桥接;失支撑=位置保持;已覆盖区间=并入删除。"""
    out = dep_ok(
        [op("op_01", "delete", object_id="object_03")],
        repairs=[
            {"repair_id": "repair_01", "event_id": "t_02", "object_id": "object_03",
             "source_op_ids": ["op_01"],
             "requirement": "事件动作需重新设计(object_03 已删除),替代动作由规划器决定",
             "reason": "成立条件 requires_present 引用了被删除对象", "window": [8.0, 9.0]},
            {"repair_id": "repair_02", "event_id": "t_01", "object_id": "object_03",
             "source_op_ids": ["op_01"],
             "requirement": "对象 object_03 在该区间失去来自 object_04 的支撑,需重新安置",
             "reason": "支撑关系被删除破坏", "window": [0.0, 8.0]},
        ])
    plan = PLANNER.plan(dependency_output=out, scene=SCENE)
    res = {r["repair_id"]: r for p in plan["passes"] for r in p["repair_resolutions"]}
    assert res["repair_01"]["resolved_as"] == "remove_action_bridge"
    assert res["repair_01"]["segments"][0]["spec_type"] == "remove_action_bridge"
    assert res["repair_02"]["resolved_as"] == "hold_position"
    assert res["repair_02"]["segments"][0]["start"] == 0.0
    assert all(check_map(plan).values())


def test_repair_covered_by_delete():
    """删除已覆盖的出现区间:修复标记为并入删除,不产生额外段。"""
    out = dep_ok(
        [op("op_01", "delete", object_id="object_03")],
        repairs=[{"repair_id": "repair_01", "event_id": "t_11", "object_id": "object_03",
                  "source_op_ids": ["op_01"],
                  "requirement": "事件需按「delete」编辑后的状态重新检查",
                  "reason": "对象被删除,其全部出现区间一并失效", "window": None}])
    plan = PLANNER.plan(dependency_output=out, scene=SCENE)
    res = [r for p in plan["passes"] for r in p["repair_resolutions"]]
    assert res[0]["resolved_as"] == "recheck_in_pass"


def test_repath_uncertainty_pending():
    """repath 的空间求解挂起 Phase 2,不确定项显式登记不丢失。"""
    out = dep_ok(
        [op("op_01", "repath", object_id="object_01", quote="让红球换条路径")],
        uncertainties=[{"uncertainty_id": "u_01", "subject": "op_01 × t_05",
                        "question": "编辑「让红球换条路径」对事件 t_05 的影响待定",
                        "missing_evidence": ["新路径由规划器决定"]}])
    plan = PLANNER.plan(dependency_output=out, scene=SCENE)
    assert plan["status"] == "ok"
    assert any(s["spec_type"] == "spatial_pending"
               for p in plan["passes"] for s in p["segments"])
    assert plan["pending_uncertainties"] and "Phase 2" in plan["pending_uncertainties"][0]["handling"]
    assert check_map(plan)["uncertainties_handled"] is True


def test_upstream_not_ok_rejected():
    bad = {"status": "error", "error": {"code": "llm_unavailable", "message": "x"}}
    out = PLANNER.plan(dependency_output=bad)
    assert out["status"] == "error"
    assert out["error"]["code"] == "upstream_not_ok"


def test_empty_input():
    out = PLANNER.plan()
    assert out["status"] == "error"
    assert out["error"]["code"] == "empty_input"


# ================================================================ B. 链路测试(Mock 依赖器 → 规划器)

def test_chain_case1_delete_all_intervals():
    dep = DEP.analyze("整段删除杯子。", SCENE)
    plan = PLANNER.plan(dependency_output=dep, scene=SCENE)
    assert plan["status"] == "ok"
    segs = [s for p in plan["passes"] for s in p["segments"]]
    windows = sorted((s["start"], s["end"]) for s in segs if s["spec_type"] == "delete_mask")
    assert windows == [(0.0, 8.0), (8.0, 9.0), (9.0, 11.0), (9.0, 13.0)]  # t_11 无窗不可物化
    assert all(check_map(plan).values())


def test_chain_case3_delay():
    dep = DEP.analyze("A 延迟 1 秒，B 必须在 A 结束后开始。", SCENE)
    plan = PLANNER.plan(dependency_output=dep, scene=SCENE)
    sched = {(s["ref"], s["start"], s["end"])
             for s in plan["passes"][0]["scheduled"] if s["kind"] == "event"}
    assert ("t_10", 6.0, 8.0) in sched
    assert all(check_map(plan).values())


def test_chain_case5_conflict_skip():
    dep = DEP.analyze("整段删除杯子，同时把后半段杯子改成蓝色。", SCENE)
    plan = PLANNER.plan(dependency_output=dep, scene=SCENE)
    # 整条指令的两处编辑互相冲突:全部跳过并反馈用户,不产出半套编辑
    assert sorted(o["op_id"] for o in plan["skipped_ops"]) == ["op_01", "op_02"]
    assert plan["edit_passes"] == 0
    assert plan["escalated_conflicts"][0]["user_feedback"]
    assert all(check_map(plan).values())


def test_chain_case6_two_passes_order_kept():
    dep = DEP.analyze("删除杯子，同时让持杯行走的人慢一点。", SCENE)
    plan = PLANNER.plan(dependency_output=dep, scene=SCENE)
    assert [p["op_ids"] for p in plan["passes"]] == [["op_01"], ["op_02"]]
    assert plan["edit_passes"] == 2
    assert all(check_map(plan).values())


def test_chain_case7_independent_single_pass():
    dep = DEP.analyze("红球停留 1 秒，远处汽车变成蓝色。", SCENE)
    plan = PLANNER.plan(dependency_output=dep, scene=SCENE)
    assert plan["edit_passes"] == 1
    assert sorted(plan["passes"][0]["op_ids"]) == ["op_01", "op_02"]
    types = {s["spec_type"] for p in plan["passes"] for s in p["segments"]}
    assert "recolor_mask" in types
    kf = plan["passes"][0]["keyframes"]
    assert any(k["object_id"] == "object_01" and k["t"] == 1.0 for k in kf)  # 停留 [0,1]
    assert all(check_map(plan).values())


def test_chain_case8_hold_position():
    dep = DEP.analyze("删除桌子。", SCENE)
    plan = PLANNER.plan(dependency_output=dep, scene=SCENE)
    segs = [s for p in plan["passes"] for s in p["segments"]]
    hold = [s for s in segs if s["spec_type"] == "hold_position"]
    assert hold and hold[0]["object_id"] == "object_03"
    assert (hold[0]["start"], hold[0]["end"]) == (0.0, 8.0)
    assert all(check_map(plan).values())


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
