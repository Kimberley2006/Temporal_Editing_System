"""计划级静态验证:执行计划交给执行器前的最后闸门。

检查项(全部可离线判定):
  windows_in_range      排程无越界/负起点
  constraints_satisfied after_end_of 约束在最终窗口下全部成立
  preserve_intact       保持对象未被任何已执行编辑触及
  ops_covered           未跳过的 op 恰好出现在一个批次中,跳过项未被执行
  repairs_resolved      每条修复要求都有落地方案(或来源被跳过而作废)
  uncertainties_handled 每个不确定项被消解或明确挂起(Phase 2 处理)
"""

from typing import List

from .schemas import ExecutionPlan, TaskPackage, VerificationCheck


def verify(package: TaskPackage, plan: ExecutionPlan, event_windows: dict) -> List[VerificationCheck]:
    checks: List[VerificationCheck] = []

    checks.append(VerificationCheck(
        name="windows_in_range",
        passed=not any(v["kind"] in ("time_out_of_range", "op_out_of_range",
                                     "negative_start") for v in plan.violations),
        detail=";".join(v["message"] for v in plan.violations) or "排程均在视频范围内"))

    bad_constraints = []
    for ev in package.events:
        final = event_windows.get(ev.event_id)
        if final is None:
            continue
        for cond in ev.conditions:
            if cond.get("kind") != "after_end_of":
                continue
            for ref in cond.get("refs", []):
                base = event_windows.get(ref)
                if base is not None and final[0] < base[1] - 1e-9:
                    bad_constraints.append("%s 需在 %s 结束(%g)后,实际起点 %g"
                                           % (ev.event_id, ref, base[1], final[0]))
    checks.append(VerificationCheck(
        name="constraints_satisfied", passed=not bad_constraints,
        detail=";".join(bad_constraints) or "全部 after_end_of 约束成立"))

    touched = {o.object_id for o in package.ops if not o.skipped and o.object_id}
    violated = sorted(touched & set(package.preserve_object_ids))
    checks.append(VerificationCheck(
        name="preserve_intact", passed=not violated,
        detail="保持对象被触及: %s" % violated if violated
        else "保持对象(%s)未被触及" % ",".join(package.preserve_object_ids)))

    executed = [oid for p in plan.passes for oid in p.op_ids]
    expected = [o.op_id for o in package.ops if not o.skipped]
    dup = {o for o in executed if executed.count(o) > 1}
    checks.append(VerificationCheck(
        name="ops_covered",
        passed=sorted(executed) == sorted(expected) and not dup
               and not any(o.skipped for o in package.ops if o.op_id in executed),
        detail="已执行 %s / 应执行 %s / 重复 %s" % (executed, expected, sorted(dup) or "无")))

    resolved_ids = {r.repair_id for p in plan.passes for r in p.repair_resolutions}
    dangling = [r.repair_id for r in package.repairs
                if r.repair_id not in resolved_ids
                and not all(_skipped(package, oid) for oid in r.source_op_ids)]
    checks.append(VerificationCheck(
        name="repairs_resolved", passed=not dangling,
        detail="未落地修复: %s" % dangling if dangling else "全部修复要求已落地"))

    handled = len(plan.resolved_uncertainties) + len(plan.pending_uncertainties)
    checks.append(VerificationCheck(
        name="uncertainties_handled",
        passed=handled >= len(package.uncertainties),
        detail="已消解 %d,挂起 %d,上游共 %d"
               % (len(plan.resolved_uncertainties),
                  len(plan.pending_uncertainties), len(package.uncertainties))))

    return checks


def _skipped(package: TaskPackage, op_id: str) -> bool:
    for o in package.ops:
        if o.op_id == op_id:
            return o.skipped
    return True
