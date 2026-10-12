"""编辑依赖分析器:一个修改会影响另一个事件或修改时,判断影响是否传播、传播到哪里、
是否与用户的其他要求冲突,并给出编辑先后顺序(最少遍数、最少返工)。

流水线四步(与解析器同构,输入也相同:instruction + scene):
  1. 提取 —— Qwen3-VL-8B-Instruct 把指令拆解为结构化编辑(删除/延迟/变色/减速等
     丰富操作超出规则解析器句式,由 LLM 提取并绑定场景 ID);
  2. 传播 —— 规则 + LLM 状态判定:候选查找(同对象/成立条件/时序约束/交互)、
     时间门控(后面发生 ≠ 受影响)、级联一跳;
  3. 冲突 —— 保持要求违例、作用范围重叠、相反意图;冲突不能靠排序解决,只标记;
  4. 排序 —— 依赖边拓扑分批,输出 edit_order(直接编辑的执行顺序)。

给规划器的输出:direct_edits / affected_events / repair_requirements /
conflicts / uncertainties / edit_order。修复与替代动作的最终方案由规划器决定。
"""

import copy
from typing import List, Optional

from instruction_parser.defaults import TASK_DEFAULTS
from instruction_parser.schemas import Event, SceneInfo

from .llm import LLMError, QwenVLClient
from .prompts import EXTRACT_SYSTEM, EXTRACT_USER, scene_brief, timeline_brief
from .propagation import (PropagationContext, analyze_propagation, build_edit_order,
                          detect_conflicts)
from .schemas import (EditOp, PreserveRequirement, Scope, StatedConstraint,
                      SUPPORTED_ACTIONS, parse_timeline)


def _error(code: str, message: str, **details) -> dict:
    err = {"code": code, "message": message}
    err.update(details)
    return {"status": "error", "error": err}


def _warning(code: str, message: str, **details) -> dict:
    w = {"code": code, "message": message}
    w.update(details)
    return w


class EditDependencyAnalyzer:
    def __init__(self, client=None):
        """client: QwenVLClient / MockQwenClient;缺省按环境变量构造(本地 vLLM 或 DashScope)。"""
        self.client = client

    # 对外主入口(与 InstructionParser.parse 同签名) --------------------------------

    def analyze(self, instruction: str, scene) -> dict:
        raw_scene = scene
        if isinstance(scene, dict):
            try:
                timeline_data = scene.get("timeline")
                scene = SceneInfo.from_dict(scene)
            except (KeyError, ValueError, TypeError) as exc:
                return _error("invalid_scene", "场景对象表不合法: %s" % exc)
        elif isinstance(scene, SceneInfo):
            # SceneInfo 不携带 timeline;直接传入 SceneInfo 时无既有事件可用
            timeline_data = getattr(scene, "timeline", None)
            raw_scene = None
        else:
            return _error("invalid_scene", "scene 需为 dict 或 SceneInfo")
        if not instruction or not str(instruction).strip():
            return _error("empty_instruction", "指令为空")

        try:
            events = parse_timeline(timeline_data) if isinstance(timeline_data, list) else []
        except ValueError as exc:
            return _error("invalid_timeline", "timeline 不合法: %s" % exc)

        duration = scene.video.duration_seconds

        # 第 1 步:LLM 提取
        extracted = self._extract(str(instruction), scene, events, duration)
        if extracted["status"] == "error":
            return extracted["payload"]
        ops, preserves, constraints, warnings = (extracted["ops"], extracted["preserves"],
                                                 extracted["constraints"], extracted["warnings"])
        if not ops:
            return _error("no_edit_operation", "指令中未解析到任何编辑操作")

        # 第 2 步:传播
        ctx = PropagationContext(ops=ops, preserves=preserves, constraints=constraints,
                                 timeline=events, duration=duration, client=self.client)
        propagation = analyze_propagation(ctx)
        warnings.extend(ctx.warnings)

        # 第 3 步:冲突
        conflicts = detect_conflicts(ops, preserves, duration)

        # 第 4 步:排序
        order = build_edit_order(ops, propagation["affected"], propagation["repairs"], duration)
        if order["cyclic"]:
            conflicts_cyclic = _warning("edit_order_cyclic",
                                        "编辑间存在循环依赖,已按指令顺序兜底排序")
            warnings.append(conflicts_cyclic)

        return self._assemble(instruction, scene, events, ops, preserves, constraints,
                              propagation, conflicts, order, warnings)

    # 第 1 步:提取 ---------------------------------------------------------------

    def _extract(self, instruction: str, scene: SceneInfo, events: list,
                 duration: Optional[float], feedback: Optional[List[str]] = None) -> dict:
        client = self.client or QwenVLClient()
        system = EXTRACT_SYSTEM.format(
            scene_brief=scene_brief(scene),
            timeline_brief=timeline_brief(events),
            duration=duration if duration is not None else "未知")
        user = EXTRACT_USER.format(instruction=instruction)
        if feedback:
            user += "\n\n上一次输出存在以下问题,请修正后重新输出完整 JSON:\n- " + "\n- ".join(feedback)

        for attempt in range(2):
            try:
                answer = client.complete_json("extract_edits", system, user)
            except LLMError as exc:
                return {"status": "error", "payload": _error(exc.code, "LLM 提取失败:%s" % exc.message)}
            issues = []
            parsed = self._parse_extraction(answer, scene, events, duration)
            issues.extend(parsed["issues"])
            if not issues:
                return {"status": "ok", "ops": parsed["ops"],
                        "preserves": parsed["preserves"],
                        "constraints": parsed["constraints"],
                        "warnings": parsed["warnings"]}
            if attempt == 0:
                feedback = issues
                continue
            return {"status": "error",
                    "payload": _error("llm_extraction_invalid",
                                      "LLM 提取结果校验未通过:%s" % ";".join(issues),
                                      issues=issues)}
        return {"status": "error",
                "payload": _error("llm_extraction_invalid", "LLM 提取结果校验未通过")}

    def _parse_extraction(self, answer: dict, scene: SceneInfo, events: list,
                          duration: Optional[float]) -> dict:
        issues: List[str] = []
        warnings: List[dict] = []
        obj_ids = {o.id for o in scene.objects}
        region_ids = {r.id for r in scene.regions}
        event_ids = {e.event_id for e in events}

        ops: List[EditOp] = []
        for idx, raw in enumerate(answer.get("edits", []), start=1):
            op_id = "op_%02d" % idx
            action = raw.get("action", "")
            if action not in SUPPORTED_ACTIONS:
                issues.append("%s: 不支持的操作「%s」,支持 %s" % (op_id, action, "/".join(SUPPORTED_ACTIONS)))
                continue
            object_id = raw.get("object_id")
            event_id = raw.get("event_id")
            region_id = raw.get("region_id")
            if object_id is not None and object_id not in obj_ids:
                issues.append("%s: 对象 %s 不在场景对象表中" % (op_id, object_id))
            if event_id is not None and event_id not in event_ids:
                issues.append("%s: 事件 %s 不在 timeline 中" % (op_id, event_id))
            if region_id is not None and region_id not in region_ids:
                issues.append("%s: 区域 %s 不在场景对象表中" % (op_id, region_id))
            try:
                scope = Scope.from_dict(raw.get("scope"))
            except ValueError as exc:
                issues.append("%s: %s" % (op_id, exc))
                continue
            if scope.kind == "window" and duration is not None:
                if scope.end_seconds > duration or scope.start_seconds < 0:
                    issues.append("%s: 编辑窗口 [%g, %g] 超出视频时长 %g 秒"
                                  % (op_id, scope.start_seconds, scope.end_seconds, duration))
            params = raw.get("params") or {}
            if not isinstance(params, dict):
                issues.append("%s: params 需为对象" % op_id)
                params = {}
            if action in ("delete", "recolor", "repath") and object_id is None:
                issues.append("%s: %s 操作需要 object_id" % (op_id, action))
            if action == "delay":
                if event_id is None:
                    issues.append("%s: delay 操作需要 event_id" % op_id)
                seconds = params.get("seconds")
                if not isinstance(seconds, (int, float)) or seconds <= 0:
                    issues.append("%s: delay 需要 params.seconds > 0" % op_id)
            if action in ("slow", "speed_up") and object_id is None and event_id is None:
                issues.append("%s: %s 操作需要 object_id 或 event_id" % (op_id, action))
            if action == "recolor" and not params.get("color"):
                issues.append("%s: recolor 需要 params.color" % op_id)
            if action == "arrive" and region_id is None:
                issues.append("%s: arrive 操作需要 region_id" % op_id)
            if action == "stay":
                d = params.get("duration_seconds")
                if not isinstance(d, (int, float)) or d <= 0:
                    issues.append("%s: stay 需要 params.duration_seconds > 0" % op_id)
            ops.append(EditOp(op_id=op_id, action=action, object_id=object_id,
                              event_id=event_id, region_id=region_id, scope=scope,
                              params=params, quote=raw.get("quote", "")))

        preserves: List[PreserveRequirement] = []
        for raw in answer.get("preserves", []):
            oid = raw.get("object_id")
            if oid not in obj_ids:
                issues.append("保持要求引用了未知对象 %s" % oid)
                continue
            aspects = [a for a in raw.get("aspects", ["behavior", "appearance"])
                       if a in ("behavior", "appearance")]
            preserves.append(PreserveRequirement(object_id=oid, aspects=aspects or ["behavior"],
                                                  quote=raw.get("quote", "")))

        constraints: List[StatedConstraint] = []
        for raw in answer.get("constraints", []):
            s, r = raw.get("subject_event_id"), raw.get("ref_event_id")
            if s not in event_ids or r not in event_ids:
                issues.append("约束引用了未知事件 %s/%s" % (s, r))
                continue
            constraints.append(StatedConstraint(
                kind=raw.get("kind", "starts_after_end_of"),
                subject_event_id=s, ref_event_id=r, quote=raw.get("quote", "")))

        for frag in answer.get("unmatched", []):
            warnings.append(_warning("unmatched_reference",
                                     "指令片段无法绑定到场景表/timeline:「%s」" % frag,
                                     fragment=frag))
        return {"issues": issues, "ops": ops, "preserves": preserves,
                "constraints": constraints, "warnings": warnings}

    # 输出组装 -----------------------------------------------------------------------

    def _assemble(self, instruction, scene, events, ops, preserves, constraints,
                  propagation, conflicts, order, warnings) -> dict:
        # arrive/stay 操作转成与解析器一致的事件结构(stay 关联上一个 arrive)
        parser_events = []
        last_arrive: Optional[Event] = None
        counter = 0
        for op in ops:
            if op.action == "arrive":
                counter += 1
                ev = Event(event_id="event_%02d" % counter, type="arrive",
                           region_id=op.region_id, time_seconds=op.params.get("time_seconds"))
                parser_events.append(ev)
                last_arrive = ev
            elif op.action == "stay":
                if last_arrive is not None and op.region_id in (None, last_arrive.region_id):
                    # 跟随到达事件的停留:与解析器同构,通过 after_event_id 关联
                    counter += 1
                    parser_events.append(Event(
                        event_id="event_%02d" % counter, type="stay",
                        region_id=last_arrive.region_id, after_event_id=last_arrive.event_id,
                        duration_seconds=op.params.get("duration_seconds")))
                # 独立的对象级停留(如「红球停留 1 秒」)保留在 direct_edits,不生成事件

        behavior_ids = []
        for p in preserves:
            if "behavior" in p.aspects and p.object_id not in behavior_ids:
                behavior_ids.append(p.object_id)

        provenance = {"direct_edits": "llm_extraction",
                      "edit_order": "rule:topological",
                      "repair_requirements": "rule:propagation"}
        for i in range(1, len(ops) + 1):
            provenance["direct_edits.op_%02d" % i] = "llm_extraction"
        for ae in propagation["affected"]:
            provenance["affected_events.%s" % ae.event_id] = ae.decided_by
        for p in preserves:
            provenance["preserve.object_behavior_ids.%s" % p.object_id] = "instruction"

        defaults = copy.deepcopy(TASK_DEFAULTS)
        return {
            "status": "ok",
            "instruction": instruction,
            "direct_edits": [op.to_dict() for op in ops],
            "events": [e.to_dict() for e in parser_events],
            "preserve": {
                "object_behavior_ids": behavior_ids,
                "target_identity": defaults["preserve"]["target_identity"],
                "background": defaults["preserve"]["background"],
            },
            "edit_order": [b.to_dict() for b in order["batches"]],
            "edit_passes": len(order["batches"]),
            "affected_events": [ae.to_dict() for ae in propagation["affected"]],
            "repair_requirements": [r.to_dict() for r in propagation["repairs"]],
            "conflicts": [c.to_dict() for c in conflicts],
            "uncertainties": [u.to_dict() for u in propagation["uncertainties"]],
            "warnings": warnings,
            "provenance": provenance,
        }
