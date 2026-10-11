"""用户指令解析器(基础版):固定句式 + 规则解析。

按规格分四步处理:
  1. 提取   —— 分句,抽出对象名、区域名、到达时间、停留时长、保持要求
  2. 绑定   —— 把"红球""蓝色区域"等文字绑定到场景对象表中的 ID
  3. 规范化 —— "第 3 秒""三秒时"统一为秒数;操作统一为 arrive/stay 事件类型
  4. 校验   —— 对象唯一、区域存在、时间在视频范围内、操作是否支持

第一版仅支持中文固定句式;之后替换为大模型解析时,输出保持同样的结构。
支持的三种指令(均可附带"X 保持不变"):
  A. 指定到达时间:红球在第 3 秒到达蓝色区域
  B. 指定停留时长:红球进入蓝色区域后停留 1 秒
  C. 到达 + 停留:  红球在第 3 秒到达蓝色区域,并停留 1 秒
"""

import copy
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .defaults import TASK_DEFAULTS
from .numbers import NUM_PATTERN, parse_number
from .schemas import Event, SceneInfo

# ---------------------------------------------------------------- 常量与正则

_ARRIVE_VERBS = ("到达", "进入")
_STAY_VERBS = ("停留", "待", "呆")
_STAY_ALT = "|".join(_STAY_VERBS)

_KEEP_RE = re.compile(r"不变|不动|原样|静止")
_KEEP_SPLIT_RE = re.compile(r"保持|维持|仍然|依然")
_PRESERVE_ALL_WORDS = ("其他", "其余", "别的")

# 暂不支持的操作关键词(旋转、变色、缩放等)
_UNSUPPORTED_OPS = ("旋转", "翻转", "转动", "变色", "变成", "染成", "缩放",
                    "放大", "缩小", "变大", "变小", "消失", "出现", "变形",
                    "复制", "克隆", "删除", "去掉", "加速", "减速", "变快", "变慢")

# 时间:"在第3秒""第3秒的时候""3秒时""于三秒之际"
_TIME_RE = re.compile(r"(?:在|于)?第?(?P<num>%s)秒(?:钟)?(?:的?时候|之际|时)?" % NUM_PATTERN)
# 时长:"停留1秒""待了三秒"
_DURATION_RE = re.compile(r"(?:%s)了?(?P<num>%s)秒(?:钟)?" % (_STAY_ALT, NUM_PATTERN))
# 到达子句中,区域区之后的边界词("进入蓝色区域后停留..."的"后")
_REGION_BOUNDARY_RE = re.compile(r"之?后|然后|并且?|再|同时")
# "之前/以内"等截止语义,第一版近似为"在该时刻到达"
_DEADLINE_RE = re.compile(r"之?前|以前|以?内|之?内")

# 分句符:注意不能包含 ASCII 句点,否则"第 3.5 秒"的小数点会截断句子
_CLAUSE_SPLIT_RE = re.compile(r"[，,；;。！!？?\n]+")
_CONNECTORS = "并并且然后之再同时而且还接着"
_PARTICLES = "请让把使得向对在于"


def _error(code: str, message: str, **details) -> dict:
    err = {"code": code, "message": message}
    err.update(details)
    return {"status": "error", "error": err}


def _warning(code: str, message: str, **details) -> dict:
    w = {"code": code, "message": message}
    w.update(details)
    return w


# ---------------------------------------------------------------- 中间结构

@dataclass
class _Clause:
    """提取阶段的产物:一个子句抽出的原始槽位(尚未绑定 ID、尚未数字化)。"""

    raw: str
    kind: str = "unknown"                      # arrive | stay | preserve | unsupported | unknown
    object_text: Optional[str] = None          # 对象区文字
    region_text: Optional[str] = None          # 区域区文字
    time_text: Optional[str] = None            # 到达时间的原始数字
    duration_text: Optional[str] = None        # 停留时长的原始数字
    preserve_all: bool = False                 # "其他对象保持不变"
    preserve_background: bool = False          # "背景保持不变"
    deadline_hint: bool = False                # "之前/以内"截止语义


@dataclass
class _BoundClause:
    """绑定 + 规范化之后的子句。"""

    clause: _Clause
    object_id: Optional[str] = None
    region_id: Optional[str] = None
    time_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None


# ---------------------------------------------------------------- 解析器

class InstructionParser:
    def __init__(self, task_defaults: Optional[dict] = None):
        self.defaults = task_defaults or copy.deepcopy(TASK_DEFAULTS)

    # 对外主入口 ------------------------------------------------------------

    def parse(self, instruction: str, scene) -> dict:
        """读取指令与场景对象表,输出结构化编辑任务(dict/JSON)。"""
        if isinstance(scene, dict):
            try:
                scene = SceneInfo.from_dict(scene)
            except (KeyError, ValueError, TypeError) as exc:
                return _error("invalid_scene", "场景对象表不合法: %s" % exc)
        if not isinstance(scene, SceneInfo):
            return _error("invalid_scene", "scene 需为 dict 或 SceneInfo")
        if not instruction or not str(instruction).strip():
            return _error("empty_instruction", "指令为空")

        text = re.sub(r"\s+", "", str(instruction))
        clauses = [c for c in _CLAUSE_SPLIT_RE.split(text) if c]

        # 第 1 步:提取
        slots = [self._extract(c) for c in clauses]

        # 第 2 步:绑定(名称 -> ID)
        bound: List[_BoundClause] = []
        target_id: Optional[str] = None
        preserve_ids: List[str] = []
        background_from_instruction = False
        for slot in slots:
            result = self._bind(slot, scene, target_id, preserve_ids)
            if result["status"] == "error":
                return result["payload"]
            if result["target_id"]:
                target_id = result["target_id"]
            preserve_ids = result["preserve_ids"]
            background_from_instruction = (
                background_from_instruction or slot.preserve_background)
            bound.append(result["clause"])

        # "其他对象保持不变":目标确定后展开为除目标外的全部对象
        if any(s.kind == "preserve" and s.preserve_all for s in slots):
            for oid in (o.id for o in scene.objects if o.id != target_id):
                if oid not in preserve_ids:
                    preserve_ids.append(oid)

        # 第 3 步:规范化(数字字符串 -> 秒数)
        norm = self._normalize(bound)
        if norm["status"] == "error":
            return norm["payload"]

        # 组装事件序列
        built = self._build_events(bound)
        if built["status"] == "error":
            return built["payload"]
        events, warnings = built["events"], built["warnings"]

        # 第 4 步:校验
        check = self._validate(events, scene)
        if check["status"] == "error":
            return check["payload"]
        warnings.extend(check["warnings"])

        if not events:
            if preserve_ids or background_from_instruction:
                return _error("no_edit_operation", "指令中未解析到任何编辑操作")
            unknown = [s.raw for s in slots if s.kind == "unknown"]
            return _error("unparseable", "指令无法匹配任何支持的句式", clauses=unknown)

        return self._assemble(target_id, events, preserve_ids,
                              background_from_instruction, warnings)

    # 第 1 步:提取 -----------------------------------------------------------

    def _extract(self, clause: str) -> _Clause:
        slot = _Clause(raw=clause)
        if any(op in clause for op in _UNSUPPORTED_OPS):
            slot.kind = "unsupported"
            return slot
        if any(v in clause for v in _ARRIVE_VERBS):
            slot.kind = "arrive"
        elif any(v in clause for v in _STAY_VERBS):
            slot.kind = "stay"
        elif _KEEP_RE.search(clause):
            slot.kind = "preserve"
        else:
            return slot

        if slot.kind == "arrive":
            vpos = min(clause.find(v) for v in _ARRIVE_VERBS if v in clause)
            before, after = clause[:vpos], clause[vpos + 2:]
            time_text = None
            for m in _TIME_RE.finditer(before):
                time_text = m.group("num")
            slot.time_text = time_text
            slot.deadline_hint = bool(time_text and _DEADLINE_RE.search(before))
            obj_zone = _TIME_RE.sub("", before).strip(_PARTICLES)
            slot.object_text = obj_zone or None
            region_zone = _REGION_BOUNDARY_RE.split(after)[0].strip(_PARTICLES)
            slot.region_text = region_zone or None
            dur = _DURATION_RE.search(clause)
            if dur:
                slot.duration_text = dur.group("num")
        elif slot.kind == "stay":
            dur = _DURATION_RE.search(clause)
            if dur:
                slot.duration_text = dur.group("num")
                obj_zone = clause[:dur.start()].strip(_CONNECTORS + _PARTICLES)
                slot.object_text = obj_zone or None
            else:
                slot.object_text = clause.strip(_CONNECTORS + _PARTICLES) or None
        elif slot.kind == "preserve":
            zone = _KEEP_SPLIT_RE.split(clause)[0].strip(_CONNECTORS + _PARTICLES)
            slot.preserve_background = "背景" in clause
            if any(w in zone for w in _PRESERVE_ALL_WORDS):
                slot.preserve_all = True
            elif zone and zone != "背景":
                slot.object_text = zone
        return slot

    # 第 2 步:绑定 -------------------------------------------------------------

    def _bind(self, slot: _Clause, scene: SceneInfo,
              target_id: Optional[str], preserve_ids: List[str]) -> dict:
        """返回 {status, clause, target_id, preserve_ids};失败时 {status:"error", payload}。"""
        bound = _BoundClause(clause=slot)

        if slot.kind == "unsupported":
            op = next(op for op in _UNSUPPORTED_OPS if op in slot.raw)
            return {"status": "error",
                    "payload": _error("unsupported_operation",
                                      "暂不支持该编辑操作「%s」,当前仅支持到达/停留类指令" % op,
                                      operation=op)}
        if slot.kind == "unknown":
            return {"status": "ok", "clause": bound, "target_id": None,
                    "preserve_ids": preserve_ids}

        if slot.kind in ("arrive", "stay"):
            if slot.object_text is not None:
                hits = self._match_names(scene.objects, slot.object_text)
                err = self._check_object_hits(hits, slot.object_text, slot.kind)
                if err:
                    return {"status": "error", "payload": err}
                if hits:
                    new_id = hits[0][0].id
                    if target_id is not None and new_id != target_id:
                        return {"status": "error",
                                "payload": _error("multi_target_unsupported",
                                                  "第一版仅支持编辑一个目标对象:「%s」与「%s」同时被要求编辑"
                                                  % (target_id, new_id),
                                                  target_ids=[target_id, new_id])}
                    bound.object_id = new_id
            elif slot.kind == "arrive":
                return {"status": "error",
                        "payload": _error("missing_target", "到达子句未指明编辑对象", clause=slot.raw)}

        if slot.kind == "arrive":
            if slot.region_text is None:
                return {"status": "error",
                        "payload": _error("missing_region", "到达子句未指明目标区域", clause=slot.raw)}
            r_hits = self._match_names(scene.regions, slot.region_text)
            if not r_hits:
                return {"status": "error",
                        "payload": _error("unknown_region", "场景中不存在区域「%s」" % slot.region_text,
                                          name=slot.region_text)}
            r_err = self._check_ambiguous(r_hits, "region")
            if r_err:
                return {"status": "error", "payload": r_err}
            bound.region_id = r_hits[0][0].id

        if slot.kind == "preserve":
            if not slot.preserve_all and slot.object_text is None and not slot.preserve_background:
                return {"status": "error",
                        "payload": _error("missing_preserve_object",
                                          "保持要求缺少明确对象", clause=slot.raw)}
            if slot.object_text is not None:
                hits = self._match_names(scene.objects, slot.object_text)
                err = self._check_object_hits(hits, slot.object_text, slot.kind)
                if err:
                    return {"status": "error", "payload": err}
                if hits:
                    new_id = hits[0][0].id
                    if new_id == target_id:
                        return {"status": "error",
                                "payload": _error("preserve_conflict",
                                                  "对象「%s」既是编辑目标又被要求保持不变" % new_id,
                                                  object_id=new_id)}
                    if new_id not in preserve_ids:
                        preserve_ids = preserve_ids + [new_id]

        return {"status": "ok", "clause": bound, "target_id": bound.object_id,
                "preserve_ids": preserve_ids}

    def _match_names(self, items, zone: str) -> List[Tuple[object, str]]:
        """在 zone 中查找 name/alias 命中的条目,返回 (条目, 命中的别名) 列表。"""
        hits = []
        for item in items:
            for alias in [item.name] + list(item.aliases):
                if alias and alias in zone:
                    hits.append((item, alias))
                    break
        return hits

    def _check_object_hits(self, hits, zone: str, kind: str):
        if not hits:
            if kind == "stay":
                return None  # "并停留1秒"省略对象,关联上一个到达事件
            return _error("unknown_object", "场景中不存在对象「%s」" % zone, name=zone)
        return self._check_ambiguous(hits, "object")

    def _check_ambiguous(self, hits, what: str):
        """同一别名命中多个 ID 时报歧义;不同别名命中多个对象由 multi_target 处理。"""
        if len(hits) > 1:
            aliases = {alias for _, alias in hits}
            if len(aliases) == 1:
                alias = aliases.pop()
                return _error("ambiguous_%s" % what,
                              "场景中有 %d 个条目都叫「%s」,请明确指定 %s 的 ID" % (len(hits), alias, what),
                              name=alias, candidates=[item.id for item, _ in hits])
        return None

    # 第 3 步:规范化 -----------------------------------------------------------

    def _normalize(self, bound: List[_BoundClause]) -> dict:
        for bc in bound:
            if bc.clause.time_text is not None:
                value = parse_number(bc.clause.time_text)
                if value is None:
                    return {"status": "error",
                            "payload": _error("invalid_time_expression",
                                              "无法解析时间表达「%s」" % bc.clause.time_text,
                                              text=bc.clause.time_text)}
                bc.time_seconds = value
            if bc.clause.duration_text is not None:
                value = parse_number(bc.clause.duration_text)
                if value is None:
                    return {"status": "error",
                            "payload": _error("invalid_time_expression",
                                              "无法解析时长表达「%s」" % bc.clause.duration_text,
                                              text=bc.clause.duration_text)}
                bc.duration_seconds = value
        return {"status": "ok"}

    # 事件组装 -------------------------------------------------------------------

    def _build_events(self, bound: List[_BoundClause]) -> dict:
        events: List[Event] = []
        warnings: List[dict] = []
        last_arrive: Optional[Event] = None
        counter = 0

        def next_id() -> str:
            nonlocal counter
            counter += 1
            return "event_%02d" % counter

        for bc in bound:
            slot = bc.clause
            if slot.kind == "arrive":
                arrive = Event(event_id=next_id(), type="arrive",
                               region_id=bc.region_id, time_seconds=bc.time_seconds)
                events.append(arrive)
                last_arrive = arrive
                if bc.time_seconds is None:
                    warnings.append(_warning(
                        "arrive_time_unspecified",
                        "指令未指定到达时间,time_seconds 置空,由规划器决定",
                        event_id=arrive.event_id))
                if slot.deadline_hint:
                    warnings.append(_warning(
                        "arrive_deadline_approximated",
                        "「之前/以内」截止语义第一版近似为在该时刻到达", clause=slot.raw))
                if bc.duration_seconds is not None:
                    events.append(Event(event_id=next_id(), type="stay",
                                        region_id=bc.region_id,
                                        after_event_id=arrive.event_id,
                                        duration_seconds=bc.duration_seconds))
            elif slot.kind == "stay":
                if last_arrive is None:
                    return {"status": "error",
                            "payload": _error("stay_without_arrive",
                                              "停留事件缺少可关联的到达事件", clause=slot.raw)}
                if bc.duration_seconds is None:
                    return {"status": "error",
                            "payload": _error("missing_duration",
                                              "停留子句缺少时长,如「停留 1 秒」", clause=slot.raw)}
                events.append(Event(event_id=next_id(), type="stay",
                                    region_id=last_arrive.region_id,
                                    after_event_id=last_arrive.event_id,
                                    duration_seconds=bc.duration_seconds))
        return {"status": "ok", "events": events, "warnings": warnings}

    # 第 4 步:校验 ----------------------------------------------------------------

    def _validate(self, events: List[Event], scene: SceneInfo) -> dict:
        warnings: List[dict] = []
        duration = scene.video.duration_seconds
        arrive_by_id = {e.event_id: e for e in events if e.type == "arrive"}

        for ev in events:
            if ev.type == "arrive":
                if ev.time_seconds is not None and duration is not None:
                    if ev.time_seconds < 0 or ev.time_seconds > duration:
                        return {"status": "error",
                                "payload": _error("time_out_of_range",
                                                  "到达时间 %g 秒超出视频时长 %g 秒" % (ev.time_seconds, duration),
                                                  time_seconds=ev.time_seconds,
                                                  video_duration=duration)}
            elif ev.type == "stay":
                if ev.duration_seconds is not None and ev.duration_seconds <= 0:
                    return {"status": "error",
                            "payload": _error("invalid_duration",
                                              "停留时长必须大于 0 秒",
                                              duration_seconds=ev.duration_seconds)}
                base = arrive_by_id.get(ev.after_event_id)
                if (ev.duration_seconds is not None and base is not None
                        and base.time_seconds is not None and duration is not None):
                    end = base.time_seconds + ev.duration_seconds
                    if end > duration:
                        return {"status": "error",
                                "payload": _error("time_out_of_range",
                                                  "停留窗口结束于 %g 秒,超出视频时长 %g 秒" % (end, duration),
                                                  window_end=end, video_duration=duration)}
                elif base is not None and base.time_seconds is None:
                    warnings.append(_warning(
                        "stay_window_unchecked",
                        "到达时间未指定,停留窗口是否越界由规划器判断", event_id=ev.event_id))
        return {"status": "ok", "warnings": warnings}

    # 输出组装 ---------------------------------------------------------------------

    def _assemble(self, target_id, events: List[Event], preserve_ids: List[str],
                  background_from_instruction: bool, warnings: List[dict]) -> dict:
        provenance = {}
        for oid in preserve_ids:
            provenance["preserve.object_behavior_ids.%s" % oid] = "instruction"
        provenance["preserve.target_identity"] = "task_default"
        provenance["preserve.background"] = (
            "instruction" if background_from_instruction else "task_default")
        provenance["editable"] = "task_default"

        return {
            "status": "ok",
            "target_object_id": target_id,
            "events": [e.to_dict() for e in events],
            "preserve": {
                "object_behavior_ids": preserve_ids,
                "target_identity": self.defaults["preserve"]["target_identity"],
                "background": self.defaults["preserve"]["background"],
            },
            "editable": list(self.defaults["editable"]),
            "warnings": warnings,
            "provenance": provenance,
        }
