"""编辑依赖器的 LLM 提示词模板。

Qwen3-VL-8B-Instruct 承担两件事:
  1. extract_edits       —— 把自然语言指令拆解为结构化编辑(操作/保持/约束),绑定场景 ID;
     解析器(规则版)只支持到达/停留句式,删除、延迟、变色、减速等丰富操作由这里提取。
  2. judge_propagation   —— 对规则无法判定的候选项,依据「事件发生时的状态」判断影响
     是否传播(传播/停止/不确定)。

两处都要求输出严格 JSON;analyzer 会做结构校验,校验失败带错误反馈重试一次。
"""

EXTRACT_SYSTEM = """你是视频编辑系统的指令拆解器。把用户的中文编辑指令拆解为结构化 JSON,供下游依赖分析使用。

场景对象表(JSON):
{scene_brief}

既有视频事件 timeline(JSON):
{timeline_brief}

视频时长:{duration} 秒。

只输出一个 JSON 对象,不要输出任何其他文字,结构如下:
{{
  "edits": [
    {{
      "action": "delete|recolor|repath|delay|slow|speed_up|arrive|stay 之一",
      "object_id": "场景对象表中的 ID;delay 等作用于既有事件时填事件的目标对象,可为 null",
      "event_id": "当操作作用于 timeline 中的既有事件时填其 event_id,否则 null",
      "region_id": "arrive 操作的目标区域 ID,否则 null",
      "scope": {{"kind": "all"}} 或 {{"kind": "window", "start_seconds": 数值, "end_seconds": 数值}},
      "params": {{"seconds": 数值(delay 时), "color": "颜色名(recolor 时)", "duration_seconds": 数值(stay 时), "time_seconds": 数值(arrive 时)}},
      "quote": "指令中对应这个操作的原文子句"
    }}
  ],
  "preserves": [
    {{"object_id": "保持不变的对象 ID", "aspects": ["appearance","behavior"] 子集, "quote": "原文子句"}}
  ],
  "constraints": [
    {{"kind": "starts_after_end_of", "subject_event_id": "timeline 事件 ID", "ref_event_id": "timeline 事件 ID", "quote": "原文子句"}}
  ]
}}

要求:
- 对象/区域/事件一律使用上面表格中的 ID,不得发明新 ID;指令提到但表里没有的,放入 "unmatched": ["原文片段"]。
- 动作映射(严格按语义方向,不要弄反):「删除/去掉/消失」→ delete;「变成X色/改成X色/染成」
  → recolor 且 params.color 填颜色;「换条路径/改走别的路」→ repath;「延迟X秒/推迟X秒」
  → delay 且 params.seconds 填数值;「慢一点/放慢/减慢/走慢些」→ slow;「快一点/加速/加快」
  → speed_up;「停留X秒」→ stay 且 params.duration_seconds 填数值;「到达/进入某区域」→ arrive。
- slow/speed_up 作用于「做动作的人或物」(object_id),如「让持杯行走的人慢一点」是
  对 object(人) 做 slow,不是对事件做 speed_up;只有指令明确说「让某事件提前/推迟」
  这类时间语义时,delay/slow 才填 event_id。
- 「整段/全部」用 scope.kind="all";「后半段/从第 X 秒起」用 window,坐标秒数。
- 时序约束(如「B 必须在 A 结束后开始」)里的 A、B 按 timeline 的 name 或描述匹配到 event_id,匹配不上就放入 "unmatched"。
- 每条编辑必须带 quote,便于追溯。
- 没有保持要求/约束时对应数组为空。"""

EXTRACT_USER = """用户指令:{instruction}"""


PROPAGATION_SYSTEM = """你是视频编辑系统的影响传播判定器。一条编辑可能影响某些既有事件;规则引擎已筛出候选项,
但能否传播取决于「事件发生时刻的状态」。请逐项判定。

编辑操作:{op_brief}
候选受影响事件(含其时间窗与成立条件):{candidate_brief}
相关状态证据(其他事件给出的对象状态/关系,注意时间窗):{evidence_brief}
用户的保持要求:{preserve_brief}

判定原则:
- 传播的依据是:该事件的成立条件在编辑作用的时间窗内引用了被修改的对象/事件,且没有其他状态证据表明条件届时由别的对象满足。
- 时间上发生在后面不等于受影响;杯子在 8 秒后被人拿着,就不因 8 秒前所在的桌子被删而失去支撑。
- 证据不足(条件不明、时间窗未覆盖)时判 uncertain,不要猜。

只输出一个 JSON 对象:
{{
  "judgments": [
    {{"event_id": "候选事件 ID", "judgment": "propagated|stopped|uncertain", "reason": "一句话依据,引用时间窗/条件/状态"}}
  ]
}}"""

PROPAGATION_USER = """请判定以上候选事件。"""


def scene_brief(scene) -> str:
    lines = []
    for o in scene.objects:
        aliases = "/".join(o.aliases) if o.aliases else ""
        lines.append("- %s: %s%s" % (o.id, o.name, ("(别名:%s)" % aliases) if aliases else ""))
    for r in scene.regions:
        lines.append("- %s: 区域「%s」" % (r.id, r.name))
    return "\n".join(lines) if lines else "(空)"


def timeline_brief(timeline) -> str:
    if not timeline:
        return "(无既有事件记录)"
    lines = []
    for e in timeline:
        win = "[%g, %g]s" % e.window if e.window else "时间窗未知"
        label = e.name or e.text or e.verb
        lines.append("- %s: %s 参与者%s %s" % (e.event_id, label, e.participants, win))
        for c in e.conditions:
            lines.append("    条件 %s %s" % (c.kind, c.refs))
    return "\n".join(lines)
