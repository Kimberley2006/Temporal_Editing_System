# Temporal_Editing_System

视频时间/空间编辑系统。当前包含三个模块:

- **用户指令解码器(基础版)** —— 读取用户指令与场景对象表,输出结构化编辑任务,供下游规划器消费。
- **编辑依赖器** —— 一个修改会影响另一个事件或修改时,判断影响是否传播、传播到哪里、是否与其他要求冲突,并给出编辑先后顺序(最少遍数、最少返工)。
- **时空规划器(Phase 1)** —— 把解析器 + 依赖器的结论展开为逐段编辑执行计划:时间排程、关键帧、编辑段、修复方案与验证结论;VACE 物化在 Phase 2。

## 模块:spatio_temporal_planner

**输入主通道是依赖器输出**(其 direct_edits/events 已与解析器同构),简单 arrive/stay 指令也可直接喂解析器输出。已确认的设计决策(2026-10-12):

- **冲突策略:跳过并反馈** —— 涉冲突编辑全部跳过,无冲突部分照常执行,冲突项连同建议单独反馈用户,规划器不吞冲突也不因冲突阻塞;
- **修复默认最小改动** —— 动作删除+前后帧桥接、失支撑对象位置保持、时序重排;LLM 替代动作提议为 Phase 3 可选开关。

### 内部流水线

```
TaskPackage(统一任务包) → 时间排程(区间传播) → 修复落地(最小改动模板)
   → 批次展开(照单执行依赖器 edit_order) → 静态验证 → ExecutionPlan
```

- **时间规划器**(temporal.py):delay 平移 / slow·speed_up 拉伸(默认系数 1.25 / 0.8)/ arrive 落点 / stay 锚定,再对 after_end_of 约束做定点传播;排程越界或约束不可满足记入 violations 并使验证不通过 —— "运动能否实现由规划器判断"的时间部分;
- **修复模板**(repair.py):remove_action_bridge / hold_position / rescheduled / covered_by_delete / recheck_in_pass;
- **验证器**(verifier.py):窗口范围 / 约束成立 / 保持对象未被触及 / 操作覆盖 / 修复全落地 / 不确定项全处理,六项静态检查;
- **输出 ExecutionPlan**:按批次的 scheduled(排程)、segments(delete_mask/recolor_mask/remove_action_bridge/hold_position/spatial_pending)、keyframes、修复方案、skipped_ops 与 escalated_conflicts(含 user_feedback)、violations、verification、provenance。repath 的路径与避碰在 Phase 2 空间规划器,当前显式挂起为 spatial_pending。

### 快速开始

```bash
# 离线全链路:指令 → 依赖分析(Mock LLM) → 执行计划
python3 -m spatio_temporal_planner \
  --instruction "删除杯子，同时让持杯行走的人慢一点。" \
  --scene examples/scene_dependency_demo.json --mock

# 测试(17 项:排程/冲突跳过/时间不可行/修复模板/链路用例)
python3 tests/test_planner.py
```

底座模型 Wan2.1-VACE-1.3B 已下载到 `/cache/wuhan/Wan2.1-VACE-1.3B`(约 18 GB),
Phase 2 物化适配器将消费 ExecutionPlan 的 segments/keyframes。

## 模块:instruction_parser

固定句式 + 规则解析,验证与规划器的接口。之后替换为大模型解析时,输出保持同样的结构。

### 支持的指令(第一版,中文固定句式)

| 支持类型 | 指令示例 | 提取的信息 |
|---|---|---|
| 指定到达时间 | 红球在第 3 秒到达蓝色区域 | 对象、区域、到达时间 |
| 指定停留时长 | 红球进入蓝色区域后停留 1 秒 | 对象、区域、停留时长 |
| 到达 + 停留 | 红球在第 3 秒到达蓝色区域,并停留 1 秒 | 对象、区域、到达时间、停留时长 |

- 三种指令均可附带「蓝球保持不变」「其他对象保持不变」「背景保持不变」。
- 首轮只编辑一个目标对象;时间表达支持「第 3 秒 / 3 秒时 / 第三秒 / 三秒 / 3.5 秒 / 两秒」等写法。
- 解析器只负责引用场景对象表中的 ID;视频检测与跟踪由独立模块提供。

### 快速开始

```bash
# CLI(不传 --scene 时使用 examples/scene_demo.json)
python3 -m instruction_parser \
  --instruction "红球在第 3 秒到达蓝色区域，并停留 1 秒；蓝球保持不变。"
```

```python
from instruction_parser import InstructionParser

scene = {
    "objects": [
        {"id": "object_01", "name": "红球", "aliases": ["红色球"]},
        {"id": "object_02", "name": "蓝球", "aliases": ["蓝色球"]},
    ],
    "regions": [
        {"id": "region_01", "name": "蓝色区域", "aliases": ["蓝区"]},
    ],
    "video": {"duration_seconds": 5.0, "fps": 16, "frame_count": 80},
}

task = InstructionParser().parse("红球在第 3 秒到达蓝色区域，并停留 1 秒；蓝球保持不变。", scene)
```

输出(事件为单位,`stay` 通过 `after_event_id` 关联 `arrive`):

```json
{
  "status": "ok",
  "target_object_id": "object_01",
  "events": [
    {"event_id": "event_01", "type": "arrive", "region_id": "region_01", "time_seconds": 3.0},
    {"event_id": "event_02", "type": "stay",  "region_id": "region_01",
     "after_event_id": "event_01", "duration_seconds": 1.0}
  ],
  "preserve": {
    "object_behavior_ids": ["object_02"],
    "target_identity": true,
    "background": true
  },
  "editable": ["target_path", "target_speed"],
  "warnings": [],
  "provenance": {
    "preserve.object_behavior_ids.object_02": "instruction",
    "preserve.target_identity": "task_default",
    "preserve.background": "task_default",
    "editable": "task_default"
  }
}
```

- `provenance` 记录每条要求来自用户指令(`instruction`)还是任务默认规则(`task_default`)。
- 未指定到达时间时 `time_seconds` 为 `null` 并给出 `arrive_time_unspecified` 警告,不自行补值。
- 规划器读取该结构后计算路径、速度与逐帧位置;**运动能否实现由规划器判断**,解析器只校验表达是否完整、有效。

### 场景对象表(输入)

第一版由人工建立;`localization` 中的框/mask/轨迹由检测跟踪模块提供,解析器仅引用 ID:

```json
{
  "objects": [{"id": "object_01", "name": "红球", "aliases": ["红色球"]}],
  "regions":  [{"id": "region_01", "name": "蓝色区域", "aliases": ["蓝区"]}],
  "video":    {"duration_seconds": 5.0, "fps": 16, "frame_count": 80}
}
```

### 内部处理(四步)

1. **提取**:分句,抽出对象名、区域名、时间、时长与保持要求;
2. **绑定**:将「红球」「蓝色区域」转换为场景表中的 ID;
3. **规范化**:「第 3 秒」「三秒时」统一为秒数,操作统一为 `arrive`/`stay`;
4. **校验**:对象唯一、区域存在、时间不超出视频范围、操作受支持。

### 异常输出

| 错误码 | 场景 |
|---|---|
| `ambiguous_object` / `ambiguous_region` | 同名条目命中多个 ID,等待明确指定 |
| `unknown_object` / `unknown_region` | 名称不在场景对象表中 |
| `time_out_of_range` | 到达时间或停留窗口超出视频时长 |
| `unsupported_operation` | 旋转、变色、缩放、加速等暂不支持的操作 |
| `multi_target_unsupported` | 同时编辑多个目标(第一版限定单目标) |
| `stay_without_arrive` / `missing_duration` | 停留缺少可关联的到达事件/时长 |
| `missing_target` / `missing_region` / `missing_preserve_object` | 子句缺少对象/区域/保持对象 |
| `preserve_conflict` | 同一对象既是编辑目标又被要求保持不变 |
| `unparseable` / `empty_instruction` / `invalid_scene` | 无法匹配句式 / 空指令 / 场景表不合法 |

错误统一为 `{"status": "error", "error": {"code", "message", ...详情}}`。

### 测试

```bash
python3 tests/test_parser.py   # 或 pytest tests/
```

### 已知限制与后续

- 仅中文固定句式;「之前/以内」截止语义近似为在该时刻到达(有 warning 标注);
- 「保持在某区域」等 stay 变体句式暂未覆盖;
- 规划器接口就绪后,可将本解析器替换为 LLM 解析,输出结构不变。

## 模块:edit_dependency

**编辑依赖分析器主要处理「一个修改会影响另一个事件或修改」的情况。**它判断影响是否应该
传播、传播到哪里,以及是否与用户的其他要求冲突;多条修改指令经它分析后转换为编辑先后
顺序(先编辑哪段、后编辑哪段,以获得最高效且最少编辑改动的执行方案)。

输入与指令解析器完全一致:`analyze(instruction, scene)`;场景对象表可选附带 `timeline`
字段(检测跟踪模块提供的既有视频事件:参与对象、时间窗、成立条件),依赖传播的判定依据。

### 处理的七种情况

| 情况 | 指令示例 | 依赖器行为 |
|---|---|---|
| 同一对象跨事件出现 | 「整段删除杯子。」杯子在 A、B、C 中出现 | 删除要求关联到同一杯子的全部出现区间 |
| 事件依赖被修改的对象 | 「删除杯子。」后面有人拿起杯子喝水 | 标记拿取、喝水动作需要修复;替代动作由规划器决定 |
| 时间修改影响后续安排 | 「A 延迟 1 秒,B 必须在 A 结束后开始。」 | 只检查有约束联系的 B,不默认所有后续事件都延迟 |
| 空间修改影响交互 | 「让红球换条路径,蓝球保持不变。」 | 碰撞事件待定(新路径由规划器决定),保留蓝球保持要求 |
| 多条指令相互冲突 | 「整段删除杯子,同时把后半段杯子改成蓝色。」 | 标记冲突,不能仅靠编辑排序解决 |
| 级联影响遇到其他编辑 | 「删除杯子,同时让持杯行走的人慢一点。」 | 标记持杯动作需要修复,同时保留人物减速要求 |
| 修改彼此独立 | 「红球停留 1 秒,远处汽车变成蓝色。」 | 无关联则不触发额外修改,可同批编辑 |

最关键的原则:**时间上发生在后面,不等于一定受到前面的修改影响。**传播需要依据同一对象、
事件成立条件或明确的用户约束。删除桌子后,桌上的杯子可能需要重新安置;如果杯子已经被人
拿着,就不能仅因为它之前在桌上而继续传播「失去支撑」的影响 —— 依赖器根据**事件发生时的
状态**(条件的引用对象与时间窗覆盖)决定传播还是停止。

### 输出(给规划器)

- **direct_edits**:用户明确要求修改什么(对象/事件、动作、范围、参数,带原文引用);
- **edit_order / edit_passes**:编辑先后顺序。拓扑分批,同批互不依赖可一遍完成;
  删除等改变存在前提的操作先做,后续编辑基于其结果一次算对,避免返工;
- **affected_events**:哪些事件因这些修改需要重新检查(判定 propagated/uncertain/stopped,
  附证据与判定来源 rule/llm);
- **repair_requirements**:哪些依赖需要调整,注明原因;修复方案由规划器决定;
- **conflicts**:无法兼容的要求(scope_overlap / preserve_violation / opposite_intent);
- **uncertainties**:关系证据不足或依赖规划器结果的待定项。

`events` 字段与解析器同构(arrive/stay 操作按解析器事件结构输出),接口可直接替换。

### 内部处理(四步)

1. **提取**:Qwen3-VL-8B-Instruct 把指令拆解为结构化编辑(删除/延迟/变色/减速/换路径等
   丰富操作超出规则解析器句式),绑定场景对象表 ID;校验失败带反馈自动重试一次;
2. **传播**:规则做候选查找(同一对象/成立条件/时序约束/空间交互)与时间门控,
   时间窗未知或语义不明的候选交 Qwen3-VL 按「事件发生时的状态」判定;
3. **冲突**:保持要求违例、作用范围重叠、相反意图;冲突只标记,不靠排序掩盖;
4. **排序**:依赖边拓扑分批(Kahn 分层),输出 edit_order。

### 模型接入(Qwen3-VL-8B-Instruct)

权重已下载到 `/cache/wuhan/Qwen3-VL-8B-Instruct`(约 17 GB)。三种方式任选:

```bash
# 方式一:本地 vLLM 服务(端口 8000 被占用时换 8001;共享显卡上可用低显存比例)
# 若报 CXXABI_1.3.15 not found,加 LD_PRELOAD 指向 conda 的 libstdc++
pip install vllm
CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6 \
  vllm serve /cache/wuhan/Qwen3-VL-8B-Instruct --served-model-name Qwen3-VL-8B-Instruct \
  --port 8001 --max-model-len 8192
export QWEN_VL_BASE_URL=http://127.0.0.1:8001/v1

# 方式二:云端 API(无需本地显卡,设置密钥即自动切换)
export DASHSCOPE_API_KEY=sk-xxx        # 阿里云百炼,模型名 qwen3-vl-8b-instruct

# 方式三:任意 OpenAI 兼容端点
export QWEN_VL_BASE_URL=http://your-endpoint/v1
export QWEN_VL_API_KEY=...             # 可选,默认 EMPTY
export QWEN_VL_MODEL=Qwen3-VL-8B-Instruct
```

### 快速开始

```bash
# CLI(离线演示场景 examples/scene_dependency_demo.json,含 timeline)
python3 -m edit_dependency --instruction "删除杯子，同时让持杯行走的人慢一点。" \
  --scene examples/scene_dependency_demo.json

# 代表性数据集演示(8 个用例;默认 Mock 离线跑,--live 接真实模型)
python3 examples/dependency_demo.py
python3 examples/dependency_demo.py --live
```

```python
from edit_dependency import EditDependencyAnalyzer

result = EditDependencyAnalyzer().analyze("删除杯子，同时让持杯行走的人慢一点。", scene)
# result["edit_order"]: [{"batch": 1, "op_ids": ["op_01"], ...},
#                        {"batch": 2, "op_ids": ["op_02"], ...}]
# result["affected_events"] / ["repair_requirements"] / ["conflicts"] / ["uncertainties"]
```

### 异常输出

| 错误码 | 场景 |
|---|---|
| `llm_unavailable` / `llm_http_error` / `llm_timeout` | 模型端点不可达/返回错误 |
| `llm_bad_response` | 模型应答无法解析为 JSON |
| `llm_extraction_invalid` | 提取结果校验两次未通过(ID 不存在、参数缺失、窗口越界等) |
| `invalid_timeline` | timeline 事件不合法(重复 ID、window 需要 start < end) |
| `empty_instruction` / `invalid_scene` / `no_edit_operation` | 与解析器同义的输入错误 |

冲突(conflicts)不是错误:`status` 仍为 `ok`,由规划器/用户裁决后重跑。
传播判定调用失败时降级为 uncertain 并给出 warning,不中断分析。

### 测试

```bash
python3 tests/test_dependency.py   # 8 个代表性用例 + 接口/异常,离线运行(Mock 应答)
```

## 目录结构

```
instruction_parser/     解析器包
  parser.py             四步流水线主逻辑
  schemas.py            场景对象表 / 事件数据结构
  numbers.py            中文/阿拉伯数字 -> 秒数
  defaults.py           任务默认规则(身份/背景保持、可编辑量)
edit_dependency/        编辑依赖器包
  analyzer.py           主流水线(提取→传播→冲突→排序)与输出组装
  propagation.py        候选查找/时间门控/冲突检测/拓扑排序规则
  llm.py                Qwen3-VL 客户端(OpenAI 兼容)+ 离线 Mock
  prompts.py            提取与传播判定提示词
  schemas.py            EditOp/TimelineEvent/分析产物数据结构
spatio_temporal_planner/  时空规划器包(Phase 1)
  task_package.py       统一任务包构建 + 冲突跳过策略
  temporal.py           时间规划器(区间传播排程)
  repair.py             修复落地方案(最小改动模板)
  verifier.py           计划级静态验证(六项检查)
  planner.py            主编排 → ExecutionPlan
  schemas.py            任务包/执行计划数据结构
examples/scene_demo.json            解析器示例场景
examples/scene_dependency_demo.json 依赖器示例场景(含 timeline)
examples/dependency_cases.py        代表性验证数据集(8 用例 + Mock 应答)
examples/dependency_demo.py         依赖器演示(离线/--live)
examples/live_vs_mock.py            真实模型 vs Mock 对照实验
tests/test_parser.py      解析器测试
tests/test_dependency.py 依赖器测试(七种情况 + 状态门控 + 接口/异常)
tests/test_planner.py     规划器测试(排程/冲突跳过/修复模板/链路用例)
PROGRESS.md               实验进度记录
```
