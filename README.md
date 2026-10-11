# Temporal_Editing_System

视频时间/空间编辑系统。当前包含第一个模块:**用户指令解码器(基础版)** —— 读取用户指令与场景对象表,输出结构化编辑任务,供下游规划器消费。

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

## 目录结构

```
instruction_parser/     解析器包
  parser.py             四步流水线主逻辑
  schemas.py            场景对象表 / 事件数据结构
  numbers.py            中文/阿拉伯数字 -> 秒数
  defaults.py           任务默认规则(身份/背景保持、可编辑量)
examples/scene_demo.json  示例场景对象表
tests/test_parser.py      测试(三种句式 + 全部错误码)
```
