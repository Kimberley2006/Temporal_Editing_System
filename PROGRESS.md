# 实验进度记录

> 更新时间:2026-10-12(第二轮:时空规划器 Phase 1 + 底座模型下载)
> 项目:Temporal_Editing_System(视频时间/空间编辑系统)

## 当前状态总览

| 事项 | 状态 | 说明 |
|---|---|---|
| 底座模型 Wan2.1-VACE-1.3B | ✅ 已下载 | `/cache/wuhan/Wan2.1-VACE-1.3B`(约 18 GB,VAE+diffusion+T5 编码器齐全) |
| 指令解析器 instruction_parser | ✅ 已有 | 规则版(到达/停留句式),20 项测试 |
| 编辑依赖器 edit_dependency | ✅ 完成 | 依赖传播/冲突/编辑排序,19 项测试;真实 Qwen3-VL 8/8 用例一致 |
| 时空规划器 spatio_temporal_planner | ✅ Phase 1 完成 | 排程/冲突跳过/最小改动修复/静态验证,17 项独立测试;Phase 2(空间路径+VACE 物化)待做 |
| Qwen3-VL-8B-Instruct | ✅ 已下载 | `/cache/wuhan/Qwen3-VL-8B-Instruct`(约 17 GB) |

**三套件回归:解析器 20 + 依赖器 19 + 规划器 17,全部通过(离线)。**

## 一、本次完成的工作

### 1. 编辑依赖器模块(`edit_dependency/`)

**定位**:处理「一个修改会影响另一个事件或修改」的情况 —— 判断影响是否传播、传播到哪里、
是否与用户的其他要求冲突,并把多条拆解后的用户指令转换为**编辑先后顺序**(先编辑哪段、
后编辑哪段,以获得最高效且最少编辑改动的方案)。

**输入接口与指令解析器完全一致**:`analyze(instruction, scene)`。场景对象表可选扩展
`timeline` 字段(检测跟踪模块提供的既有视频事件:参与对象、时间窗、成立条件),
不破坏解析器对同一份 scene 的使用。

**四步流水线**:

1. **提取** —— Qwen3-VL-8B-Instruct 把自然语言指令拆解为结构化编辑
   (delete/recolor/repath/delay/slow/speed_up/arrive/stay),绑定场景 ID;
   校验失败自动带反馈重试一次;
2. **传播** —— 规则做候选查找(同一对象/成立条件引用/时序约束/空间交互)与
   时间门控;时间窗未知或语义不明的候选交 LLM 按「事件发生时的状态」判定;
3. **冲突** —— 保持要求违例 / 作用范围重叠 / 相反意图;冲突只标记不掩盖;
4. **排序** —— 依赖边拓扑分批(Kahn 分层),`edit_passes` 即编辑遍数。

**给规划器的输出**:`direct_edits`(直接编辑)/ `edit_order`(编辑先后顺序)/
`affected_events`(受影响事件,含 propagated/uncertain/stopped 判定与证据)/
`repair_requirements`(修复要求,方案由规划器决定)/ `conflicts` / `uncertainties`。
`events` 字段与解析器输出同构,arrive/stay 按解析器事件结构输出。

### 2. 代表性验证数据集与测试结果

数据集:`examples/scene_dependency_demo.json`(6 对象 + 11 个 timeline 事件)+
`examples/dependency_cases.py`(8 用例 + 脚本化 LLM 应答)。
覆盖规格表全部七种情况 + 「删除桌子/桌上杯子」状态门控样例。

离线测试(`tests/test_dependency.py`,19 项断言全部通过,解析器 20 项无回归):

| # | 指令 | 验证要点 | 结果 |
|---|---|---|---|
| 1 | 整段删除杯子。 | 关联到杯子全部出现区间 t_01/t_02/t_03/t_04/t_11 | ✅ |
| 2 | 删除杯子。 | 拿取、喝水动作需修复;替代动作由规划器决定 | ✅ |
| 3 | A 延迟 1 秒,B 必须在 A 结束后开始。 | 仅受约束的 B(t_10)受影响,其余事件不延迟 | ✅ |
| 4 | 让红球换条路径,蓝球保持不变。 | 碰撞事件 uncertain(新路径规划器决定);蓝球保持要求保留 | ✅ |
| 5 | 整段删除杯子,同时把后半段杯子改成蓝色。 | scope_overlap 冲突,resolvable_by_ordering=False | ✅ |
| 6 | 删除杯子,同时让持杯行走的人慢一点。 | 持杯动作修复 + 减速要求保留;排序 [删除]→[减速] | ✅ |
| 7 | 红球停留 1 秒,远处汽车变成蓝色。 | 零受影响事件;两操作同批并行,一遍完成 | ✅ |
| 8 | 删除桌子。 | 杯子仅 [0,8]s 失支撑需安置;被拿着之后的动作不受影响(状态门控) | ✅ |

**核心原则验证**:时间上发生在后面 ≠ 受前面修改影响 —— 用例 3(只查约束联系的 B)
与用例 8(杯子 8 秒后被拿着,不因桌子被删继续传播)均按预期通过。

### 3. Qwen3-VL-8B-Instruct 接入

- **下载**:经 hf-mirror(直连 huggingface.co 不通;hub 2.x 默认 xet CDN 在镜像下 401,
  需 `HF_HUB_DISABLE_XET=1` 退回普通 HTTP)下载完成,17 GB 验证完整;
- **客户端**(`edit_dependency/llm.py`,纯标准库,OpenAI 兼容协议)三种接入方式:
  1. 本地 vLLM:`vllm serve /cache/wuhan/Qwen3-VL-8B-Instruct --port 8001`
     (8000 已被其他项目的 qwen3_5_2b vLLM 占用) + `QWEN_VL_BASE_URL`;
  2. 云端 API:设置 `DASHSCOPE_API_KEY` 自动切换到阿里云百炼 `qwen3-vl-8b-instruct`;
  3. 任意 OpenAI 兼容端点:`QWEN_VL_BASE_URL/QWEN_VL_API_KEY/QWEN_VL_MODEL`;
- **离线测试**:MockQwenClient 注入脚本化应答,19 项测试不依赖真实模型。

### 4. 环境事实备忘

- 机器:2 × A100-SXM4-80GB(实验时均空闲);/cache 挂载 12T(余 6.7T);
- 端口 8000:他人 vLLM(qwen3_5_2b,max_model_len=4096,装不下本依赖器的场景提示词,
  实测 60s 超时,不可复用);
- Python 3.10(conda node 环境),项目代码零第三方依赖,vLLM 为新增安装。

## 二、时空规划器 Phase 1(2026-10-12 第二轮)

**已确认的三个设计决策**(用户拍板,2026-10-12):

1. 冲突处理:跳过涉冲突编辑(不吞冲突、不阻塞无冲突部分),冲突项连同建议反馈用户;
2. 修复默认风格:最小改动(动作删除+桥接、位置保持),LLM 替代动作为 Phase 3 可选;
3. 底座模型 Wan2.1-VACE-1.3B 下载(已完成,18 GB)。

**实现**(`spatio_temporal_planner/`,纯标准库):

- `task_package.py` 统一任务包:依赖器输出为主通道(解析器输出为简单指令备选路径);
  冲突跳过策略在此落地(涉冲突 op 全部 skipped、空批次剔除、EscalatedConflict 反馈);
- `temporal.py` 时间规划器:delay 平移 / slow·speed_up 拉伸(默认 1.25/0.8)/ arrive 落点
  (未指定时间排同对象最后事件之后)/ stay 锚定;after_end_of 定点传播;
  越界与负起点记 violations —— "运动能否实现由规划器判断";
- `repair.py` 最小改动模板:remove_action_bridge / hold_position / rescheduled /
  covered_by_delete / recheck_in_pass;
- `verifier.py` 六项静态检查:窗口范围/约束成立/保持对象未被触及/操作覆盖/修复全落地/
  不确定项全处理;
- `planner.py` 主编排 → ExecutionPlan(批次照单执行依赖器 edit_order,不重排);
  repath 挂起为 spatial_pending + pending_uncertainties(Phase 2 空间求解)。

**独立测试**(`tests/test_planner.py`,17 项全部通过,不依赖上游运行时):

| 组 | 用例 | 验证点 |
|---|---|---|
| A 纯独立 | 解析器路径 arrive/stay | 关键帧 3s/4s |
| A | delay 只影响约束关联事件 | t_09→[3,6],t_10→[6,8],t_01 不动 |
| A | slow 默认系数 1.25 | t_09 [2,5]→[2,5.75] |
| A | 冲突跳过(含无关项照常执行) | 涉冲突 2 项跳过,op_03 独立执行 |
| A | preserve_violation 跳过 | 保持对象未被触及 |
| A | 时间不可行 | 依赖事件挤出视频末尾 → violation + 验证不通过,计划仍产出 |
| A | 修复模板 | 删除+桥接 / 位置保持 / 并入删除 |
| A | repath 挂起 | spatial_pending + 不确定项登记 |
| A | 上游非 ok / 空输入 | 拒绝接管 |
| B 链路 | 代表数据集 6 用例 | 删除全区间物化 4 段、冲突全跳过、2 遍保序、独立同批、杯子 [0,8] 位置保持 |

CLI 离线全链路冒烟通过:`python3 -m spatio_temporal_planner -i "..." -s ... --mock`。

## 三、端到端真实模型实验(2026-10-12 第一轮,已完成)

**结论:8/8 用例与离线 Mock 关键结构一致** —— 真实 Qwen3-VL-8B-Instruct 能把拆解的
多条用户指令正确转换为编辑先后顺序(direct_edits / affected / conflicts / order / passes)。

| # | 指令 | Live vs Mock | 验证要点 |
|---|---|---|---|
| 1 | 整段删除杯子。 | ✓ | 关联杯子全部出现区间 |
| 2 | 删除杯子。 | ✓ | 拿取/喝水修复,规划器决定替代 |
| 3 | A 延迟 1 秒,B 必须在 A 结束后开始。 | ✓ | 仅 B(t_10)受影响 |
| 4 | 让红球换条路径,蓝球保持不变。 | ✓ | 碰撞 uncertain,保持要求保留 |
| 5 | 整段删除杯子 + 后半段变蓝色。 | ✓ | scope_overlap 冲突标记 |
| 6 | 删除杯子 + 人慢一点。 | ✓(修复后) | 持杯修复 + 减速保留,2 遍排序 |
| 7 | 红球停留 1 秒 + 汽车变蓝。 | ✓ | 独立,同批并行 1 遍 |
| 8 | 删除桌子。 | ✓ | 状态门控:仅 [0,8]s 杯子失支撑 |

**实验过程与问题修复**(完整闭环):

1. GPU:两卡均被他机任务占用(GPU0 余 10GB,GPU1 余 28.5GB),以
   `--gpu-memory-utilization 0.30 --enforce-eager` 在 GPU1 剩余显存内部署成功;
2. vLLM 启动报 `CXXABI_1.3.15 not found`:系统 libstdc++ 过旧,
   `LD_PRELOAD=/home/ma-user/conda-env/node/lib/libstdc++.so.6` 解决;
3. 首轮对照 **7/8**:用例 6 真实模型把「慢一点」提取为 `speed_up + 事件目标`
   (方向与粒度均错)——在 prompts.py 补充动词映射规则
   (慢一点→slow/作用于对象、快一点→speed_up 等)后重验通过;
4. 用例 3 表面差异实为 Mock 脚本风格问题:真实模型对 delay 正确只填 event_id,
   已把 Mock 应答对齐(delay 不再冗余填 object_id);
5. 终验:离线 19 项测试全过 + Live 对照 8/8 一致。

**服务部署/复现**(实验后已停服务释放显存):

```bash
CUDA_VISIBLE_DEVICES=1 LD_PRELOAD=/home/ma-user/conda-env/node/lib/libstdc++.so.6 \
  vllm serve /cache/wuhan/Qwen3-VL-8B-Instruct --served-model-name Qwen3-VL-8B-Instruct \
  --port 8001 --max-model-len 8192 --gpu-memory-utilization 0.30 --enforce-eager

QWEN_VL_BASE_URL=http://127.0.0.1:8001/v1 python3 examples/live_vs_mock.py
```

## 四、遗留事项与下一步

- **规划器 Phase 2**:空间规划器(路径/避碰求解,消解 repath 类 uncertainty)+
  VACE 物化适配器(ExecutionPlan segments/keyframes → Wan2.1-VACE 的
  视频段/mask/指令三元组)+ 执行器(分段生成/拼接);权重已就位
  (`/cache/wuhan/Wan2.1-VACE-1.3B`),需先跑通 VACE 推理确认其条件控制形式;
- **规划器 Phase 3**:LLM 辅助修复生成(替代动作提议 + 可行性校验)、生成后闭环校验;
- 检测跟踪模块未接入:timeline 目前为人工构建的第一版;
- 提示词迭代:首轮真实模型实验已暴露并修复一类动词映射错误;后续按
  `live_vs_mock.py` 对照方式继续迭代;
- 解析器 README 中「替换为 LLM 解析」的路线:依赖器的 LLM 提取已具备该能力,后续可统一;
- vLLM 为 pip 新装(0.31.0);推理服务实验后均停止,GPU 显存已释放。
