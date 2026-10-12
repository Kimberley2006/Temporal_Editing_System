# NEXT_STEPS — 待执行工作清单

> 更新:2026-10-12 21:45
> 前置状态:四个模块(解析器/依赖器/规划器/检测跟踪)全部完成并验证;
> 真实视频全链路的分析阶段已跑通,产物在 `examples/real_artifacts.json`。
> 本文档记录接下来要做的事,含已设计好的补丁与可直接执行的命令。

---

## 一、立即任务:真实视频 Phase B(VACE 执行行人删除)

**现状**:分析阶段完成(检测→依赖→规划→物化全通,VACE 任务 = delete_mask 行人
frames 0-12);Phase B 等了 40 分钟没等到 ≥62GB 整卡,守护超时退出(退出码 5)。

### 步骤 1:给执行器加 T5 卸载开关(补丁已设计,应用即可)

文件:`spatio_temporal_planner/executor.py` 的 `load()` 中,把

```python
model_configs = [
    ModelConfig(path=os.path.join(md, "models_t5_umt5-xxl-enc-bf16.pth")),
    ModelConfig(path=os.path.join(md, "diffusion_pytorch_model.safetensors")),
    ModelConfig(path=os.path.join(md, "Wan2.1_VAE.pth")),
]
```

改为(T5 编码器常驻 CPU,省约 9GB 显存,编码一次只慢几秒):

```python
t5_kwargs = {"offload_device": "cpu"} if os.environ.get("VACE_T5_ON_CPU") == "1" else {}
model_configs = [
    ModelConfig(path=os.path.join(md, "models_t5_umt5-xxl-enc-bf16.pth"), **t5_kwargs),
    ModelConfig(path=os.path.join(md, "diffusion_pytorch_model.safetensors")),
    ModelConfig(path=os.path.join(md, "Wan2.1_VAE.pth")),
]
```

### 步骤 2:重挂 Phase B 守护(门槛降为 50GB)

复用 `/tmp/real_watch.sh` 的 wait_gpu 逻辑,只保留 Phase B:

```bash
cd /temp/wuhan/Temporal_Editing_System
for i in $(seq 1 60); do            # 最多 60 分钟
  for gi in 0 1; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gi)
    free=$((81920 - used))
    if [ $free -ge 50000 ]; then
      VACE_T5_ON_CPU=1 CUDA_VISIBLE_DEVICES=$gi \
        LD_PRELOAD=/home/ma-user/conda-env/node/lib/libstdc++.so.6 \
        python3 examples/real_pipeline.py --stage execute
      exit $?
    fi
  done
  sleep 60
done
echo TIMEOUT; exit 2
```

### 步骤 3:结果验证与展示

- `real_pipeline.py` 的 `verify()` 会自动输出逐帧数值:行人检测区内平均差应
  显著(>15),全图平均差应小于区内变化;
- 用 ffmpeg 抽源/编辑视频的并排帧(复用 smoke 对比图的做法)展示;
- 若显存仍不够(50GB 也等不到):备选是 VAE 也 offload(diffsynth ModelConfig
  同参数),或把 clip 降到 480×288×21 帧(质量换可行)。

### 步骤 4:回填文档

- `PROGRESS.md` 新增"五、真实视频端到端实验"小节:数据来源(MultiCamVideo
  scene473/cam01,流式提取方法)、Phase A 结果、Phase B 结果、对比帧;
- 更新记忆文件(实验阶段推进后照例更新)。

---

## 二、系统级后续(按优先级)

| 优先级 | 事项 | 说明 |
|---|---|---|
| 高 | 语义级事件推导 | 当前 timeline 只有几何事件(移动/接触/支撑);拿起/喝水类语义事件用 Qwen3-VL 对关键帧做关系问答补充,输出挂到对应 conditions |
| 高 | git 提交 | 全部代码/文档未提交;建议按模块分提交(instruction_parser 已有首次提交,补 edit_dependency / spatio_temporal_planner / detection_tracking / 文档) |
| 中 | LLM 辅助修复(规划器 Phase 3) | repair.py 预留的 llm_propose 开关:Qwen3-VL 提议替代动作 + 回到时间/空间规则做可行性校验 |
| 中 | 长视频分段执行 | VACE 单次 81 帧上限;>81 帧的编辑需分段生成 + 边界过渡拼接,执行器需支持 |
| 中 | 真实视频批量评测 | 从 MultiCamVideo 再抽 5-10 个场景(不同场景/目标),统计检测 IoU(需人工标注或用数据集 GT 相机轨迹辅助)、删除质量、保持区保真 |
| 低 | DashScope API 通道 | 有 key 时检测/依赖阶段可零 GPU 运行(llm.py 已支持,设 DASHSCOPE_API_KEY 即生效) |
| 低 | 性能优化 | vLLM 已支持 prefix caching;VACE 可试 TeaCache/FP8(diffsynth 支持) |

---

## 三、环境速查(执行上述任务时)

- GPU:共享 2×A100-80GB,常被占;**用前必查** `nvidia-smi`,等卡用守护轮询模式;
- vLLM 启动需 `LD_PRELOAD=/home/ma-user/conda-env/node/lib/libstdc++.so.6`;
- VACE 端口用 8001(8000 被他人占用);推理需 ~58GB(全 GPU)或 ~50GB(T5 offload);
- HF 下载:`HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1`;
- 机器内存紧张:视频处理逐帧读写,勿一次性读入全部帧;
- MultiCamVideo 流式提取:`cat part* | tar -xzOf - <成员路径> > out.mp4`
  (目标文件在流早期即可秒取;tar 会继续扫流,取到后 kill 即可);
- 全套测试:`python3 tests/test_parser.py && python3 tests/test_dependency.py &&
  python3 tests/test_planner.py && python3 tests/test_detection_tracking.py`(69 项)
