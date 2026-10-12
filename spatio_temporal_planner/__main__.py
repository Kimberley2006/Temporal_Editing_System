"""命令行:把"指令 → 依赖分析 → 时空规划"整链跑通并输出执行计划。

离线(--mock,内置代表数据集应答):
    python3 -m spatio_temporal_planner --instruction "删除杯子，同时让持杯行走的人慢一点。" \
      --scene examples/scene_dependency_demo.json --mock
在线(接真实 Qwen3-VL):去掉 --mock,配置 QWEN_VL_BASE_URL 或 DASHSCOPE_API_KEY。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edit_dependency import EditDependencyAnalyzer, QwenVLClient  # noqa: E402
from spatio_temporal_planner import SpatioTemporalPlanner  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="时空规划器:统一任务包 → 执行计划")
    ap.add_argument("--instruction", "-i", required=True)
    ap.add_argument("--scene", "-s", default="examples/scene_dependency_demo.json")
    ap.add_argument("--mock", action="store_true", help="离线模式(Mock LLM 应答)")
    args = ap.parse_args()

    with open(args.scene, encoding="utf-8") as f:
        scene = json.load(f)

    client = None
    if args.mock:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "examples"))
        from dependency_cases import make_mock_client
        client = make_mock_client()
    else:
        client = QwenVLClient(timeout=120)

    dep = EditDependencyAnalyzer(client=client).analyze(args.instruction, scene)
    if dep.get("status") != "ok":
        print(json.dumps(dep, ensure_ascii=False, indent=2))
        sys.exit(1)
    plan = SpatioTemporalPlanner(scene=scene).plan(dependency_output=dep, scene=scene)
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    sys.exit(0 if plan.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
