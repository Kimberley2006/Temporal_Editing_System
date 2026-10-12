"""编辑依赖器离线演示:代表性数据集 8 个用例,打印拆解结果与编辑先后顺序。

LLM 应答由 examples/dependency_cases.py 的 Mock 客户端脚本化提供(离线可跑);
接入真实模型后,同一批用例可原样用 QwenVLClient 重放:

    python examples/dependency_demo.py --live          # 需要本地 vLLM 或 DASHSCOPE_API_KEY
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dependency_cases import CASES, load_scene, make_mock_client  # noqa: E402
from edit_dependency import EditDependencyAnalyzer, QwenVLClient  # noqa: E402


def run(analyzer, scene, only=None):
    for instruction, desc in CASES:
        if only and only not in instruction:
            continue
        out = analyzer.analyze(instruction, scene)
        print("=" * 78)
        print("指令: %s" % instruction)
        print("场景: %s" % desc)
        if out["status"] != "ok":
            print("  错误: %s" % json.dumps(out["error"], ensure_ascii=False))
            continue

        print("\n[直接编辑]")
        for e in out["direct_edits"]:
            print("  %s %s %s %s %s" % (
                e["op_id"], e["action"], e.get("object_id") or e.get("event_id") or "",
                json.dumps(e["scope"], ensure_ascii=False),
                json.dumps(e.get("params", {}), ensure_ascii=False)))

        print("[编辑先后顺序] 共 %d 遍" % out["edit_passes"])
        for b in out["edit_order"]:
            tag = "可并行" if b["parallel"] else "串行"
            print("  第 %d 遍(%s): %s" % (b["batch"], tag, ", ".join(b["op_ids"])))
            print("    原因: %s" % b["reason"])
            if b["repair_ids"]:
                print("    随后修复: %s" % ", ".join(b["repair_ids"]))

        if out["affected_events"]:
            print("[受影响事件]")
            for ae in out["affected_events"]:
                src = "+".join(ae["source_op_ids"])
                print("  %s <- %s [%s/%s] %s" % (
                    ae["event_id"], src, ae["relation"], ae["judgment"],
                    ae["evidence"][0]["reason"] if ae["evidence"] else ""))
        if out["repair_requirements"]:
            print("[修复要求]")
            for r in out["repair_requirements"]:
                win = ("[%g,%g]s" % tuple(r["window"])) if r.get("window") else ""
                print("  %s %s%s: %s (%s 决定)" % (
                    r["repair_id"], r.get("event_id") or r.get("object_id"), win,
                    r["requirement"], r["decision_owner"]))
        for c in out["conflicts"]:
            print("[冲突] %s: %s (能靠排序解决: %s)" % (
                c["kind"], c["message"], c["resolvable_by_ordering"]))
        for u in out["uncertainties"]:
            print("[不确定] %s: %s" % (u["subject"], u["question"]))
        for w in out["warnings"]:
            print("[警告] %s: %s" % (w["code"], w["message"]))


def main():
    ap = argparse.ArgumentParser(description="编辑依赖器离线/在线演示")
    ap.add_argument("--live", action="store_true",
                    help="接入真实 Qwen3-VL(本地 vLLM 或 DASHSCOPE_API_KEY)")
    ap.add_argument("--case", default=None, help="按指令子串筛选用例")
    args = ap.parse_args()

    scene = load_scene()
    client = QwenVLClient() if args.live else make_mock_client()
    analyzer = EditDependencyAnalyzer(client=client)
    run(analyzer, scene, only=args.case)


if __name__ == "__main__":
    main()
