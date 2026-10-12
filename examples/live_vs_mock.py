"""Mock vs 真实 Qwen3-VL-8B-Instruct 对照实验:

对代表性数据集 8 个用例,分别用离线 Mock 与 --live 真实模型跑编辑依赖器,
对比关键字段(拆解出的编辑/受影响事件/冲突/编辑顺序),验证真实模型能否
把拆解的多条用户指令转换为正确的编辑先后顺序。

    QWEN_VL_BASE_URL=http://127.0.0.1:8001/v1 python3 examples/live_vs_mock.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dependency_cases import CASES, load_scene, make_mock_client  # noqa: E402
from edit_dependency import EditDependencyAnalyzer, QwenVLClient  # noqa: E402


def digest(out):
    """把输出压缩成可对比的摘要:拆解/门控/排序结果。"""
    if out["status"] != "ok":
        return {"status": "error", "error": out["error"]["code"]}
    return {
        "status": "ok",
        "edits": [(e["action"], e.get("object_id") or e.get("event_id"),
                   e["scope"]["kind"]) for e in out["direct_edits"]],
        "affected": sorted({ae["event_id"] for ae in out["affected_events"]}),
        "conflicts": sorted({c["kind"] for c in out["conflicts"]}),
        "order": [tuple(b["op_ids"]) for b in out["edit_order"]],
        "passes": out["edit_passes"],
    }


def main():
    scene = load_scene()
    mock = EditDependencyAnalyzer(client=make_mock_client())
    live = EditDependencyAnalyzer(client=QwenVLClient(timeout=120))

    agree = 0
    rows = []
    for instruction, desc in CASES:
        m = digest(mock.analyze(instruction, scene))
        l = digest(live.analyze(instruction, scene))
        same = (m == l)
        agree += same
        rows.append((instruction, desc, m, l, same))

    print("%-38s %-6s %-6s %s" % ("指令", "Mock", "Live", "一致"))
    print("-" * 100)
    for instruction, desc, m, l, same in rows:
        print("%-36s %-6s %-6s %s   # %s" % (
            instruction, m["status"], l["status"], "✓" if same else "✗", desc))
    print("-" * 100)
    print("关键结构一致:%d/%d" % (agree, len(rows)))

    for instruction, desc, m, l, same in rows:
        if not same:
            print("\n== 差异用例:%s" % instruction)
            print("  Mock: %s" % json.dumps(m, ensure_ascii=False))
            print("  Live: %s" % json.dumps(l, ensure_ascii=False))


if __name__ == "__main__":
    main()
