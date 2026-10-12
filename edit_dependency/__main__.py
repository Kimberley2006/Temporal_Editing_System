"""命令行示例:
    python -m edit_dependency --instruction "整段删除杯子" --scene examples/scene_dependency_demo.json

模型端点:默认本地 vLLM(http://127.0.0.1:8000/v1);设置 DASHSCOPE_API_KEY 走云端 API;
也可用 --base-url/--api-key/--model 显式指定。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from edit_dependency import EditDependencyAnalyzer, QwenVLClient  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="编辑依赖器:多修改指令的依赖传播/冲突/编辑顺序分析")
    ap.add_argument("--instruction", "-i", required=True, help="用户编辑指令(可含多条修改)")
    ap.add_argument("--scene", "-s", default=None,
                    help="场景对象表 JSON 路径(可含 timeline;缺省用内置依赖演示场景)")
    ap.add_argument("--base-url", default=None, help="Qwen3-VL OpenAI 兼容端点")
    ap.add_argument("--api-key", default=None, help="端点密钥")
    ap.add_argument("--model", default=None, help="模型名(默认 Qwen3-VL-8B-Instruct)")
    ap.add_argument("--timeout", type=float, default=None, help="单次调用超时秒数")
    args = ap.parse_args()

    if args.scene:
        with open(args.scene, encoding="utf-8") as f:
            scene = json.load(f)
    else:
        default_scene = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "examples", "scene_dependency_demo.json")
        with open(default_scene, encoding="utf-8") as f:
            scene = json.load(f)

    client = QwenVLClient(base_url=args.base_url, api_key=args.api_key,
                          model=args.model, timeout=args.timeout)
    analyzer = EditDependencyAnalyzer(client=client)
    result = analyzer.analyze(args.instruction, scene)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result.get("status") == "ok" else 1)


if __name__ == "__main__":
    main()
