"""命令行示例:python -m instruction_parser --instruction "..." --scene examples/scene_demo.json"""

import argparse
import json
import os

from .parser import InstructionParser


def main():
    ap = argparse.ArgumentParser(description="用户指令解析器(基础版):固定句式 + 规则解析")
    ap.add_argument("--instruction", "-i", required=True, help="用户编辑指令(中文固定句式)")
    ap.add_argument("--scene", "-s", default=None,
                    help="场景对象表 JSON 文件路径(缺省用内置示例场景)")
    args = ap.parse_args()

    if args.scene:
        with open(args.scene, encoding="utf-8") as f:
            scene = json.load(f)
    else:
        default_scene = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "examples", "scene_demo.json")
        with open(default_scene, encoding="utf-8") as f:
            scene = json.load(f)

    parser = InstructionParser()
    print(json.dumps(parser.parse(args.instruction, scene),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
