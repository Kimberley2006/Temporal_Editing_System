"""测试用例演示:批量运行用户指令,逐条打印结构化 JSON,并可写入文件。

用法:
    python3 examples/demo_cases.py                        # 仅打印
    python3 examples/demo_cases.py --out results.json     # 打印并写入 JSON 文件
    python3 examples/demo_cases.py --scene my.json --out results.json

可在 CASES 列表中增删自己的指令;场景对象表可参照 examples/scene_demo.json 修改。
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from instruction_parser import InstructionParser  # noqa: E402

# 场景:object_01=红球(别名:红色球), object_02=蓝球, region_01=蓝色区域(别名:蓝区), 视频时长 5 秒
CASES = [
    # ---- 三种基本指令 ----
    "红球在第 3 秒到达蓝色区域",                                   # A. 指定到达时间
    "红球进入蓝色区域后停留 1 秒",                                 # B. 指定停留时长(到达时间为空)
    "红球在第 3 秒到达蓝色区域，并停留 1 秒；蓝球保持不变。",        # C. 到达+停留+保持(规格主例)
    # ---- 句式变体 ----
    "红球在第三秒的时候到达蓝区，并停留两秒",                        # 中文数字 + 区域别名
    "红色球在第 2.5 秒到达蓝色区域，其他对象保持不变",                # 对象别名 + 小数 + 其他对象保持
    "红球到达蓝色区域",                                            # 未指定到达时间 -> null + 警告
    # ---- 异常指令 ----
    "红球在第 8 秒到达蓝色区域",                                   # 时间超出视频时长
    "绿球在第 3 秒到达蓝色区域",                                   # 对象不存在
    "红球旋转一圈",                                               # 不支持的操作
    "红球在第 3 秒到达蓝色区域，蓝球在第 4 秒到达蓝色区域",           # 多目标(第一版不支持)
]


def main():
    default_scene = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "scene_demo.json")
    ap = argparse.ArgumentParser(description="指令解析器测试用例演示")
    ap.add_argument("--scene", "-s", default=default_scene, help="场景对象表 JSON 文件")
    ap.add_argument("--out", "-o", default=None, help="把全部结果写入该 JSON 文件")
    args = ap.parse_args()

    with open(args.scene, encoding="utf-8") as f:
        scene = json.load(f)

    parser = InstructionParser()
    results = []
    for i, instruction in enumerate(CASES, 1):
        result = parser.parse(instruction, scene)
        results.append({"case_id": i, "instruction": instruction, "result": result})
        print("=" * 72)
        print("[case %02d] 指令: %s" % (i, instruction))
        print("-" * 72)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print()

    if args.out:
        payload = {"scene_file": os.path.abspath(args.scene),
                   "case_count": len(results), "cases": results}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print("已写入 %s(%d 条用例)" % (args.out, len(results)))


if __name__ == "__main__":
    main()
