"""数字解析:把"第 3 秒""三秒""第3.5秒"里的数字统一为秒数(float)。"""

import re

CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNIT = {"十": 10, "百": 100, "千": 1000}

# 数字 token:阿拉伯数字(含小数)或中文数字(含小数"点"),供时间/时长正则复用
NUM_PATTERN = r"\d+(?:\.\d+)?|[零一二两三四五六七八九十百千]+(?:点[零一二三四五六七八九]+)?"
NUM_RE = re.compile(r"^%s$" % NUM_PATTERN)


def _cn_integer_to_int(text: str) -> int:
    """中文整数转 int,如 二十五 -> 25、十五 -> 15。支持到千位,超出口径返回 0。"""
    total, section = 0, 0
    for ch in text:
        if ch in CN_DIGIT:
            section = section * 10 + CN_DIGIT[ch] if ch in CN_DIGIT and section and ch != "零" else CN_DIGIT[ch]
        elif ch in CN_UNIT:
            unit = CN_UNIT[ch]
            if section == 0:
                section = 1  # "十"开头,如 十五
            total += section * unit
            section = 0
    return total + section


def parse_number(text: str):
    """把数字字符串统一为 float;无法解析时返回 None。"""
    if text is None:
        return None
    text = text.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    if not NUM_RE.match(text):
        return None
    if "点" in text:
        int_part, _, frac_part = text.partition("点")
        value = _cn_integer_to_int(int_part)
        frac = 0.0
        for i, ch in enumerate(frac_part):
            if ch not in CN_DIGIT:
                return None
            frac += CN_DIGIT[ch] * (10 ** -(i + 1))
        return value + frac
    return float(_cn_integer_to_int(text))
