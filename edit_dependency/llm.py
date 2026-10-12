"""Qwen3-VL-8B-Instruct 客户端(OpenAI 兼容协议,标准库实现,无第三方依赖)。

生产环境用 vLLM/SGLang 等把模型服务成 OpenAI 兼容端点,例如:
    vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
配置通过环境变量或构造参数覆盖:
    QWEN_VL_BASE_URL   默认 http://127.0.0.1:8000/v1
    QWEN_VL_API_KEY    默认 EMPTY
    QWEN_VL_MODEL      默认 Qwen3-VL-8B-Instruct
    QWEN_VL_TIMEOUT    默认 60(秒)

测试/离线演示用 MockQwenClient 注入脚本化应答,不依赖真实模型。
"""

import json
import os
import re
import urllib.error
import urllib.request
from typing import Callable, Dict, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_MODEL = "Qwen3-VL-8B-Instruct"
# 云端 API(OpenAI 兼容):设置 DASHSCOPE_API_KEY 即自动启用,无需本地下载模型
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL = "qwen3-vl-8b-instruct"


class LLMError(Exception):
    """LLM 调用失败(端点不可达/超时/应答无法解析)。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _extract_json(text: str) -> dict:
    """从模型应答中抽取 JSON 对象(容忍 ```json 围栏与前后说明文字)。"""
    stripped = re.sub(r"```(?:json)?", "", text).strip()
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise LLMError("llm_bad_response", "应答中未找到 JSON 对象:%s" % text[:200])
    try:
        data = json.loads(stripped[start:end + 1])
    except ValueError as exc:
        raise LLMError("llm_bad_response", "JSON 解析失败:%s" % exc)
    if not isinstance(data, dict):
        raise LLMError("llm_bad_response", "应答不是 JSON 对象")
    return data


class QwenVLClient:
    """对话式客户端。analyzer 只依赖 complete_json(task, payload) 这一个接口。"""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, timeout: Optional[float] = None):
        dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
        if not base_url and not os.environ.get("QWEN_VL_BASE_URL") and dashscope_key:
            # 未显式配置本地端点而提供了 DashScope 密钥:直接走云端 API
            base_url = DASHSCOPE_BASE_URL
            if not api_key:
                api_key = dashscope_key
            if not model and not os.environ.get("QWEN_VL_MODEL"):
                model = DASHSCOPE_MODEL
        self.base_url = (base_url or os.environ.get("QWEN_VL_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("QWEN_VL_API_KEY") or "EMPTY"
        self.model = model or os.environ.get("QWEN_VL_MODEL") or DEFAULT_MODEL
        self.timeout = float(timeout or os.environ.get("QWEN_VL_TIMEOUT") or 60)

    def chat(self, system: str, user: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 2048,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer %s" % self.api_key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise LLMError("llm_http_error", "端点返回 %d:%s"
                           % (exc.code, exc.read().decode("utf-8", "ignore")[:200]))
        except (urllib.error.URLError, OSError) as exc:
            raise LLMError("llm_unavailable", "无法连接模型端点 %s:%s" % (self.base_url, exc))
        except ValueError as exc:
            raise LLMError("llm_bad_response", "端点应答不是合法 JSON:%s" % exc)
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("llm_bad_response", "应答缺少 choices[0].message.content:%s" % exc)

    def complete_json(self, task: str, system: str, user: str) -> dict:
        return _extract_json(self.chat(system, user))


class MockQwenClient:
    """离线测试用:handler(task, user_text) 返回 dict(或抛 LLMError 模拟故障)。"""

    def __init__(self, handler: Callable[[str, str], Dict]):
        self.handler = handler
        self.calls = []   # [(task, user_text), ...] 记录调用,便于断言

    def complete_json(self, task: str, system: str, user: str) -> dict:
        self.calls.append((task, user))
        result = self.handler(task, user)
        if not isinstance(result, dict):
            raise LLMError("llm_bad_response", "mock 应答需为 dict")
        return result

    def chat(self, system: str, user: str) -> str:
        return json.dumps(self.complete_json("chat", system, user), ensure_ascii=False)
