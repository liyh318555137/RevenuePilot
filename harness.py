import json
import asyncio
from collections.abc import Callable
from typing import Any


DEFAULT_SYSTEM_PROMPT = "你是一名专业、谨慎的 B2B SaaS 销售分析助手。"

# 所有 JSON 调用的默认输出上限。sales_manager 要输出五个中文列表
# （reasons / opportunities / risks / next_actions / unknowns），2000 会让它
# 卡在边界上：响应在字符串中间被截断，json.loads 报「Unterminated string」，
# 只能靠重试碰运气。评审员的 payload 更大，同样吃紧。
DEFAULT_JSON_MAX_TOKENS = 4000


def build_thread_id(opportunity_id: str, run_id: str) -> str:
    return f"{opportunity_id}-{run_id}"


class AgentHarness:
    """统一管理模型调用、重试和结构化响应解析。"""

    def __init__(
        self,
        client: Any,
        model: str = "deepseek-flash",
        logger: Callable[[str, str], None] | None = None,
    ):
        self.client = client
        self.model = model
        self.logger = logger or (lambda _level, _message: None)

    async def run_text(
        self,
        prompt: str,
        *,
        agent_name: str,
        max_retries: int = 3,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        result = await self._run(
            prompt,
            agent_name=agent_name,
            max_retries=max_retries,
            json_mode=False,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not isinstance(result, str):
            raise TypeError("文本模型调用返回了非文本结果")
        return result

    async def run_json(
        self,
        prompt: str,
        *,
        agent_name: str,
        max_retries: int = 3,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        result = await self._run(
            prompt,
            agent_name=agent_name,
            max_retries=max_retries,
            json_mode=True,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not isinstance(result, dict):
            raise TypeError("JSON 模型调用返回了非对象结果")
        return result

    async def _run(
        self,
        prompt: str,
        *,
        agent_name: str,
        max_retries: int,
        json_mode: bool,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str | dict[str, Any]:
        for attempt in range(1, max_retries + 1):
            try:
                self.logger(
                    "LLM",
                    f"{agent_name} 调用 Attempt {attempt}",
                )

                base_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

                request: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                base_prompt
                                + ("你必须输出合法 JSON。" if json_mode else "")
                            ),
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                }

                if json_mode:
                    request["response_format"] = {"type": "json_object"}
                    request["max_tokens"] = (
                        max_tokens
                        if max_tokens is not None
                        else DEFAULT_JSON_MAX_TOKENS
                    )
                elif max_tokens is not None:
                    request["max_tokens"] = max_tokens

                if temperature is not None:
                    request["temperature"] = temperature

                response = await self.client.chat.completions.create(**request)
                content = response.choices[0].message.content

                if not content:
                    raise ValueError("DeepSeek 返回了空内容")

                if json_mode:
                    return json.loads(content)
                return content
            except Exception as error:
                self.logger("ERROR", f"{agent_name} 调用失败：{error}")

                if attempt == max_retries:
                    raise

                wait_seconds = 2 ** (attempt - 1)
                self.logger("RETRY", f"{wait_seconds} 秒后重试")
                await asyncio.sleep(wait_seconds)

        raise RuntimeError("无法完成模型调用")
