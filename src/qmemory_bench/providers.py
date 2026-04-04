"""LLM provider abstraction for benchmark judge/scoring.

Supports 6 national providers + OpenAI. All use OpenAI-compatible chat/completions.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ── Provider Registry ───────────────────────────────────────────

@dataclass
class ProviderInfo:
    name: str
    base_url: str
    default_model: str
    description: str


PROVIDERS: dict[str, ProviderInfo] = {
    "deepseek": ProviderInfo(
        name="DeepSeek V3.2",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        description="性价比最高，推荐默认",
    ),
    "minimax": ProviderInfo(
        name="MiniMax 2.7",
        base_url="https://api.minimax.chat/v1",
        default_model="MiniMax-Text-01",
        description="长文本强",
    ),
    "zhipu": ProviderInfo(
        name="智谱 GLM",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        default_model="glm-4-flash",
        description="有免费额度，入门友好",
    ),
    "kimi": ProviderInfo(
        name="Kimi 2.5",
        base_url="https://api.moonshot.cn/v1",
        default_model="moonshot-v1-8k",
        description="中文理解强",
    ),
    "qwen": ProviderInfo(
        name="通义千问",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        default_model="qwen-plus",
        description="阿里云生态",
    ),
    "doubao": ProviderInfo(
        name="豆包",
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        default_model="doubao-1.5-pro-32k",
        description="字节豆包",
    ),
    "openai": ProviderInfo(
        name="OpenAI",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        description="通用",
    ),
}


# ── LLM Client ──────────────────────────────────────────────────

class LLMJudge:
    """OpenAI-compatible LLM client for benchmark judging."""

    def __init__(
        self,
        provider: str = "deepseek",
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        timeout: float = 60.0,
    ):
        info = PROVIDERS.get(provider, PROVIDERS["deepseek"])
        self.provider_name = info.name
        self.base_url = (base_url or info.base_url).rstrip("/")
        self.model = model or info.default_model
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout, proxy=None, trust_env=False)

    async def complete(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0,
        json_mode: bool = False,
        max_tokens: int = 1024,
    ) -> str:
        """Call chat/completions endpoint with retry + exponential backoff."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._client.post(
                    f"{self.base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_exc = e
                if attempt < 2:
                    delay = 2 ** attempt  # 1s, 2s
                    logger.warning(
                        f"LLM call failed ({self.provider_name}), attempt {attempt+1}/3, "
                        f"retrying in {delay}s: {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(delay)

        logger.error(f"LLM call failed after 3 attempts ({self.provider_name}): {last_exc}")
        raise last_exc  # type: ignore[misc]

    async def close(self):
        await self._client.aclose()

    def __repr__(self) -> str:
        return f"LLMJudge({self.provider_name}, model={self.model})"


class MultiKeyLLMJudge:
    """Round-robin LLM judge across multiple API keys for higher throughput."""

    def __init__(
        self,
        provider: str = "deepseek",
        api_keys: list[str] | None = None,
        model: str = "",
        base_url: str = "",
        timeout: float = 60.0,
    ):
        if not api_keys:
            raise ValueError("At least one API key required")
        self._judges = [
            LLMJudge(provider=provider, api_key=key, model=model,
                      base_url=base_url, timeout=timeout)
            for key in api_keys
        ]
        self._index = 0
        self.provider_name = self._judges[0].provider_name
        self.model = self._judges[0].model

    async def complete(self, prompt: str, **kwargs) -> str:
        judge = self._judges[self._index % len(self._judges)]
        self._index += 1
        return await judge.complete(prompt, **kwargs)

    async def close(self):
        for j in self._judges:
            await j.close()

    def __repr__(self) -> str:
        return f"MultiKeyLLMJudge({self.provider_name}, keys={len(self._judges)}, model={self.model})"


def get_provider_info(provider: str) -> ProviderInfo:
    """Get provider info by key."""
    return PROVIDERS.get(provider, PROVIDERS["deepseek"])


def list_providers() -> list[dict[str, str]]:
    """List all supported providers for UI dropdowns."""
    return [
        {
            "key": key,
            "name": info.name,
            "default_model": info.default_model,
            "description": info.description,
        }
        for key, info in PROVIDERS.items()
    ]
