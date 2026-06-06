"""NVIDIA NIM chat completions client (OpenAI-compatible)."""

from __future__ import annotations

import os
from typing import Any

import httpx

from surfer.env import ENV_FILE, env_status, load_env

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"


class NvidiaLLMError(RuntimeError):
    """Raised when the NVIDIA API request fails."""


class NvidiaLLM:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            load_env()
        self.api_key = (api_key or os.getenv("NVIDIA_API_KEY") or "").strip()
        if not self.api_key:
            hint = env_status()
            raise NvidiaLLMError(
                "NVIDIA_API_KEY is not set. "
                + (hint or f"Add NVIDIA_API_KEY=... to {ENV_FILE} or export it in your shell.")
            )
        self.model = model
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> NvidiaLLM:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
        """Send a chat completion request and return the assistant message content."""
        response = self._client.post(
            f"{NVIDIA_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": 512,
                "response_format": {"type": "json_object"},
            },
        )
        if response.status_code != 200:
            raise NvidiaLLMError(
                f"NVIDIA API error {response.status_code}: {response.text[:200]}"
            )

        data: dict[str, Any] = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise NvidiaLLMError("NVIDIA API returned no choices")

        content = choices[0].get("message", {}).get("content", "")
        if not content:
            raise NvidiaLLMError("NVIDIA API returned empty content")
        return content
