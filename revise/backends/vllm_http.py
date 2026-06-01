"""OpenAI-compatible vLLM HTTP backend."""

from __future__ import annotations

from typing import Any, Callable, Optional

from revise.pnp.utils import chat_once, get_model_id


class VllmHttpBackend:
    """Small backend adapter around an OpenAI-compatible vLLM endpoint."""

    def __init__(
        self,
        *,
        restart_server: Optional[Any] = None,
        chat_fn: Optional[Callable[..., str]] = None,
        max_edge: int = 384,
        quality: int = 85,
    ) -> None:
        self.restart_server = restart_server
        self.chat_fn = chat_fn
        self.max_edge = int(max_edge)
        self.quality = int(quality)

    def chat(self, **kwargs: Any) -> str:
        try:
            return self._chat_once(**kwargs)
        except Exception:
            if self.restart_server is None:
                raise
            self.restart_server()
            return self._chat_once(**kwargs)

    def _chat_once(self, **kwargs: Any) -> str:
        if self.chat_fn is not None:
            return self.chat_fn(**kwargs)
        return chat_once(
            kwargs["base_url"],
            kwargs["model_id"],
            kwargs["system_prompt"],
            kwargs["user_text"],
            list(kwargs.get("images") or []),
            float(kwargs["temperature"]),
            float(kwargs["top_p"]),
            int(kwargs["max_tokens"]),
            int(kwargs["timeout_s"]),
            max_edge=self.max_edge,
            quality=self.quality,
        )

    def get_model_id(self, base_url: str, model_id: Optional[str] = None) -> str:
        return get_model_id(base_url, model_id=model_id)
