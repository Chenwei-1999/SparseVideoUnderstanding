"""HuggingFace in-process vision-language backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PIL import Image

from revise.llava_next_runtime import is_llava_qwen_checkpoint, load_llava_next_runtime
from revise.pnp.utils import apply_processor_chat_template, configure_llava_processor, ensure_writable_hf_cache

REPO_ROOT = Path(__file__).resolve().parents[2]


def _prep_image(img: Image.Image, *, max_edge: int = 512) -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_edge:
        img = img.copy()
        img.thumbnail((max_edge, max_edge), Image.LANCZOS)
    return img


def _build_chat_messages(
    system_prompt: str,
    user_text: str,
    images: list[Image.Image],
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    content: list[dict[str, Any]] = []
    parts = user_text.split("<image>") if images else [user_text]
    if images and (len(parts) - 1) == len(images):
        for i, img in enumerate(images):
            if parts[i]:
                content.append({"type": "text", "text": parts[i]})
            content.append({"type": "image"})
        if parts[-1]:
            content.append({"type": "text", "text": parts[-1]})
    else:
        for _ in images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": user_text})

    conv: list[dict[str, Any]] = []
    if system_prompt:
        conv.append({"role": "system", "content": system_prompt})
    conv.append({"role": "user", "content": content})
    return conv, [_prep_image(img) for img in images]


def _chat_once_hf(
    model: Any,
    processor: Any,
    system_prompt: str,
    user_text: str,
    images: list[Image.Image],
    *,
    device: Any,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    max_len: int,
) -> str:
    if hasattr(model, "chat_once"):
        return str(
            model.chat_once(
                system_prompt=system_prompt,
                user_text=user_text,
                images=[_prep_image(img) for img in images],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        )

    if processor is None:
        raise RuntimeError("hf_processor_missing: processor is required for transformers AutoModel generation")

    import torch

    conv, prepped_images = _build_chat_messages(system_prompt, user_text, images)
    chat = apply_processor_chat_template(processor, conv, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat, images=prepped_images, return_tensors="pt")
    input_len = int(inputs["input_ids"].shape[-1])
    if input_len + max_new_tokens > max_len:
        raise RuntimeError(f"prompt_too_long: input_len={input_len} max_len={max_len} max_new={max_new_tokens}")

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
    if temperature and temperature > 0:
        gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})
    else:
        gen_kwargs.update({"do_sample": False})
    with torch.inference_mode():
        generated = model.generate(**inputs, **gen_kwargs)
    gen_ids = generated[:, input_len:]
    return str(processor.batch_decode(gen_ids, skip_special_tokens=True)[0])


def _load_transformers_components() -> tuple[Any, Any, Any]:
    from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

    return AutoConfig, AutoModelForVision2Seq, AutoProcessor


def load_model_and_processor(
    model_path: str,
    dtype: str,
    device: Any,
    *,
    attn_implementation: str | None = "sdpa",
    fallback_without_attn: bool = False,
    allow_llava_next_runtime: bool = True,
) -> tuple[Any, Any]:
    """Load a local HuggingFace VLM without making HF a dataset concern."""
    ensure_writable_hf_cache(REPO_ROOT / "data" / "revise_assets" / "hf_home")

    import torch

    torch_dtype = torch.bfloat16
    if dtype == "float16":
        torch_dtype = torch.float16
    if dtype == "float32":
        torch_dtype = torch.float32

    if allow_llava_next_runtime and is_llava_qwen_checkpoint(model_path):
        return load_llava_next_runtime(model_path, dtype, device), None

    AutoConfig, AutoModelForVision2Seq, AutoProcessor = _load_transformers_components()
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    configure_llava_processor(processor, model_config)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    try:
        model = AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs)
    except Exception:
        if not (fallback_without_attn and "attn_implementation" in model_kwargs):
            raise
        model_kwargs.pop("attn_implementation", None)
        model = AutoModelForVision2Seq.from_pretrained(model_path, **model_kwargs)
    model.eval()
    model.to(device)
    return model, processor


def _load_model_and_processor(model_path: str, dtype: str, device: Any) -> tuple[Any, Any]:
    return load_model_and_processor(model_path, dtype, device, attn_implementation="sdpa")


class HFInProcessBackend:
    """Backend adapter for a locally loaded HuggingFace model and processor."""

    def __init__(
        self,
        *,
        model: Any,
        processor: Any,
        device: Any,
        max_len: int,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.max_len = int(max_len)

    def chat(
        self,
        *,
        base_url: str,
        model_id: str,
        system_prompt: str,
        user_text: str,
        images: list[Any],
        temperature: float,
        top_p: float,
        max_tokens: int,
        timeout_s: int,
    ) -> str:
        _ = base_url, model_id, timeout_s
        return _chat_once_hf(
            model=self.model,
            processor=self.processor,
            system_prompt=system_prompt,
            user_text=user_text,
            images=images,
            device=self.device,
            max_new_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            max_len=self.max_len,
        )

    def get_model_id(self, base_url: str, model_id: Optional[str] = None) -> str:
        _ = base_url
        return model_id or "hf-in-process"
