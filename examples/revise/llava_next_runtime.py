from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
from PIL import Image


def is_llava_qwen_checkpoint(model_path: str | os.PathLike[str]) -> bool:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return False
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    arch = config.get("architectures") or []
    if isinstance(arch, str):
        arch = [arch]
    return "LlavaQwenForCausalLM" in arch


def _dtype_from_name(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    return torch.bfloat16


def _patch_transformers_modeling_utils() -> None:
    import transformers.modeling_utils as modeling_utils
    from transformers import pytorch_utils

    for name in (
        "apply_chunking_to_forward",
        "find_pruneable_heads_and_indices",
        "prune_linear_layer",
    ):
        if not hasattr(modeling_utils, name) and hasattr(pytorch_utils, name):
            setattr(modeling_utils, name, getattr(pytorch_utils, name))


def _add_llava_next_path(llava_next_path: str | None = None) -> None:
    default_path = Path(__file__).resolve().parents[2] / "data" / "revise_assets" / "third_party" / "LLaVA-NeXT"
    raw_path = llava_next_path or os.getenv("REVISE_LLAVA_NEXT_PATH", "").strip()
    if not raw_path and default_path.exists():
        raw_path = str(default_path)
    if not raw_path:
        return
    path = Path(raw_path).expanduser()
    if not (path / "llava" / "model" / "builder.py").exists():
        raise RuntimeError(f"invalid_llava_next_path: {path} does not contain llava/model/builder.py")
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)


class LlavaNextRuntime:
    def __init__(
        self,
        *,
        tokenizer: Any,
        model: Any,
        image_processor: Any,
        max_length: int,
        torch_dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.max_length = int(max_length or 32768)
        self.torch_dtype = torch_dtype
        self.device = device
        self.config = model.config

        from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        self._default_image_token = DEFAULT_IMAGE_TOKEN
        self._image_token_index = IMAGE_TOKEN_INDEX
        self._conv_templates = conv_templates
        self._process_images = process_images
        self._tokenizer_image_token = tokenizer_image_token

    def chat_once(
        self,
        *,
        system_prompt: str,
        user_text: str,
        images: list[Image.Image],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        image_prompt = "\n".join([self._default_image_token] * len(images))
        if image_prompt:
            question = f"{image_prompt}\n{user_text}"
        else:
            question = user_text
        if system_prompt:
            question = f"{system_prompt.strip()}\n\n{question}"

        conv = copy.deepcopy(self._conv_templates["qwen_1_5"])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = self._tokenizer_image_token(
            prompt,
            self.tokenizer,
            self._image_token_index,
            return_tensors="pt",
        ).unsqueeze(0)
        input_len = int(input_ids.shape[-1])
        if input_len + max_new_tokens > self.max_length:
            raise RuntimeError(
                f"prompt_too_long: input_len={input_len} max_len={self.max_length} max_new={max_new_tokens}"
            )
        input_ids = input_ids.to(self.device)

        image_sizes = [img.size for img in images]
        image_tensor = self._process_images(images, self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [x.to(device=self.device, dtype=self.torch_dtype) for x in image_tensor]
        else:
            image_tensor = image_tensor.to(device=self.device, dtype=self.torch_dtype)

        gen_kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}
        if temperature and temperature > 0:
            gen_kwargs.update({"do_sample": True, "temperature": float(temperature), "top_p": float(top_p)})
        else:
            gen_kwargs.update({"do_sample": False})

        with torch.inference_mode():
            generated = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                **gen_kwargs,
            )
        gen_ids = generated[:, input_len:]
        return str(self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0])


def load_llava_next_runtime(
    model_path: str,
    dtype: str,
    device: torch.device | str,
    *,
    llava_next_path: str | None = None,
) -> LlavaNextRuntime:
    _add_llava_next_path(llava_next_path)
    _patch_transformers_modeling_utils()
    try:
        from llava.model.builder import load_pretrained_model
    except Exception as exc:
        raise RuntimeError(
            "llava_next_unavailable: install or clone LLaVA-NeXT and set REVISE_LLAVA_NEXT_PATH. "
            "Official model card uses LLaVA-NeXT load_pretrained_model(..., model_name='llava_qwen')."
        ) from exc

    torch_dtype = _dtype_from_name(dtype)
    torch_device = torch.device(device)
    device_map: str | dict[str, int | str]
    if torch_device.type == "cuda":
        device_map = "auto"
    else:
        device_map = {"": "cpu"}

    tokenizer, model, image_processor, max_length = load_pretrained_model(
        model_path,
        None,
        "llava_qwen",
        device_map=device_map,
        torch_dtype=dtype,
        attn_implementation="sdpa",
    )
    model.eval()
    return LlavaNextRuntime(
        tokenizer=tokenizer,
        model=model,
        image_processor=image_processor,
        max_length=int(max_length or 32768),
        torch_dtype=torch_dtype,
        device=torch_device,
    )
