"""Shared helpers for converting REVISE rollouts into trainable trajectories."""

from __future__ import annotations

from typing import Optional


def build_revise_training_trajectory(
    *,
    initial_prompt_ids: list[int],
    trajectory_prompt_ids: list[int],
    response_mask: list[int],
    response_logprobs: list[float],
    response_length: int,
) -> tuple[list[int], list[int], list[int], Optional[list[float]]]:
    """Return the trainable REVISE trajectory after the initial prompt.

    The response sequence contains every generated assistant turn plus the
    interleaved user/observation turns. ``response_mask`` marks assistant tokens
    with 1 and observation tokens with 0, matching VERL's multi-turn contract.
    """

    if trajectory_prompt_ids[: len(initial_prompt_ids)] != initial_prompt_ids:
        raise ValueError("REVISE trajectory does not start with the initial prompt")

    full_response_ids = trajectory_prompt_ids[len(initial_prompt_ids) :]
    if len(full_response_ids) != len(response_mask):
        raise ValueError(
            "REVISE response_ids/response_mask length mismatch: "
            f"{len(full_response_ids)} != {len(response_mask)}"
        )

    response_logprobs_out: Optional[list[float]]
    if response_logprobs and len(response_logprobs) == len(response_mask):
        response_logprobs_out = response_logprobs[:response_length]
    else:
        response_logprobs_out = None

    return (
        initial_prompt_ids,
        full_response_ids[:response_length],
        response_mask[:response_length],
        response_logprobs_out,
    )
