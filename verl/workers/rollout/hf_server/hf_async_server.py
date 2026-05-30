# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from uuid import uuid4

import ray
from ray.actor import ActorHandle

from verl.single_controller.ray import RayClassWithInitArgs
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutMode, RolloutReplica, TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


@ray.remote(num_cpus=1)
class HFTransformersServer:
    """A lightweight Ray actor that forwards token-in/token-out generation to
    the hybrid ActorRolloutRef worker group.

    This avoids vLLM/SGLang model support constraints by using the actor model
    itself (Transformers) for generation.
    """

    def __init__(
        self,
        config: RolloutConfig,
        model_config: HFModelConfig,
        rollout_mode: RolloutMode,
        workers: list[ActorHandle],
        replica_rank: int,
        node_rank: int,
        nnodes: int,
    ):
        self.config: RolloutConfig = omega_conf_to_dataclass(config)
        self.model_config: HFModelConfig = omega_conf_to_dataclass(model_config, dataclass_type=HFModelConfig)
        self.rollout_mode = rollout_mode
        self.workers = workers
        self.replica_rank = int(replica_rank)
        self.node_rank = int(node_rank)
        self.nnodes = int(nnodes)

        # OpenAI-style address is only used for logging/prometheus; keep a stable identifier.
        self._server_address = f"hf://replica{self.replica_rank}-node{self.node_rank}"

    async def wake_up(self):
        if self.rollout_mode in (RolloutMode.HYBRID, RolloutMode.COLOCATED):
            await asyncio.gather(*[w.wake_up.remote() for w in self.workers])
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("HFTransformersServer: skip wake_up in standalone mode")

    async def sleep(self):
        if self.rollout_mode in (RolloutMode.HYBRID, RolloutMode.COLOCATED):
            await asyncio.gather(*[w.sleep.remote() for w in self.workers])
        elif self.rollout_mode == RolloutMode.STANDALONE:
            logger.info("HFTransformersServer: skip sleep in standalone mode")

    async def clear_kv_cache(self):
        # We don't maintain a persistent KV cache across requests here.
        return

    async def generate(
        self,
        *,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        refs = [
            w.hf_generate.remote(
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id or uuid4().hex,
                image_data=image_data,
                video_data=video_data,
            )
            for w in self.workers
        ]
        results = await asyncio.gather(*refs)
        for r in results:
            if r is not None:
                return r
        return TokenOutput(token_ids=[], stop_reason="aborted")


class HFReplica(RolloutReplica):
    """RolloutReplica implementation for the HFTransformersServer."""

    def __init__(
        self,
        replica_rank: int,
        config: RolloutConfig,
        model_config: HFModelConfig,
        gpus_per_node: int = 8,
        is_reward_model: bool = False,
    ) -> None:
        super().__init__(replica_rank, config, model_config, gpus_per_node, is_reward_model)
        self.server_class = HFTransformersServer

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        # HF backend currently only supports HYBRID mode (reuse the actor worker group).
        raise NotImplementedError("HF rollout replica supports hybrid mode only.")

    async def launch_servers(self):
        assert len(self.workers) == self.world_size, (
            f"worker number {len(self.workers)} not equal to world size {self.world_size}"
        )

        worker_node_ids = await asyncio.gather(
            *[
                worker.__ray_call__.remote(lambda self: ray.get_runtime_context().get_node_id())
                for worker in self.workers
            ]
        )
        node_id = worker_node_ids[0]
        name = (
            f"hf_server_{self.replica_rank}"
            if not self.is_reward_model
            else f"hf_server_reward_{self.replica_rank}"
        )
        name = name + f"_{uuid4().hex[:8]}"

        server = self.server_class.options(
            scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                node_id=node_id,
                soft=False,
            ),
            name=name,
        ).remote(
            config=self.config,
            model_config=self.model_config,
            rollout_mode=self.rollout_mode,
            workers=self.workers,
            replica_rank=self.replica_rank,
            node_rank=0,
            nnodes=1,
        )
        self.servers = [server]
        self._server_handle = server
        self._server_address = await server.__ray_call__.remote(lambda self: self._server_address)

