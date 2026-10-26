# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_reward_breakdown_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.experimental.reward.cot_rewriter.training_logger import get_logger as get_training_logger
from verl.experimental.reward.cot_rewriter.combined_reward import (
    check_response_format,
    compute_group_wise_kl_reward,
    compute_kl_reward_from_logprobs,
)

WorkerType = type[Worker]

# Training logger for key points (rollout, reward, backward)
training_logger = get_training_logger("trainer")


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
    # TaskRunner GPU resources (used for resource check adjustment)
    # When TaskRunner is allocated GPUs for reward computation, these GPUs are
    # time-shared with training GPUs (reward runs after vLLM offload), so we need
    # to add them back when checking available resources.
    task_runner_num_gpus: float = 0.0

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())

        # Add back TaskRunner GPU resources since they are time-shared with training GPUs
        # (TaskRunner runs reward computation after vLLM memory is released)
        total_available_gpus += self.task_runner_num_gpus

        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.reweight_method,
                config.pf_ppo.weight_pow,
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GPG,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic"
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"} and (
            config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1
            or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1
        ):
            assert config.actor_rollout_ref.model.use_remove_padding, (
                "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
            )

        if self.use_critic and config.critic.strategy in {"fsdp", "fsdp2"}:
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, (
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."
                )

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if self.val_reward_fn is not None and val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        self.val_dataloader = None
        if self.val_reward_fn is not None and self.val_dataset is not None:
            val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
            if val_batch_size is None:
                val_batch_size = len(self.val_dataset)

            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=val_batch_size,
                num_workers=num_workers,
                shuffle=self.config.data.get("validation_shuffle", True),
                drop_last=False,
                collate_fn=collate_fn,
            )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        if self.val_dataloader is not None:
            assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            # Handle both list-like values and scalar values (bool, int, float, str)
            # Check if v is a numpy scalar (0-dimensional array) - treat as scalar
            is_numpy_scalar = hasattr(v, 'ndim') and getattr(v, 'ndim', None) == 0
            if hasattr(v, '__len__') and not is_numpy_scalar and len(v) == n:
                base_data[k] = v
            elif not hasattr(v, '__len__') or is_numpy_scalar:
                # Scalar value (bool, int, float, str, numpy scalar, etc.) - broadcast to all samples
                base_data[k] = v
            elif k == "saved_prompts" and isinstance(v, dict) and len(v) > 0:
                # Special handling for saved_prompts: it's a dict of lists
                # Flatten it into individual fields with "saved_prompts." prefix
                for inner_key, inner_list in v.items():
                    # Check if inner_list is a numpy scalar (0-dimensional array)
                    is_numpy_scalar = hasattr(inner_list, 'ndim') and getattr(inner_list, 'ndim', None) == 0
                    if isinstance(inner_list, (list, tuple)) and len(inner_list) == n:
                        base_data[f"saved_prompts.{inner_key}"] = inner_list
                    elif not hasattr(inner_list, '__len__') or is_numpy_scalar:
                        # Scalar value - broadcast to all samples
                        base_data[f"saved_prompts.{inner_key}"] = inner_list

        def convert_to_json_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            import numpy as np
            if isinstance(obj, np.ndarray):
                # For ndarrays, convert to nested list
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating, np.bool_)):
                # For numpy scalars, convert to Python native types
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_json_serializable(item) for item in obj]
            else:
                return obj

        lines = []
        for i in range(n):
            entry = {}
            for k, v in base_data.items():
                # Check if v is a numpy scalar (0-dimensional array)
                is_numpy_scalar = hasattr(v, 'ndim') and getattr(v, 'ndim', None) == 0
                if hasattr(v, '__getitem__') and not is_numpy_scalar and i < len(v):
                    entry[k] = convert_to_json_serializable(v[i])
                elif not hasattr(v, '__getitem__') or is_numpy_scalar:
                    # Scalar value - add to every entry
                    entry[k] = convert_to_json_serializable(v)
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        import time
        print(f"[VALIDATION] ====== START validation at step {self.global_steps} ======")
        val_start_time = time.time()
        
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        val_dataloader_len = len(self.val_dataloader)
        print(f"[VALIDATION] Total validation batches: {val_dataloader_len}, val_batch_size: {self.config.data.val_batch_size}")
        
        for batch_idx, test_data in enumerate(self.val_dataloader):
            batch_start_time = time.time()
            print(f"[VALIDATION] Processing batch {batch_idx + 1}/{val_dataloader_len}...")
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            print(f"[VALIDATION] Batch {batch_idx + 1}: Starting generate_sequences (pad_size={pad_size})...")
            gen_start = time.time()
            
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            gen_time = time.time() - gen_start
            print(f"[VALIDATION] Batch {batch_idx + 1}: generate_sequences completed in {gen_time:.2f}s")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            print(f"[VALIDATION] Batch {batch_idx + 1}: Starting reward function...")
            reward_start = time.time()
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_time = time.time() - reward_start
            print(f"[VALIDATION] Batch {batch_idx + 1}: reward function completed in {reward_time:.2f}s")
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, val in result["reward_extra_info"].items():
                    # Skip scalar metadata values (e.g., penalty_mode, base_reward_range)
                    # Only collect per-sample list values
                    if isinstance(val, list):
                        reward_extra_infos_dict[key].extend(val)
                        print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")
                    else:
                        # Scalar metadata - skip for validation (not per-sample data)
                        print(f"[VALIDATION] Skipping scalar metadata '{key}': {val}")

            # Extract data fields from non_tensor_batch (original_cot, ground_truth, etc.)
            # These are nested inside the "reward_model" dict per sample
            batch_size = reward_tensor.shape[0]
            reward_model_data = test_batch.non_tensor_batch.get("reward_model", None)
            if reward_model_data is not None:
                data_field_keys = ["original_cot", "ground_truth", "question"]
                for fkey in data_field_keys:
                    vals = []
                    for si in range(batch_size):
                        item = reward_model_data[si] if hasattr(reward_model_data, '__getitem__') else reward_model_data
                        if isinstance(item, dict):
                            vals.append(item.get(fkey, ""))
                        else:
                            vals.append("")
                    reward_extra_infos_dict[fkey].extend(vals)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
            
            batch_time = time.time() - batch_start_time
            print(f"[VALIDATION] Batch {batch_idx + 1}/{val_dataloader_len} completed in {batch_time:.2f}s")

        print(f"[VALIDATION] All batches processed. Total samples: {len(sample_scores)}")

        # Cleanup reward computer after all validation batches to release GPU memory.
        # (Per-batch cleanup is skipped during validation to avoid SGLang port reuse errors.)
        try:
            from verl.experimental.reward.cot_rewriter.combined_reward import cleanup_reward_computer
            cleanup_reward_computer()
            print("[VALIDATION] Reward computer cleanup completed.")
        except Exception as _cleanup_exc:
            print(f"[VALIDATION] Warning: reward computer cleanup failed: {_cleanup_exc}")

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        val_total_time = time.time() - val_start_time
        print(f"[VALIDATION] ====== COMPLETED in {val_total_time:.2f}s ======")
        print(f"[VALIDATION] Metrics: {list(metric_dict.keys())[:5]}...")
        
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                # Save prompts before pop for logging
                prompts_for_logging = [batch.batch["input_ids"][i] for i in range(len(batch.batch["input_ids"]))]
                
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        # === ROLLOUT START ===
                        training_logger.log_gpu_memory(f"[STEP_{self.global_steps}][ROLLOUT_START]")
                        
                        # Log per-sample prompts with progress
                        batch_size = len(prompts_for_logging)
                        total_samples = len(self.train_dataloader) * self.config.actor_rollout_ref.rollout.n
                        current_batch_start = (self.global_steps - 1) * self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        training_logger.log_per_sample_prompts(
                            prompts=prompts_for_logging,
                            prefix=f"[STEP_{self.global_steps}][ROLLOUT]",
                            max_tokens_to_print=200,
                            tokenizer=self.tokenizer,
                            current_idx=current_batch_start,
                            total=total_samples
                        )
                        
                        # Also log summary stats
                        training_logger.log_prompt_stats(
                            prompts=prompts_for_logging,
                            prefix=f"[STEP_{self.global_steps}][ROLLOUT]",
                            sample_count=2,
                            max_tokens_to_print=200,
                            tokenizer=self.tokenizer
                        )
                        
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        
                        # === ROLLOUT END ===
                        training_logger.log_gpu_memory(f"[STEP_{self.global_steps}][ROLLOUT_END]")
                        
                        # Log per-sample responses with progress
                        responses = gen_batch_output.batch["responses"]
                        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                        training_logger.log_per_sample_responses(
                            responses=responses,
                            prefix=f"[STEP_{self.global_steps}][ROLLOUT]",
                            max_tokens_to_print=200,
                            tokenizer=self.tokenizer,
                            current_idx=current_batch_start,
                            total=total_samples,
                            pad_token_id=pad_token_id,
                            max_samples=2
                        )
                        
                        # Also log summary stats (vLLM-style)
                        training_logger.log_sequence_lengths(
                            sequences=responses,
                            pad_token_id=pad_token_id,
                            name="RESPONSE",
                            prefix=f"[STEP_{self.global_steps}][ROLLOUT]"
                        )
                        training_logger.log_tensor_stats(
                            tensor=responses.float(),
                            name="RESPONSE_TOKEN_IDS",
                            prefix=f"[STEP_{self.global_steps}][ROLLOUT]"
                        )
                        training_logger.log_response_samples(
                            responses=[responses[i] for i in range(len(responses))],
                            prefix=f"[STEP_{self.global_steps}][ROLLOUT]",
                            sample_count=2,
                            max_tokens_to_print=200,
                            tokenizer=self.tokenizer
                        )
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    batch.meta_info["global_steps"] = self.global_steps

                    # recompute old_log_probs BEFORE reward (for KL reward computation)
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    # === REWARD COMPUTATION (after old_log_prob) ===
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # === REWARD START ===
                        training_logger.log_gpu_memory(f"[STEP_{self.global_steps}][REWARD_START]")
                        training_logger.log_data_stats(
                            batch, 
                            prefix=f"[STEP_{self.global_steps}][REWARD]",
                            tokenizer=self.tokenizer,
                            field="prompts"
                        )
                        training_logger.log_response_preview(
                            batch.batch["responses"],
                            prefix=f"[STEP_{self.global_steps}][REWARD]",
                            tokenizer=self.tokenizer,
                            num_tokens=10
                        )
                        
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                            # Log reward breakdown statistics to SwanLab
                            reward_breakdown = compute_reward_breakdown_metrics(reward_extra_infos_dict)
                            metrics.update(reward_breakdown)

                        # === REWARD END ===
                        training_logger.log_gpu_memory(f"[STEP_{self.global_steps}][REWARD_END]")

                        # Log reward statistics (vLLM-style)
                        if reward_tensor is not None:
                            training_logger.log_rewards(
                                rewards=reward_tensor,
                                prefix=f"[STEP_{self.global_steps}][REWARD]"
                            )
                            # Log per-sample reward stats
                            training_logger.log_tensor_stats(
                                tensor=reward_tensor.sum(dim=-1),
                                name="SAMPLE_REWARD",
                                prefix=f"[STEP_{self.global_steps}][REWARD]"
                            )

                        training_logger.log_data_stats(
                            batch,
                            prefix=f"[STEP_{self.global_steps}][REWARD_POST]",
                            tokenizer=self.tokenizer,
                            field="prompts"
                        )
                    # === END REWARD COMPUTATION ===

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            # Log reward breakdown statistics to SwanLab
                            reward_breakdown = compute_reward_breakdown_metrics(reward_extra_infos_dict)
                            metrics.update(reward_breakdown)
                        batch.batch["token_level_scores"] = reward_tensor

                        # === Extract student_log_probs tensor BEFORE non_tensor_batch conversion ===
                        # student_log_probs is a torch.Tensor that must go into batch.batch (not non_tensor_batch).
                        # We pop it from reward_extra_infos_dict to prevent the loop below from
                        # converting it to a numpy array in non_tensor_batch.
                        if reward_extra_infos_dict and "student_log_probs" in reward_extra_infos_dict:
                            slp = reward_extra_infos_dict.pop("student_log_probs")
                            if slp is not None and isinstance(slp, torch.Tensor):
                                batch.batch["student_log_probs"] = slp.to(
                                    device=batch.batch["responses"].device
                                )

                        if reward_extra_infos_dict:
                            # Ensure all values are arrays (handle scalars from reward functions)
                            # Scalar values (e.g., avg_length, avg_reward) cause np.array_split to fail
                            # Skip global config values that should not be per-sample (e.g., penalty_mode, base_reward_range)
                            global_config_keys = {"penalty_mode", "base_reward_range", "lambda_logic", "lambda_useful",
                                              "too_short_threshold_low", "too_short_threshold_high",
                                              "too_short_active_threshold", "too_short_target_threshold",
                                              "too_long_threshold_low", "too_long_threshold_high",
                                              "too_long_active_threshold", "global_steps",
                                              "fsf_reward_scale", "fsf_reward_mode", "fsf_temperature", "fsf_top_k", "fsf_top_p",
                                              "use_logic_reward", "use_useful_reward", "alpha_cog",
                                              "use_tiered_reward", "tier_step", "tier_reward",
                                              "reward_mode", "use_reward_delta", "truncation_penalty",
                                              "format_penalty", "too_short_max_penalty", "too_long_max_penalty",
                                              "fsf_parse_failed_count", "fsf_group_skipped",
                                              "correct_judge_parse_failed_count"}  # Scalar counts, not per-sample
                            for k, v in reward_extra_infos_dict.items():
                                # Skip global config values that are the same for all samples
                                if k in global_config_keys:
                                    continue
                                if isinstance(v, (int, float, str, bool)):
                                    # Wrap scalar in a single-element list
                                    batch.non_tensor_batch[k] = np.array([v], dtype=object)
                                elif isinstance(v, list):
                                    batch.non_tensor_batch[k] = np.array(v, dtype=object)
                                else:
                                    batch.non_tensor_batch[k] = np.array(v)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # === KL Reward: Group-wise Adaptive Normalization ===
                        # Check if reward_mode is "kl" from custom_reward_function.reward_kwargs
                        # (not reward_model.reward_kwargs — that path doesn't exist in the config)
                        reward_kwargs = {}
                        if hasattr(self.config, 'custom_reward_function') and self.config.custom_reward_function is not None:
                            reward_kwargs = self.config.custom_reward_function.get("reward_kwargs", {})
                        if not reward_kwargs:
                            reward_kwargs = self.config.reward_model.get("reward_kwargs", {})
                        if reward_kwargs.get("reward_mode") == "kl":
                            # Get configuration
                            alpha_kl = reward_kwargs.get("alpha_kl", 0.01)
                            kl_eps = reward_kwargs.get("kl_eps", 1e-6)
                            group_size = self.config.actor_rollout_ref.rollout.n
                            
                            # Get old_log_probs (Actor) - computed before reward
                            actor_log_probs = batch.batch["old_log_probs"]  # [batch_size, response_length]
                            response_mask = batch.batch["response_mask"]  # [batch_size, response_length]
                            
                            # Get student_log_probs from reward_extra_infos_dict if available
                            # Student computes log probs on CoT tokens with only question as context
                            student_log_probs = None
                            if reward_extra_infos_dict and "student_log_probs" in reward_extra_infos_dict:
                                student_log_probs = reward_extra_infos_dict["student_log_probs"]
                            elif "student_log_probs" in batch.batch:
                                student_log_probs = batch.batch["student_log_probs"]
                            
                            if student_log_probs is not None:
                                # Compute raw KL reward per sample using full response_mask
                                # (No need to exclude XML tags as the new data format has no tags)
                                r_kl_raw = compute_kl_reward_from_logprobs(
                                    actor_log_probs=actor_log_probs,
                                    student_log_probs=student_log_probs,
                                    response_mask=response_mask,
                                    alpha_kl=alpha_kl,
                                )

                                # Reshape to [num_groups, group_size]
                                batch_size = r_kl_raw.shape[0]
                                num_groups = batch_size // group_size
                                if batch_size % group_size != 0:
                                    training_logger.warning(
                                        f"[KL-v2] Batch size {batch_size} is not divisible by group_size {group_size}. "
                                        f"Skipping KL reward normalization."
                                    )
                                else:
                                    r_kl_grouped = r_kl_raw.view(num_groups, group_size)

                                    # Apply group-wise adaptive normalization for r_kl_cot
                                    # tau = max(r_std, eps) + r_mean
                                    r_kl_cot_01 = compute_group_wise_kl_reward(
                                        r_kl_matrix=r_kl_grouped,
                                        eps=kl_eps,
                                    ).view(-1)  # [batch_size]

                                    # === r_kl_answer: D_KL(student(answer|q,rewrite_cot) || student(answer|q,original_cot)) ===
                                    kl_answer_values_list = None
                                    if reward_extra_infos_dict and "kl_answer_values" in reward_extra_infos_dict:
                                        kl_answer_values_list = reward_extra_infos_dict.get("kl_answer_values")
                                    elif "kl_answer_values" in batch.non_tensor_batch:
                                        kl_answer_values_list = batch.non_tensor_batch["kl_answer_values"]

                                    # Get weights from config (default: 3/4 for cot, 1/4 for answer)
                                    w_kl_cot = reward_kwargs.get("w_kl_cot", 0.75)
                                    w_kl_answer = reward_kwargs.get("w_kl_answer", 0.25)

                                    if kl_answer_values_list is not None and len(kl_answer_values_list) == batch_size:
                                        # Convert to tensor and normalize with same group-wise method
                                        r_kl_answer_raw = torch.tensor(
                                            [float(v) for v in kl_answer_values_list],
                                            dtype=r_kl_cot_01.dtype,
                                            device=r_kl_cot_01.device,
                                        )
                                        r_kl_answer_grouped = r_kl_answer_raw.view(num_groups, group_size)
                                        r_kl_answer_01 = compute_group_wise_kl_reward(
                                            r_kl_matrix=r_kl_answer_grouped,
                                            eps=kl_eps,
                                        ).view(-1)  # [batch_size]

                                        # Combined reward: weighted sum
                                        r_cog_01_flat = w_kl_cot * r_kl_cot_01 + w_kl_answer * r_kl_answer_01

                                        # Store raw kl_answer values in non_tensor_batch for rollout/val logging
                                        r_kl_answer_raw_list = r_kl_answer_raw.detach().cpu().tolist()
                                        batch.non_tensor_batch["kl_answer_values"] = np.array(
                                            r_kl_answer_raw_list, dtype=object
                                        )
                                    else:
                                        # Fallback: no answer KL available, use only cot KL
                                        r_cog_01_flat = r_kl_cot_01
                                        r_kl_answer_raw = None
                                        r_kl_answer_01 = None
                                        training_logger.warning(
                                            "[KL-v2] kl_answer_values not found or size mismatch. "
                                            "Using only r_kl_cot for reward."
                                        )

                                    # --- Fix: Preserve penalties and assign to last valid token ---
                                    # 1. Extract old total reward from token_level_scores
                                    #    (only last valid token is non-zero, so sum gives the scalar)
                                    token_level_scores = batch.batch["token_level_scores"]
                                    old_total = token_level_scores.sum(dim=-1)  # [batch_size]

                                    # 2. Extract old R_cog (already includes correct_judge penalty)
                                    old_r_cog_np = batch.non_tensor_batch["cognitive_rewards"]
                                    old_r_cog = torch.tensor(
                                        [float(x) for x in old_r_cog_np],
                                        dtype=r_cog_01_flat.dtype,
                                        device=r_cog_01_flat.device,
                                    )

                                    # 3. penalty_portion = everything except R_cog
                                    #    (format, truncation, length, FSF R_logic, etc.)
                                    penalty_portion = old_total - old_r_cog

                                    # 4. Apply correct_judge penalty to r_cog_01
                                    correct_judge_penalty_value = reward_kwargs.get(
                                        "correct_judge_penalty_value", 0.0
                                    )
                                    if "is_correct_results" in batch.non_tensor_batch:
                                        is_correct_results = batch.non_tensor_batch["is_correct_results"]
                                        for i in range(r_cog_01_flat.shape[0]):
                                            if is_correct_results[i] is False or is_correct_results[i] is None:
                                                r_cog_01_flat[i] = correct_judge_penalty_value

                                    # 5. Rebuild total reward: new R_cog + preserved penalties
                                    new_total = r_cog_01_flat + penalty_portion

                                    # 5.1 Re-apply zero-reward penalties: the penalty system sets
                                    # total_rewards=0 for penalized samples, but the formula
                                    # penalty_portion = old_total(=0) - old_r_cog breaks this
                                    # (it produces penalty_portion = -old_r_cog instead of capturing
                                    # the actual penalty). Fix: force new_total=0 for penalized samples.
                                    if "penalized_flags" in batch.non_tensor_batch:
                                        pf = batch.non_tensor_batch["penalized_flags"]
                                        for i in range(new_total.shape[0]):
                                            if int(pf[i]):
                                                new_total[i] = 0.0

                                    # 6. Assign to last valid token (consistent with non-KL path)
                                    seq_lengths = response_mask.sum(dim=1).long() - 1
                                    seq_lengths = seq_lengths.clamp(min=0, max=response_mask.shape[1] - 1)
                                    new_token_level_rewards = torch.zeros_like(token_level_scores)
                                    for i in range(new_total.shape[0]):
                                        new_token_level_rewards[i, seq_lengths[i]] = new_total[i]
                                    batch.batch["token_level_rewards"] = new_token_level_rewards

                                    # 7. Also update token_level_scores so that
                                    #    critic/score/mean reflects the KL-normalized total
                                    batch.batch["token_level_scores"] = new_token_level_rewards.clone()

                                    # 8. Sync cognitive_rewards so logs/metrics reflect r_cog_01
                                    r_cog_01_list = r_cog_01_flat.detach().cpu().tolist()
                                    batch.non_tensor_batch["cognitive_rewards"] = np.array(
                                        r_cog_01_list, dtype=object
                                    )

                                    # 9. Override stale metrics from compute_reward_breakdown_metrics
                                    #    (it ran before the KL block and recorded old values)
                                    metrics["reward/cognitive/mean"] = float(np.mean(r_cog_01_list))
                                    metrics["reward/cognitive/max"] = float(np.max(r_cog_01_list))
                                    metrics["reward/cognitive/min"] = float(np.min(r_cog_01_list))

                                    new_total_list = new_total.detach().cpu().tolist()
                                    metrics["reward/total/mean"] = float(np.mean(new_total_list))
                                    metrics["reward/total/max"] = float(np.max(new_total_list))
                                    metrics["reward/total/min"] = float(np.min(new_total_list))

                                    # Log KL reward metrics (grouped under reward/ for SwanLab)
                                    metrics["reward/r_kl_cot/min"] = r_kl_raw.min().item()
                                    metrics["reward/r_kl_cot/max"] = r_kl_raw.max().item()
                                    metrics["reward/r_kl_cot/mean"] = r_kl_raw.mean().item()
                                    metrics["reward/r_kl_cot_01/min"] = r_kl_cot_01.min().item()
                                    metrics["reward/r_kl_cot_01/max"] = r_kl_cot_01.max().item()
                                    metrics["reward/r_kl_cot_01/mean"] = r_kl_cot_01.mean().item()
                                    # Keep old metric names for backward compat
                                    metrics["reward/r_kl/min"] = r_kl_raw.min().item()
                                    metrics["reward/r_kl/max"] = r_kl_raw.max().item()
                                    metrics["reward/r_kl/mean"] = r_kl_raw.mean().item()
                                    metrics["reward/r_cog_01/min"] = r_cog_01_flat.min().item()
                                    metrics["reward/r_cog_01/max"] = r_cog_01_flat.max().item()
                                    metrics["reward/r_cog_01/mean"] = r_cog_01_flat.mean().item()

                                    # Log r_kl_answer metrics
                                    if r_kl_answer_raw is not None:
                                        metrics["reward/r_kl_answer/min"] = r_kl_answer_raw.min().item()
                                        metrics["reward/r_kl_answer/max"] = r_kl_answer_raw.max().item()
                                        metrics["reward/r_kl_answer/mean"] = r_kl_answer_raw.mean().item()
                                        metrics["reward/r_kl_answer_01/min"] = r_kl_answer_01.min().item()
                                        metrics["reward/r_kl_answer_01/max"] = r_kl_answer_01.max().item()
                                        metrics["reward/r_kl_answer_01/mean"] = r_kl_answer_01.mean().item()

                                    metrics["reward/penalty_portion/mean"] = penalty_portion.mean().item()
                                    metrics["reward/new_total/mean"] = new_total.mean().item()

                                    training_logger.info(
                                        f"[KL-v2] Applied group-wise KL reward normalization: "
                                        f"r_kl_cot_mean={r_kl_raw.mean().item():.4f} → "
                                        f"r_kl_cot_01_mean={r_kl_cot_01.mean().item():.4f}, "
                                        f"r_kl_answer_mean={r_kl_answer_raw.mean().item():.4f}, "
                                        f"r_cog_01_mean={r_cog_01_flat.mean().item():.4f}, "
                                        f"penalty_portion_mean={penalty_portion.mean().item():.4f}, "
                                        f"new_total_mean={new_total.mean().item():.4f}"
                                    )
                            else:
                                training_logger.warning(
                                    "[KL-v2] student_log_probs not found. "
                                    "Make sure Student Model is configured to return log_probs."
                                )

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # === BACKWARD START ===
                        training_logger.log_gpu_memory(f"[STEP_{self.global_steps}][BACKWARD_START]")
                        
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        
                        # === BACKWARD END ===
                        training_logger.log_gpu_memory(f"[STEP_{self.global_steps}][BACKWARD_END]")
                        
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

                            # Collect all reward-related fields from non_tensor_batch
                            # reward_extra_infos_dict from compute_reward already contains reward breakdown
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            # Also add any other reward fields from non_tensor_batch
                            reward_fields = [
                                "original_cot", "ground_truth",
                                # Reward breakdown fields
                                "cognitive_rewards", "useful_rewards",
                                "rewrite_mean_log_probs", "original_mean_log_probs", "no_cot_mean_log_probs",
                                "rewrite_ppls", "original_ppls",  # PPL values (ppl mode only)
                                "relative_improvement_rates",  # Formula 1: avg_prob(rewrite)/avg_prob(original) - 1 (observation)
                                "logic_rewards", "total_rewards", "original_fsfs", "rewrite_fsfs",
                                "ged", "ged_01", "logic_reward_valid_flags",
                                "reward_deltas",  # rewrite-original log_prob diff
                                "r_cog_before_correct_judge",
                                # Penalty fields
                                "truncation_penalties", "format_penalties", "too_short_penalties", "too_long_penalties",
                                "too_short_penalties_actual", "too_long_penalties_actual",
                                "repeat_penalties", "repeat_penalties_actual", "similarities",
                                "penalized_flags", "penalty_reasons",
                                # Correct Judge fields
                                "is_correct_results", "correct_judge_penalties",
                                # DAPO filter_groups flags
                                "truncated", "format_valid", "length_ratio",
                                # FSF parse status
                                "fsf_parse_failed_count", "fsf_group_skipped", "ged_parse_failed_count",
                                # FSF DOT outputs (raw FSF outputs for debugging/analysis)
                                "original_dot_outputs", "rewrite_dot_outputs",
                                # Saved prompts for debugging (student_model and fsf_judge requests)
                                "saved_prompts",
                                # KL answer divergence values (raw, before normalization)
                                "kl_answer_values",
                            ]
                            for field in reward_fields:
                                if field in batch.non_tensor_batch and field not in reward_extra_infos_dict:
                                    reward_extra_infos_dict[field] = batch.non_tensor_batch[field].tolist()

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        print(f"[TRAINER] Step {self.global_steps}: Starting validation (test_freq={self.config.trainer.test_freq})...")
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        print(f"[TRAINER] Step {self.global_steps}: Validation completed")
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)


class RayDAPOTrainer(RayPPOTrainer):
    """
    DAPO Trainer that extends RayPPOTrainer with filter_groups support.

    This trainer implements the Dynamic Sampling feature from DAPO:
    - Filters groups of samples based on metric variance
    - Continues generating until enough diverse groups are collected

    Key differences from RayPPOTrainer:
    - Supports algorithm.filter_groups.enable configuration
    - Implements multi-batch generation loop with filtering
    - Uses metric-based group filtering (acc, score, seq_reward, etc.)

    Reference: https://arxiv.org/abs/2503.14476
    """

    def fit(self):
        """
        The training loop of DAPO.
        Extends RayPPOTrainer.fit() with filter_groups support.
        """
        from omegaconf import OmegaConf
        from collections import defaultdict

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        timing_raw = defaultdict(float)
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}

                do_profile = (
                    self.global_steps in self.config.trainer.profile_steps
                    if self.config.trainer.profile_steps is not None
                    else False
                )
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(do_profile)

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in new_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                # Save prompts before pop for logging
                prompts_for_logging = [new_batch.batch["input_ids"][i] for i in range(len(new_batch.batch["input_ids"]))]

                gen_batch = new_batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.gen_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # === ROLLOUT ===
                    with marked_timer("gen", timing_raw):
                        training_logger.log_gpu_memory(f"[STEP_{self.global_steps}][ROLLOUT_START]")

                        batch_size = len(prompts_for_logging)
                        total_samples = len(self.train_dataloader) * self.config.actor_rollout_ref.rollout.n
                        current_batch_start = (self.global_steps - 1) * self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n

                        training_logger.log_per_sample_prompts(
                            prompts=prompts_for_logging,
                            prefix=f"[STEP_{self.global_steps}][ROLLOUT]",
                            max_tokens_to_print=200,
                            tokenizer=self.tokenizer,
                            current_idx=current_batch_start,
                            total=total_samples
                        )

                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                        training_logger.log_gpu_memory(f"[STEP_{self.global_steps}][ROLLOUT_END]")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if "response_mask" not in new_batch.batch.keys():
                        new_batch.batch["response_mask"] = compute_response_mask(new_batch)

                    # === PRE-FILTER FOR length_ratio (DAPO optimization) ===
                    # If filter_groups uses length_ratio, compute it first without loading Student Model
                    # This avoids wasting time computing R_cog for samples that will be filtered out
                    if (
                        self.config.algorithm.filter_groups.enable
                        and self.config.algorithm.filter_groups.metric == "length_ratio"
                    ):
                        print("[DAPO] Pre-filtering using length_ratio (without loading Student Model)...")
                        from verl.experimental.reward.cot_rewriter.combined_reward import compute_length_ratio_only

                        # Get low threshold from config
                        length_threshold = 0.4
                        if hasattr(self.config, 'custom_reward_function') and hasattr(self.config.custom_reward_function, 'reward_kwargs'):
                            length_threshold = self.config.custom_reward_function.reward_kwargs.get('too_short_threshold_low', 0.4)

                        # Compute length_ratio only (fast, no Student Model)
                        field_mapping = None
                        if hasattr(self.config, 'custom_reward_function') and hasattr(self.config.custom_reward_function, 'reward_kwargs'):
                            field_mapping = self.config.custom_reward_function.reward_kwargs.get('field_mapping')

                        length_ratio_result = compute_length_ratio_only(
                            data=new_batch,
                            tokenizer=self.tokenizer,
                            field_mapping=field_mapping,
                        )
                        length_ratios = length_ratio_result["length_ratio"]
                        format_valids = length_ratio_result["format_valid"]  # Get format validity flags
                        new_batch.non_tensor_batch["length_ratio"] = np.array(length_ratios)
                        new_batch.non_tensor_batch["format_valid"] = np.array(format_valids)

                        # Pre-filter: keep only groups with at least 1 sample >= length_threshold
                        # AND at least 1 sample with valid format
                        prompt_uid2length_vals = defaultdict(list)
                        prompt_uid2format_vals = defaultdict(list)
                        for uid, length_val, format_val in zip(
                            new_batch.non_tensor_batch["uid"], length_ratios, format_valids, strict=True
                        ):
                            prompt_uid2length_vals[uid].append(length_val)
                            prompt_uid2format_vals[uid].append(format_val)

                        kept_prompt_uids = []
                        filtered_by_length = []
                        filtered_by_format = []
                        for uid in prompt_uid2length_vals.keys():
                            length_vals = prompt_uid2length_vals[uid]
                            format_vals = prompt_uid2format_vals[uid]

                            # Check: at least 1 sample >= length_threshold
                            has_valid_length = sum(1 for v in length_vals if v >= length_threshold) >= 1
                            # Check: at least 1 sample with valid format
                            has_valid_format = sum(1 for v in format_vals if v == 1) >= 1

                            if has_valid_length and has_valid_format:
                                kept_prompt_uids.append(uid)
                            elif not has_valid_length:
                                filtered_by_length.append(uid)
                            elif not has_valid_format:
                                filtered_by_format.append(uid)

                        kept_traj_idxs = [
                            idx for idx, uid in enumerate(new_batch.non_tensor_batch["uid"])
                            if uid in kept_prompt_uids
                        ]

                        num_before = len(new_batch.non_tensor_batch["uid"])
                        new_batch = new_batch[kept_traj_idxs]
                        num_after = len(new_batch.non_tensor_batch["uid"])
                        print(f"[DAPO] Pre-filter: {num_before} -> {num_after} samples ({len(kept_prompt_uids)} groups kept)")
                        print(f"[DAPO]   - Filtered by length_ratio < {length_threshold}: {len(filtered_by_length)} groups")
                        print(f"[DAPO]   - Filtered by all invalid format: {len(filtered_by_format)} groups")

                        # If no samples left, skip to next generation
                        if num_after == 0:
                            print(f"[DAPO] All groups filtered out by pre-filter. Continuing to next generation...")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError("[DAPO] All groups filtered out and max_num_gen_batches reached.")

                    # === REWARD ===
                    with marked_timer("reward", timing_raw):
                        training_logger.log_gpu_memory(f"[STEP_{self.global_steps}][REWARD_START]")

                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_tensor)

                        reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # Log reward breakdown statistics to SwanLab
                        reward_breakdown = compute_reward_breakdown_metrics(reward_extra_infos_dict)
                        metrics.update(reward_breakdown)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                        training_logger.log_gpu_memory(f"[STEP_{self.global_steps}][REWARD_END]")

                    # === FILTER GROUPS (DAPO-specific) ===
                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:
                        metric_name = self.config.algorithm.filter_groups.metric
                        # For length_ratio: apply reward variance filter after reward computation
                        if metric_name == "length_ratio":
                            # Compute seq_final_reward for variance filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).float().numpy()
                            )

                            # Group by uid and compute variance
                            prompt_uid2rewards = defaultdict(list)
                            for uid, reward in zip(
                                new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch["seq_final_reward"], strict=True
                            ):
                                prompt_uid2rewards[uid].append(reward)

                            # Filter out groups with zero variance
                            total_groups = len(prompt_uid2rewards)
                            filtered_by_variance = []
                            kept_prompt_uids = []

                            for uid, rewards in prompt_uid2rewards.items():
                                variance = np.var(rewards)
                                if variance > 0 or len(rewards) == 1:
                                    kept_prompt_uids.append(uid)
                                else:
                                    filtered_by_variance.append(uid)

                            kept_traj_idxs = [
                                idx for idx, uid in enumerate(new_batch.non_tensor_batch["uid"])
                                if uid in kept_prompt_uids
                            ]

                            num_before = len(new_batch.non_tensor_batch["uid"])
                            new_batch = new_batch[kept_traj_idxs]
                            num_after = len(new_batch.non_tensor_batch["uid"])
                            batch = new_batch
                            num_prompt_in_batch = len(kept_prompt_uids)
                            print(f"[DAPO] Post-filter (reward variance): {num_before} -> {num_after} samples ({len(kept_prompt_uids)}/{total_groups} groups kept)")
                            print(f"[DAPO]   - Filtered by reward variance==0: {len(filtered_by_variance)} groups")
                        else:
                            # For other metrics, use the original filter logic
                            if metric_name == "seq_final_reward":
                                new_batch.non_tensor_batch["seq_final_reward"] = (
                                    new_batch.batch["token_level_rewards"].sum(dim=-1).float().numpy()
                                )
                            elif metric_name == "seq_reward":
                                new_batch.non_tensor_batch["seq_reward"] = (
                                    new_batch.batch["token_level_scores"].sum(dim=-1).float().numpy()
                                )

                            # Collect the sequence reward for each trajectory
                            prompt_uid2metric_vals = defaultdict(list)
                            for uid, metric_val in zip(
                                new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                            ):
                                prompt_uid2metric_vals[uid].append(metric_val)

                            # Special handling for 'truncated' metric: keep groups with at least 2 non-truncated samples
                            if metric_name == "truncated":
                                kept_prompt_uids = [
                                    uid
                                    for uid, metric_vals in prompt_uid2metric_vals.items()
                                    if sum(1 for v in metric_vals if v == 0) >= 2  # at least 2 non-truncated samples
                                ]
                            # Special handling for 'format_valid' metric: keep groups with at least 1 valid format sample
                            elif metric_name == "format_valid":
                                kept_prompt_uids = [
                                    uid
                                    for uid, metric_vals in prompt_uid2metric_vals.items()
                                    if sum(1 for v in metric_vals if v == 1) >= 1  # at least 1 valid format sample
                                ]
                            # Special handling for 'length_ratio' metric (not expected here since we pre-filter, but kept for safety)
                            elif metric_name == "length_ratio":
                                # Get low threshold from config (default to 0.4)
                                length_threshold = 0.4
                                if hasattr(self.config, 'custom_reward_function') and hasattr(self.config.custom_reward_function, 'reward_kwargs'):
                                    length_threshold = self.config.custom_reward_function.reward_kwargs.get('too_short_threshold_low', 0.4)

                                kept_prompt_uids = [
                                    uid
                                    for uid, metric_vals in prompt_uid2metric_vals.items()
                                    if sum(1 for v in metric_vals if v >= length_threshold) >= 1
                                ]
                            else:
                                # For other metrics (seq_reward, etc.), use variance-based filtering
                                prompt_uid2metric_std = {}
                                for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                                    prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

                                kept_prompt_uids = [
                                    uid
                                    for uid, std in prompt_uid2metric_std.items()
                                    if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                                ]
                            num_prompt_in_batch += len(kept_prompt_uids)

                            kept_traj_idxs = []
                            for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                                if traj_from_prompt_uid in kept_prompt_uids:
                                    kept_traj_idxs.append(idx)

                            new_batch = new_batch[kept_traj_idxs]
                            batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"[DAPO] {num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"[DAPO] {num_gen_batches=}. Keep generating...")
                                progress_bar.update(1)
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"[DAPO] {num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    # === UPDATING ===
                    batch.batch["response_mask"] = compute_response_mask(batch)

                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # === OLD_LOG_PROB ===
                    with marked_timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    if self.use_critic:
                        with marked_timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        )

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                            metrics.update(actor_output_metrics)

                    # === DUMP ROLLOUT GENERATIONS ===
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()

                            # Collect all reward-related fields from non_tensor_batch
                            reward_extra_infos_dict = {}
                            reward_fields = [
                                "request_id", "original_cot", "ground_truth",
                                # Reward breakdown fields
                                "cognitive_rewards", "useful_rewards",
                                "rewrite_mean_log_probs", "original_mean_log_probs", "no_cot_mean_log_probs",
                                "rewrite_ppls", "original_ppls",  # PPL values (ppl mode only)
                                "relative_improvement_rates",  # Formula 1: avg_prob(rewrite)/avg_prob(original) - 1 (observation)
                                "logic_rewards", "total_rewards", "original_fsfs", "rewrite_fsfs",
                                "ged", "ged_01", "logic_reward_valid_flags",
                                "reward_deltas",  # rewrite-original log_prob diff
                                "r_cog_before_correct_judge",
                                # Penalty fields
                                "truncation_penalties", "format_penalties", "too_short_penalties", "too_long_penalties",
                                "too_short_penalties_actual", "too_long_penalties_actual",
                                "repeat_penalties", "repeat_penalties_actual", "similarities",
                                "penalized_flags", "penalty_reasons",
                                # Correct Judge fields
                                "is_correct_results", "correct_judge_penalties",
                                # DAPO filter_groups flags
                                "truncated", "format_valid", "length_ratio",
                                # FSF parse status
                                "fsf_parse_failed_count", "fsf_group_skipped", "ged_parse_failed_count",
                                # FSF DOT outputs (raw FSF outputs for debugging/analysis)
                                "original_dot_outputs", "rewrite_dot_outputs",
                                # Saved prompts for debugging (student_model and fsf_judge requests)
                                "saved_prompts",
                                # KL answer divergence values (raw, before normalization)
                                "kl_answer_values",
                            ]
                            for field in reward_fields:
                                if field in batch.non_tensor_batch:
                                    reward_extra_infos_dict[field] = batch.non_tensor_batch[field].tolist()

                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # === VALIDATION ===
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    self._stop_profiling(do_profile)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1
