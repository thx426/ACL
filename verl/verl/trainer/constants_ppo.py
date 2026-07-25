# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import json
import os

import torch
from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

from verl.utils.device import get_device_capability

_major, _ = get_device_capability()
# Opt-in GB200 NCCL WAR: set TLLM_DISABLE_NVLS_MNNVL=1 in the launch shell to disable
# both NCCL_NVLS_ENABLE and NCCL_MNNVL_ENABLE on Blackwell. Required by async-RL
# Megatron on GB200 nodes without IMEX (mbridge all_gather raises NCCL 801).
_gb200_nccl_env = {}
if (_major or 0) >= 10 and os.environ.get("TLLM_DISABLE_NVLS_MNNVL", "0") == "1":
    _gb200_nccl_env = {"NCCL_NVLS_ENABLE": "0", "NCCL_MNNVL_ENABLE": "0"}

# On ROCm, Ray 2.x force-clears accelerator visibility for num_gpus=0 actors
# (e.g. the SGLang server actor), leaving them unable to see any GPU. Disable
# that override so the actor keeps its HIP visibility. Scoped to ROCm to avoid
# changing Ray's default behavior on other platforms.
_rocm_ray_env = {}
if torch.version.hip is not None:
    _rocm_ray_env = {"RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO": "0"}

# CUDA_DEVICE_MAX_CONNECTIONS=1 is only needed by the Megatron engine on
# pre-Blackwell Hopper/Ampere GPUs (compute capability 8.x/9.x) to serialize
# kernel launches for TP/CP communication overlap. Blackwell/GB200 (>=10) does
# not need it, and Torch-FSDP2 / Megatron-FSDP must NOT set it to 1. So we set
# it conditionally at runtime (see get_ppo_ray_runtime_env), not unconditionally.
#
# https://github.com/NVIDIA/Megatron-LM/blob/core_v0.18.0/skills/mcore-run-on-slurm/SKILL.md#cuda_device_max_connections
_is_hopper_or_ampere = (_major or 0) in (8, 9)


def _uses_megatron(config) -> bool:
    """Return True if any trainable engine in the config uses the Megatron strategy."""
    if config is None:
        return False
    from omegaconf import OmegaConf

    for key in ("actor_rollout_ref.actor.strategy", "critic.strategy"):
        if OmegaConf.select(config, key, default=None) == "megatron":
            return True
    return False


def _uses_mindspeed(config) -> bool:
    """Return True if any trainable engine in the config uses the mindspeed strategy."""
    if config is None:
        return False
    from omegaconf import OmegaConf

    for key in ("actor_rollout_ref.actor.strategy", "critic.strategy"):
        if OmegaConf.select(config, key, default=None) == "mindspeed":
            return True
    return False


PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # Needed for multi-processes colocated on same NPU device
        # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
        "HSA_NO_SCRATCH_RECLAIM": "1",
        **_gb200_nccl_env,
        **_rocm_ray_env,
    },
}


def get_ppo_ray_runtime_env(config=None):
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.

    Args:
        config: Optional training config. When the engine strategy is Megatron
            and the GPU is Hopper/Ampere, CUDA_DEVICE_MAX_CONNECTIONS=1 is set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    # Only Megatron on Hopper/Ampere needs CUDA_DEVICE_MAX_CONNECTIONS=1.
    if _is_hopper_or_ampere and _uses_megatron(config):
        runtime_env["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    if _uses_mindspeed(config):
        runtime_env["env_vars"]["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)
    # Always forward these at call-time, not import-time.
    for key in ("PYTHONHASHSEED", "VERL_FULL_DETERMINISM", "VLLM_BATCH_INVARIANT", "VERL_RL_INSIGHT_ENABLE"):
        runtime_env["env_vars"][key] = os.environ.get(key, "0")
    # trace-opd distillation microscope: workers sample per-token distillation
    # quantities to this directory when set (see trainer/distillation/losses.py).
    # trace-opd R11 (codex M5): the teacher-scoring backend switch and the
    # agent-loop hang-guard timeout must reach every Ray worker explicitly —
    # process-inheritance is not guaranteed for existing/remote clusters.
    for key in ("TRACE_TEACHER_SCORING_BACKEND", "TRACE_AGENT_TASK_TIMEOUT_S"):
        if os.environ.get(key):
            runtime_env["env_vars"][key] = os.environ[key]
    # trace-opd (CD-006 2026-07-25): the caching-allocator GC threshold is a
    # memory-safety setting, so it must reach the workers by the same explicit
    # route rather than relying on inheritance. Only forwarded when the launcher
    # actually set it, so the upstream default stays untouched otherwise.
    if os.environ.get("PYTORCH_CUDA_ALLOC_CONF"):
        runtime_env["env_vars"]["PYTORCH_CUDA_ALLOC_CONF"] = os.environ["PYTORCH_CUDA_ALLOC_CONF"]
    for key in ("TRACE_DISTILL_MICROSCOPE_DIR", "TRACE_DISTILL_SAMPLE_EVERY"):
        if os.environ.get(key):
            runtime_env["env_vars"][key] = os.environ[key]
    return runtime_env
