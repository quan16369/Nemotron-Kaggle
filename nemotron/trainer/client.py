"""ServiceClient and TrainingClient that support both tinker and modal backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import modal
from tinker.types import AdamParams

T = TypeVar("T")


@dataclass
class LogprobsData:
    """Mirrors tinker logprobs: result.loss_fn_outputs[i]["logprobs"].data"""

    data: list[float]


class Future(Generic[T]):
    """Simple resolved future matching tinker's async future interface."""

    def __init__(self, value: T) -> None:
        self._value = value

    async def result_async(self) -> T:
        return self._value


class TrainingClient:
    """Training client that delegates to either tinker or modal backend."""

    def __init__(
        self,
        backend: Literal["tinker", "modal"],
        tinker_client: object | None = None,
        modal_remote: modal.cls.Obj | None = None,
    ) -> None:
        self._backend = backend
        self._tinker_client = tinker_client
        self._modal_remote = modal_remote

    async def forward_backward_async(
        self,
        data: list,
        loss_fn: str = "cross_entropy",
        loss_fn_config: dict[str, float] | None = None,
        micro_batch_size: int | None = None,
    ) -> Future:
        if loss_fn_config is None:
            loss_fn_config = {}

        if self._backend == "tinker":
            assert self._tinker_client is not None
            return await self._tinker_client.forward_backward_async(  # type: ignore[attr-defined]
                data, loss_fn=loss_fn, loss_fn_config=loss_fn_config
            )

        # Modal: serialize Datum objects into plain lists for RPC
        batch_tokens: list[list[int]] = []
        batch_target_tokens: list[list[int]] = []
        batch_weights: list[list[float]] = []

        for datum in data:
            batch_tokens.append(datum.model_input.chunks[0].tokens)
            batch_target_tokens.append(datum.loss_fn_inputs["target_tokens"].data)
            if "weights" in datum.loss_fn_inputs:
                batch_weights.append(datum.loss_fn_inputs["weights"].data)
            elif "advantages" in datum.loss_fn_inputs:
                batch_weights.append(datum.loss_fn_inputs["advantages"].data)
            else:
                batch_weights.append([1.0] * len(batch_target_tokens[-1]))

        assert self._modal_remote is not None
        raw = await self._modal_remote.forward_backward.remote.aio(
            batch_tokens=batch_tokens,
            batch_target_tokens=batch_target_tokens,
            batch_weights=batch_weights,
            loss_fn=loss_fn,
            loss_fn_config=loss_fn_config,
            micro_batch_size=micro_batch_size,
        )

        # Wrap raw dicts to match tinker's result interface
        loss_fn_outputs = [
            {"logprobs": LogprobsData(data=entry["logprobs"])}
            for entry in raw["loss_fn_outputs"]
        ]
        result = _ForwardBackwardResult(
            loss_fn_outputs=loss_fn_outputs,
            metrics=raw.get("metrics", {}),
        )
        return Future(result)

    async def optim_step_async(self, adam_params: AdamParams) -> Future:
        if self._backend == "tinker":
            assert self._tinker_client is not None
            return await self._tinker_client.optim_step_async(adam_params)  # type: ignore[attr-defined]

        assert self._modal_remote is not None
        raw = await self._modal_remote.optim_step.remote.aio(
            learning_rate=adam_params.learning_rate,
            beta1=adam_params.beta1,
            beta2=adam_params.beta2,
            eps=adam_params.eps,
            weight_decay=adam_params.weight_decay,
            grad_clip_norm=adam_params.grad_clip_norm,
        )
        return Future(_OptimResult(metrics=raw))

    async def save_checkpoint_async(self, name: str, log_path: str) -> None:
        if self._backend == "tinker":
            from tinker_cookbook import checkpoint_utils

            loop_state = {"name": name}
            await checkpoint_utils.save_checkpoint_async(
                training_client=self._tinker_client,
                name=name,
                log_path=log_path,
                kind="both",
                loop_state=loop_state,
            )
        else:
            assert self._modal_remote is not None
            await self._modal_remote.save_checkpoint.remote.aio(path="/adapter/weights")



class ServiceClient:
    """ServiceClient that supports both tinker and modal backends."""

    def __init__(self, backend: Literal["tinker", "modal"] = "tinker") -> None:
        self.backend = backend

    async def create_lora_training_client_async(
        self,
        base_model: str,
        rank: int,
        train_mlp: bool = True,
        train_attn: bool = True,
        train_unembed: bool = True,
    ) -> TrainingClient:
        if self.backend == "modal":
            remote = modal.Cls.from_name("trainer-gpu", "Trainer")()
            # Model init happens via @modal.enter() on container startup.
            # First method call will block until init completes.
            return TrainingClient(backend="modal", modal_remote=remote)

        import tinker

        tinker_sc = tinker.ServiceClient()
        tinker_tc = await tinker_sc.create_lora_training_client_async(
            base_model=base_model,
            rank=rank,
            train_mlp=train_mlp,
            train_attn=train_attn,
            train_unembed=train_unembed,
        )
        return TrainingClient(backend="tinker", tinker_client=tinker_tc)


@dataclass
class _ForwardBackwardResult:
    loss_fn_outputs: list[dict[str, LogprobsData]]
    metrics: dict[str, float]


@dataclass
class _OptimResult:
    metrics: dict[str, float]
