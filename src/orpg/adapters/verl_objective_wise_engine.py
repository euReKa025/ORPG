from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict
from torch import Tensor
from verl.utils.py_functional import append_to_dict
from verl.utils.metric import AggregationType, Metric
from verl.utils.tensordict_utils import maybe_fix_3d_position_ids
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead

from orpg.adapters.verl_objective_wise import (
    OBJECTIVE_WISE_PROBE_VARIANT_NAMES,
)
from orpg.gradient_reconciliation import (
    GradientCollection,
    ReconciliationDiagnostics,
    cagrad_reconcile,
    correctness_priority_pcgrad_reconcile,
    modulewise_correctness_priority_pcgrad_reconcile,
    pcgrad_reconcile,
    sum_gradients,
)
from orpg.positive_gradient_reconciliation import positive_pcgrad_reconcile


@dataclass(frozen=True, slots=True)
class ObjectiveWiseEngineSettings:
    aggregator: str
    objective_count: int
    shared_regularizer: bool
    cagrad_conflict_aversion: float = 0.5
    final_clip_norm: float = 1.0
    gradient_probe_only: bool = False
    positive_rule: str = "off"
    positive_strength: float = 0.0
    positive_q: float = 0.5
    positive_preserve_norm: bool = True

    def __post_init__(self) -> None:
        if self.aggregator not in {
            "sum",
            "pcgrad",
            "pcgrad_priority",
            "pcgrad_modulewise_priority",
            "cagrad",
        }:
            raise ValueError(f"unsupported objective-wise aggregator: {self.aggregator}")
        if self.objective_count < 2:
            raise ValueError("objective-wise training requires at least two objectives")
        if (
            self.aggregator
            in {"pcgrad_priority", "pcgrad_modulewise_priority", "cagrad"}
            and self.objective_count != 2
        ):
            raise ValueError(
                "Stage A priority PCGrad and CAGrad require exactly two objectives"
            )
        if self.final_clip_norm <= 0.0:
            raise ValueError("final_clip_norm must be positive")
        if self.positive_rule not in {"off", "A", "B", "C"}:
            raise ValueError("unknown positive coordination rule")
        if not math.isfinite(self.positive_strength) or not 0 <= self.positive_strength <= 1:
            raise ValueError("positive strength must be finite and in [0,1]")
        if not math.isfinite(self.positive_q) or not 0 <= self.positive_q <= 1:
            raise ValueError("positive q must be finite and in [0,1]")
        if self.positive_rule == "off" and self.positive_strength != 0:
            raise ValueError("disabled positive rule requires zero strength")
        if self.positive_rule != "off" and (
            self.aggregator not in {"pcgrad", "pcgrad_priority"} or self.objective_count != 2
        ):
            raise ValueError("positive coordination requires a two-objective PCGrad branch")
        if not isinstance(self.positive_preserve_norm, bool):
            raise ValueError("positive_preserve_norm must be boolean")
        if self.gradient_probe_only and self.positive_rule != "off":
            raise ValueError("legacy gradient probe does not implement positive coordination")


def _as_metric_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


def _append_metrics(
    destination: dict[str, Any],
    source: Mapping[str, Any],
    *,
    prefix: str | None = None,
) -> None:
    append_to_dict(destination, dict(source), prefix=prefix or "")


def _append_losses(destination: list[Any], output: Mapping[str, Any]) -> None:
    if "loss" not in output:
        return
    destination.extend(_as_metric_list(output["loss"]))


class ObjectiveWiseFSDPEngineWithLMHead(FSDPEngineWithLMHead):
    """FSDP engine that reconciles reward-specific policy gradients."""

    def configure_objective_wise(
        self,
        *,
        settings: ObjectiveWiseEngineSettings,
        policy_loss_function: Callable[..., Any] | object,
    ) -> None:
        self._objective_wise_settings = settings
        self._objective_policy_loss_function = policy_loss_function

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        parameters = [parameter for parameter in self.module.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("objective-wise engine found no trainable parameters")
        return parameters

    def _trainable_parameter_groups(
        self,
        parameters: Sequence[torch.nn.Parameter],
    ) -> tuple[str, ...]:
        """Map the stable parameter order to immediate owning-module names."""

        named = [
            (name, parameter)
            for name, parameter in self.module.named_parameters()
            if parameter.requires_grad
        ]
        if len(named) != len(parameters) or any(
            named_parameter is not parameter
            for (_, named_parameter), parameter in zip(named, parameters, strict=True)
        ):
            raise RuntimeError("named parameter order differs from trainable parameter order")
        return tuple(
            name.rsplit(".", maxsplit=1)[0] if "." in name else name
            for name, _ in named
        )

    @staticmethod
    def _capture_gradients(
        parameters: Sequence[torch.nn.Parameter],
    ) -> list[Tensor]:
        return [
            (
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter)
            )
            for parameter in parameters
        ]

    @staticmethod
    def _assign_gradients(
        parameters: Sequence[torch.nn.Parameter],
        gradients: Sequence[Tensor],
    ) -> None:
        for parameter, gradient in zip(parameters, gradients, strict=True):
            if parameter.grad is None:
                parameter.grad = gradient.detach().clone()
            else:
                parameter.grad.copy_(gradient)

    def _reduce_gradient_scalar(self, value: Tensor) -> Tensor:
        group = self.get_data_parallel_group()
        if group is not None:
            torch.distributed.all_reduce(
                value,
                op=torch.distributed.ReduceOp.SUM,
                group=group,
            )
        return value

    def _reconcile(
        self,
        objective_gradients: Sequence[Sequence[Tensor]],
        *,
        parameter_groups: Sequence[object] | None = None,
    ) -> tuple[list[Tensor], ReconciliationDiagnostics]:
        settings = self._objective_wise_settings
        collection = GradientCollection(objectives=objective_gradients)
        if settings.positive_rule != "off":
            return positive_pcgrad_reconcile(
                collection, rule=settings.positive_rule, strength=settings.positive_strength,
                q=settings.positive_q, preserve_norm=settings.positive_preserve_norm,
                negative_branch="priority" if settings.aggregator == "pcgrad_priority" else "symmetric",
                reduce_scalar=self._reduce_gradient_scalar,
            )
        if settings.aggregator == "sum":
            return sum_gradients(
                collection,
                reduce_scalar=self._reduce_gradient_scalar,
            )
        if settings.aggregator == "pcgrad":
            return pcgrad_reconcile(
                collection,
                reduce_scalar=self._reduce_gradient_scalar,
            )
        if settings.aggregator == "pcgrad_priority":
            return correctness_priority_pcgrad_reconcile(
                collection,
                reduce_scalar=self._reduce_gradient_scalar,
            )
        if settings.aggregator == "pcgrad_modulewise_priority":
            return modulewise_correctness_priority_pcgrad_reconcile(
                collection,
                parameter_groups=parameter_groups,
                reduce_scalar=self._reduce_gradient_scalar,
            )
        return cagrad_reconcile(
            collection,
            conflict_aversion=settings.cagrad_conflict_aversion,
            reduce_scalar=self._reduce_gradient_scalar,
        )

    def _gradient_norm(self, gradients: Sequence[Tensor]) -> float:
        local_squared = sum(
            (gradient.float().square().sum() for gradient in gradients),
            start=torch.zeros(
                (),
                device=gradients[0].device,
                dtype=torch.float32,
            ),
        )
        squared = self._reduce_gradient_scalar(local_squared)
        return float(squared.clamp_min(0.0).sqrt().detach().cpu())

    def _relative_gradient_distance(
        self,
        candidate: Sequence[Tensor],
        reference: Sequence[Tensor],
        *,
        epsilon: float = 1e-12,
    ) -> float:
        difference = [
            candidate_tensor - reference_tensor
            for candidate_tensor, reference_tensor in zip(
                candidate,
                reference,
                strict=True,
            )
        ]
        return self._gradient_norm(difference) / max(
            self._gradient_norm(reference),
            epsilon,
        )

    def _run_gradient_probe(
        self,
        data: TensorDict,
        *,
        loss_function: Callable[..., Any],
        parameters: Sequence[torch.nn.Parameter],
        original_advantages: Tensor,
    ) -> dict[str, Any]:
        """Run preregistered exact backwards without mutating model parameters."""

        settings = self._objective_wise_settings
        if "objective_probe_rollout_advantages" not in data:
            raise KeyError("gradient probe batch has no compact probe advantages")
        if "loss_mask" not in data:
            raise KeyError("gradient probe batch has no loss_mask")
        probe = data["objective_probe_rollout_advantages"]
        if probe.ndim != 3:
            raise ValueError("probe advantages must have shape [B, V, K]")
        if probe.shape[1] != len(OBJECTIVE_WISE_PROBE_VARIANT_NAMES):
            raise ValueError("probe advantage count differs from preregistered variants")
        if probe.shape[2] != settings.objective_count:
            raise ValueError("probe objective count differs from engine settings")
        loss_mask = data["loss_mask"]
        parameter_groups = self._trainable_parameter_groups(parameters)

        shared_gradients = [torch.zeros_like(parameter) for parameter in parameters]
        if settings.shared_regularizer:
            data["advantages"] = torch.zeros_like(original_advantages)
            self.optimizer_zero_grad()
            self.forward_backward_batch(
                data,
                loss_function,
                forward_only=False,
            )
            shared_gradients = self._capture_gradients(parameters)

        metrics: dict[str, Any] = {}
        reconcilers: tuple[
            tuple[
                str,
                Callable[[GradientCollection], tuple[list[Tensor], ReconciliationDiagnostics]],
            ],
            ...,
        ] = (
            (
                "sum",
                lambda collection: sum_gradients(
                    collection,
                    reduce_scalar=self._reduce_gradient_scalar,
                ),
            ),
            (
                "symmetric_pcgrad",
                lambda collection: pcgrad_reconcile(
                    collection,
                    reduce_scalar=self._reduce_gradient_scalar,
                ),
            ),
            (
                "correctness_priority",
                lambda collection: correctness_priority_pcgrad_reconcile(
                    collection,
                    reduce_scalar=self._reduce_gradient_scalar,
                ),
            ),
            (
                "modulewise_priority",
                lambda collection: modulewise_correctness_priority_pcgrad_reconcile(
                    collection,
                    parameter_groups=parameter_groups,
                    reduce_scalar=self._reduce_gradient_scalar,
                ),
            ),
            (
                "cagrad",
                lambda collection: cagrad_reconcile(
                    collection,
                    conflict_aversion=settings.cagrad_conflict_aversion,
                    reduce_scalar=self._reduce_gradient_scalar,
                ),
            ),
        )

        try:
            for variant_index, variant_name in enumerate(
                OBJECTIVE_WISE_PROBE_VARIANT_NAMES
            ):
                objective_gradients: list[list[Tensor]] = []
                for objective_index in range(settings.objective_count):
                    rollout_advantage = probe[:, variant_index, objective_index]
                    data["advantages"] = (
                        rollout_advantage.unsqueeze(-1).to(
                            device=loss_mask.device,
                            dtype=original_advantages.dtype,
                        )
                        * loss_mask
                    )
                    self.optimizer_zero_grad()
                    self.forward_backward_batch(
                        data,
                        self._objective_policy_loss_function,
                        forward_only=False,
                    )
                    objective_gradients.append(self._capture_gradients(parameters))

                correctness_final = [
                    correctness + shared
                    for correctness, shared in zip(
                        objective_gradients[0],
                        shared_gradients,
                        strict=True,
                    )
                ]
                correctness_norm = self._gradient_norm(correctness_final)
                prefix = f"objective_probe/{variant_name}"
                metrics[f"{prefix}/correctness_only/final_norm"] = [
                    correctness_norm
                ]
                metrics[
                    f"{prefix}/correctness_only/relative_distance_to_correctness"
                ] = [0.0]

                collection = GradientCollection(objectives=objective_gradients)
                for reconciler_name, reconcile in reconcilers:
                    reconciled, diagnostics = reconcile(collection)
                    final = [
                        policy + shared
                        for policy, shared in zip(
                            reconciled,
                            shared_gradients,
                            strict=True,
                        )
                    ]
                    reconcile_prefix = f"{prefix}/{reconciler_name}"
                    metrics[f"{reconcile_prefix}/final_norm"] = [
                        self._gradient_norm(final)
                    ]
                    metrics[
                        f"{reconcile_prefix}/relative_distance_to_correctness"
                    ] = [
                        self._relative_gradient_distance(
                            final,
                            correctness_final,
                        )
                    ]
                    metrics[f"{reconcile_prefix}/cosine_similarity"] = [
                        diagnostics.cosine_similarity
                    ]
                    metrics[f"{reconcile_prefix}/projection_rate"] = [
                        diagnostics.projection_rate
                    ]
                    metrics[f"{reconcile_prefix}/local_projection_rate"] = [
                        diagnostics.local_projection_rate
                    ]
                    metrics[f"{reconcile_prefix}/active_objective_count"] = [
                        float(diagnostics.active_objective_count)
                    ]
                    metrics[f"{reconcile_prefix}/solver_bypassed"] = [
                        float(diagnostics.solver_bypassed)
                    ]
                    for objective_index, norm in enumerate(
                        diagnostics.objective_norms
                    ):
                        metrics[
                            f"{reconcile_prefix}/objective_{objective_index}_norm"
                        ] = [norm]
        finally:
            data["advantages"] = original_advantages
            self.optimizer_zero_grad()

        metrics["objective_probe/completed"] = [1.0]
        metrics["objective_probe/variant_count"] = [
            float(len(OBJECTIVE_WISE_PROBE_VARIANT_NAMES))
        ]
        return metrics

    def train_batch(
        self,
        data: TensorDict,
        loss_function: Callable[..., Any],
    ) -> dict[str, Any]:
        maybe_fix_3d_position_ids(data)
        if not hasattr(self, "_objective_wise_settings"):
            raise RuntimeError("objective-wise FSDP engine was not configured")
        settings = self._objective_wise_settings
        if "objective_advantages" not in data:
            raise KeyError("objective-wise batch has no objective_advantages")
        objective_advantages = data["objective_advantages"]
        if objective_advantages.ndim != 3:
            raise ValueError("objective_advantages must have shape [B, K, R]")
        if objective_advantages.shape[1] != settings.objective_count:
            raise ValueError("objective advantage count differs from engine settings")

        parameters = self._trainable_parameters()
        original_advantages = data["advantages"]
        if settings.gradient_probe_only:
            metrics = self._run_gradient_probe(
                data,
                loss_function=loss_function,
                parameters=parameters,
                original_advantages=original_advantages,
            )
            if self.is_mp_src_rank_with_outputs():
                metrics["grad_norm"] = 0.0
            return {
                "model_output": {},
                "loss": [0.0],
                "metrics": metrics,
            }
        objective_outputs: list[Mapping[str, Any]] = []
        objective_gradients: list[list[Tensor]] = []
        objective_seconds: list[float] = []
        shared_output: Mapping[str, Any] | None = None
        shared_gradients = [torch.zeros_like(parameter) for parameter in parameters]

        try:
            for objective_index in range(settings.objective_count):
                data["advantages"] = objective_advantages[:, objective_index, :]
                self.optimizer_zero_grad()
                started_at = time.perf_counter()
                output = self.forward_backward_batch(
                    data,
                    self._objective_policy_loss_function,
                    forward_only=False,
                )
                objective_seconds.append(time.perf_counter() - started_at)
                objective_outputs.append(output)
                objective_gradients.append(self._capture_gradients(parameters))

            reconcile_started_at = time.perf_counter()
            parameter_groups = (
                self._trainable_parameter_groups(parameters)
                if settings.aggregator == "pcgrad_modulewise_priority"
                else None
            )
            reconciled, diagnostics = self._reconcile(
                objective_gradients,
                parameter_groups=parameter_groups,
            )
            reconciliation_seconds = time.perf_counter() - reconcile_started_at

            if settings.shared_regularizer:
                data["advantages"] = torch.zeros_like(original_advantages)
                self.optimizer_zero_grad()
                shared_output = self.forward_backward_batch(
                    data,
                    loss_function,
                    forward_only=False,
                )
                shared_gradients = self._capture_gradients(parameters)

            final_gradients = [
                policy_gradient + shared_gradient
                for policy_gradient, shared_gradient in zip(
                    reconciled,
                    shared_gradients,
                    strict=True,
                )
            ]
            self.optimizer_zero_grad()
            self._assign_gradients(parameters, final_gradients)
            grad_norm = self.optimizer_step()
        finally:
            data["advantages"] = original_advantages

        metrics: dict[str, Any] = {}
        losses: list[Any] = []
        for objective_index, output in enumerate(objective_outputs):
            output_metrics = output.get("metrics", {})
            _append_metrics(metrics, output_metrics)
            _append_metrics(
                metrics,
                output_metrics,
                prefix=f"objective_wise/objective_{objective_index}/",
            )
            _append_losses(losses, output)
        if shared_output is not None:
            shared_metrics = {
                key: value
                for key, value in shared_output.get("metrics", {}).items()
                if key in {"kl_loss", "kl_coef", "actor/entropy_loss"}
            }
            _append_metrics(metrics, shared_metrics)
            _append_losses(losses, shared_output)

        metrics.update(
            {
                "objective_wise/cosine_similarity": [diagnostics.cosine_similarity],
                "objective_wise/conflict_rate": [diagnostics.conflict_rate],
                "objective_wise/projection_rate": [diagnostics.projection_rate],
                "objective_wise/local_conflict_rate": [
                    diagnostics.local_conflict_rate
                ],
                "objective_wise/local_projection_rate": [
                    diagnostics.local_projection_rate
                ],
                "objective_wise/active_objective_count": [
                    float(diagnostics.active_objective_count)
                ],
                "objective_wise/solver_bypassed": [
                    float(diagnostics.solver_bypassed)
                ],
                "objective_wise/combined_norm_before": [
                    diagnostics.combined_norm_before
                ],
                "objective_wise/combined_norm_after": [
                    diagnostics.combined_norm_after
                ],
                "objective_wise/reconciliation_seconds": [reconciliation_seconds],
                "objective_wise/repeated_backward_seconds": [
                    sum(objective_seconds)
                ],
                "objective_wise/final_grad_norm_before_clip": [grad_norm],
                "objective_wise/final_grad_norm_after_clip": [
                    min(grad_norm, settings.final_clip_norm)
                    if math.isfinite(grad_norm)
                    else grad_norm
                ],
                "objective_wise/final_clip_activated": [
                    float(math.isfinite(grad_norm) and grad_norm > settings.final_clip_norm)
                ],
            }
        )
        for objective_index, norm in enumerate(diagnostics.objective_norms):
            metrics[f"objective_wise/objective_{objective_index}/gradient_norm"] = [
                norm
            ]

        if settings.positive_rule != "off":
            pair = int(diagnostics.active_objective_count == 2)
            negative = int(pair and diagnostics.conflict_rate > 0)
            counts = {
                "coordination_calls": 1,
                "pair_opportunities": pair,
                "positive_opportunities": pair-negative,
                "negative_pairs": negative,
                "bypass_calls": int(diagnostics.solver_bypassed),
                "positive_branch_calls": int(getattr(diagnostics, "positive_branch_entered", False)),
                "positive_changed_calls": int(getattr(diagnostics, "positive_direction_changed", False)),
            }
            # Pinned Verl SUM sums mini-batches, while aggregate_dp averages
            # replicated rank counters. Counts therefore describe global pairs,
            # not the number of FSDP shard workers.
            for key, value in counts.items():
                metrics[f"objective_wise/positive/{key}"] = Metric(
                    value=value, aggregation=AggregationType.SUM)
            norms = diagnostics.objective_norms
            metrics.update({
                "objective_wise/positive/alpha": [getattr(diagnostics, "positive_alpha", 0.0)],
                "objective_wise/positive/reference_angle_radians": [getattr(diagnostics, "reference_angle_radians", 0.0)],
                "objective_wise/positive/output_norm_ratio": [
                    diagnostics.combined_norm_after/diagnostics.combined_norm_before
                    if diagnostics.combined_norm_before > 0 else 1.0
                ],
                "objective_wise/positive/norm_ratio_valid": [float(pair)],
                "objective_wise/positive/primary_secondary_norm_ratio": [norms[0]/norms[1] if pair else 0.0],
            })

        if self.is_mp_src_rank_with_outputs():
            metrics["grad_norm"] = grad_norm  # type: ignore[assignment]
        model_output = (
            shared_output.get("model_output", {})
            if shared_output is not None
            else objective_outputs[-1].get("model_output", {})
        )
        return {
            "model_output": model_output,
            "loss": losses,
            "metrics": metrics,
        }
