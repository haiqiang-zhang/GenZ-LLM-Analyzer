"""Model-parallel topology and feasibility helpers.

This module is deliberately model-name agnostic.  Parallelism support is
derived from :class:`GenZ.Models.ModelConfig` architecture metadata, while
runtime-specific disaggregated-KV restrictions are represented as explicit
capabilities.  This keeps callers such as RAG-CM from maintaining a second
table of attention/KV-head counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class ParallelismValidationError(ValueError):
    """Raised when a requested model-parallel layout is not realizable."""


@dataclass(frozen=True)
class KVShardLayout:
    """How a model's KV heads are laid out across tensor-parallel ranks.

    ``replicas_per_head`` is greater than one only when TP is wider than the
    number of KV heads.  ``head_partition_saturated`` also includes equality;
    this is exposed separately because some KV-transfer implementations treat
    ``TP == num_kv_heads`` as the boundary of their replicated-layout path.
    """

    total_kv_heads: int
    tensor_parallel: int
    kv_heads_per_rank: int
    unique_shards: int
    replicas_per_head: int
    head_partition_saturated: bool

    @property
    def replicated(self) -> bool:
        return self.replicas_per_head > 1


@dataclass(frozen=True)
class ModelParallelismCheck:
    """Structured result for a single model engine's TP/PP feasibility."""

    feasible: bool
    code: str
    reason: str
    model: str
    tensor_parallel: int
    pipeline_parallel: int
    attention_heads_per_rank: int | None = None
    kv_layout: KVShardLayout | None = None

    @property
    def parallelism(self) -> tuple[int, int]:
        return self.tensor_parallel, self.pipeline_parallel

    def require(self) -> "ModelParallelismCheck":
        """Return this result, or raise a stable validation exception."""

        if not self.feasible:
            raise ParallelismValidationError(self.reason)
        return self


@dataclass(frozen=True)
class KVTransferCapabilities:
    """Capabilities of a disaggregated prefill-to-decode KV transport.

    The fields describe transport mechanics, not particular models.  A caller
    can supply another capability object as connectors evolve without changing
    model metadata or adding model-name conditionals.
    """

    name: str
    requires_integral_tp_ratio: bool = True
    supports_saturated_producer_fan_in: bool = True
    supports_mamba_heterogeneous_tp: bool = True


GENERIC_KV_TRANSFER_CAPABILITIES = KVTransferCapabilities(
    name="generic",
)

# vLLM's NIXL connector requires an integral TP ratio.  It cannot currently
# fan in from a producer whose TP width has reached/exceeded its KV-head count,
# and it requires homogeneous TP for Mamba state transfer.  These are connector
# capabilities; the checks below still derive every model fact from ModelConfig.
NIXL_KV_TRANSFER_CAPABILITIES = KVTransferCapabilities(
    name="nixl",
    requires_integral_tp_ratio=True,
    supports_saturated_producer_fan_in=False,
    supports_mamba_heterogeneous_tp=False,
)


@dataclass(frozen=True)
class DisaggregatedParallelismCheck:
    """Structured result for a prefill/decode model-parallel pair."""

    feasible: bool
    code: str
    reason: str
    model: str
    prefill: ModelParallelismCheck
    decode: ModelParallelismCheck
    capabilities: KVTransferCapabilities
    tp_ratio: int | None = None

    def require(self) -> "DisaggregatedParallelismCheck":
        """Return this result, or raise a stable validation exception."""

        if not self.feasible:
            raise ParallelismValidationError(self.reason)
        return self


def _model_label(model: Any) -> str:
    return str(getattr(model, "model", model))


def _resolve_model_config(model: Any) -> Any:
    required = (
        "num_attention_heads",
        "num_key_value_heads",
        "num_decoder_layers",
    )
    if all(hasattr(model, field) for field in required):
        return model

    # Lazy import avoids the Models -> get_language_model -> parallelism cycle.
    from GenZ.Models.get_language_model import get_configs

    return get_configs(model)


def _as_positive_int(value: Any, field: str) -> tuple[int | None, str | None]:
    if isinstance(value, bool):
        return None, f"{field} must be a positive integer, got {value!r}"
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None, f"{field} must be a positive integer, got {value!r}"
    try:
        exact = float(value) == float(parsed)
    except (TypeError, ValueError, OverflowError):
        exact = False
    if parsed < 1 or not exact:
        return None, f"{field} must be a positive integer, got {value!r}"
    return parsed, None


def _invalid_model_check(
    *,
    code: str,
    reason: str,
    model: Any,
    tensor_parallel: int,
    pipeline_parallel: int,
) -> ModelParallelismCheck:
    return ModelParallelismCheck(
        feasible=False,
        code=code,
        reason=reason,
        model=_model_label(model),
        tensor_parallel=tensor_parallel,
        pipeline_parallel=pipeline_parallel,
    )


def check_model_parallelism(
    model: Any,
    tensor_parallel: int = 1,
    pipeline_parallel: int = 1,
    *,
    batch_size: int | None = None,
) -> ModelParallelismCheck:
    """Check whether one model engine can realize the requested TP/PP.

    Tensor parallelism must partition query heads exactly.  KV heads may either
    partition across ranks or replicate across ranks, but the relevant width
    must divide exactly in either direction.  Pipeline parallelism may use an
    uneven layer partition (as vLLM does), so it only needs to be no wider than
    the model's decoder-layer count.  When ``batch_size`` is provided, each PP
    stage must receive at least one request.

    Unknown registry names fail closed in the returned result.  Callers can
    support an external model without editing this function by passing a fully
    populated ``ModelConfig`` instance.
    """

    raw_tp = tensor_parallel
    raw_pp = pipeline_parallel
    tp, tp_error = _as_positive_int(raw_tp, "tensor_parallel")
    pp, pp_error = _as_positive_int(raw_pp, "pipeline_parallel")
    safe_tp = tp if tp is not None else 0
    safe_pp = pp if pp is not None else 0
    if tp_error:
        return _invalid_model_check(
            code="invalid_tensor_parallel",
            reason=tp_error,
            model=model,
            tensor_parallel=safe_tp,
            pipeline_parallel=safe_pp,
        )
    if pp_error:
        return _invalid_model_check(
            code="invalid_pipeline_parallel",
            reason=pp_error,
            model=model,
            tensor_parallel=safe_tp,
            pipeline_parallel=safe_pp,
        )
    assert tp is not None and pp is not None

    try:
        config = _resolve_model_config(model)
    except Exception as exc:  # noqa: BLE001 - a check must fail closed
        return _invalid_model_check(
            code="unknown_model",
            reason=f"Cannot resolve GenZ model {_model_label(model)!r}: {exc}",
            model=model,
            tensor_parallel=tp,
            pipeline_parallel=pp,
        )

    label = _model_label(config)
    try:
        attention_heads = int(config.num_attention_heads)
        kv_heads = int(config.num_key_value_heads)
        layers = int(config.num_decoder_layers)
    except (TypeError, ValueError, OverflowError, AttributeError) as exc:
        return _invalid_model_check(
            code="invalid_model_metadata",
            reason=f"Model {label!r} has invalid parallelism metadata: {exc}",
            model=config,
            tensor_parallel=tp,
            pipeline_parallel=pp,
        )
    if attention_heads < 1 or kv_heads < 1 or layers < 1:
        return _invalid_model_check(
            code="invalid_model_metadata",
            reason=(
                f"Model {label!r} requires positive attention heads, KV heads, "
                f"and decoder layers; got H={attention_heads}, Hkv={kv_heads}, "
                f"layers={layers}"
            ),
            model=config,
            tensor_parallel=tp,
            pipeline_parallel=pp,
        )

    if attention_heads % tp:
        return _invalid_model_check(
            code="attention_heads_not_divisible",
            reason=(
                f"Model {label!r} has {attention_heads} attention heads, which "
                f"cannot be evenly partitioned across tensor_parallel={tp}"
            ),
            model=config,
            tensor_parallel=tp,
            pipeline_parallel=pp,
        )

    if kv_heads >= tp:
        if kv_heads % tp:
            return _invalid_model_check(
                code="kv_heads_not_divisible",
                reason=(
                    f"Model {label!r} has {kv_heads} KV heads, which cannot be "
                    f"evenly partitioned across tensor_parallel={tp}"
                ),
                model=config,
                tensor_parallel=tp,
                pipeline_parallel=pp,
            )
        kv_layout = KVShardLayout(
            total_kv_heads=kv_heads,
            tensor_parallel=tp,
            kv_heads_per_rank=kv_heads // tp,
            unique_shards=tp,
            replicas_per_head=1,
            head_partition_saturated=tp >= kv_heads,
        )
    else:
        if tp % kv_heads:
            return _invalid_model_check(
                code="kv_replication_not_divisible",
                reason=(
                    f"tensor_parallel={tp} is wider than model {label!r}'s "
                    f"{kv_heads} KV heads but is not an integer replication "
                    "multiple"
                ),
                model=config,
                tensor_parallel=tp,
                pipeline_parallel=pp,
            )
        kv_layout = KVShardLayout(
            total_kv_heads=kv_heads,
            tensor_parallel=tp,
            kv_heads_per_rank=1,
            unique_shards=kv_heads,
            replicas_per_head=tp // kv_heads,
            head_partition_saturated=True,
        )

    if pp > layers:
        return _invalid_model_check(
            code="pipeline_parallel_exceeds_layers",
            reason=(
                f"pipeline_parallel={pp} exceeds model {label!r}'s "
                f"{layers} decoder layers"
            ),
            model=config,
            tensor_parallel=tp,
            pipeline_parallel=pp,
        )

    if batch_size is not None:
        batch, batch_error = _as_positive_int(batch_size, "batch_size")
        if batch_error:
            return _invalid_model_check(
                code="invalid_batch_size",
                reason=batch_error,
                model=config,
                tensor_parallel=tp,
                pipeline_parallel=pp,
            )
        assert batch is not None
        if batch < pp:
            return _invalid_model_check(
                code="pipeline_batch_too_small",
                reason=(
                    f"batch_size={batch} is smaller than pipeline_parallel={pp}; "
                    "at least one request per pipeline stage is required"
                ),
                model=config,
                tensor_parallel=tp,
                pipeline_parallel=pp,
            )

    return ModelParallelismCheck(
        feasible=True,
        code="ok",
        reason="",
        model=label,
        tensor_parallel=tp,
        pipeline_parallel=pp,
        attention_heads_per_rank=attention_heads // tp,
        kv_layout=kv_layout,
    )


def check_disaggregated_parallelism(
    model: Any,
    *,
    prefill_tp: int,
    prefill_pp: int = 1,
    decode_tp: int,
    decode_pp: int = 1,
    prefill_batch_size: int | None = None,
    decode_batch_size: int | None = None,
    capabilities: KVTransferCapabilities = NIXL_KV_TRANSFER_CAPABILITIES,
) -> DisaggregatedParallelismCheck:
    """Check a disaggregated prefill/decode pair and its KV transport."""

    prefill = check_model_parallelism(
        model,
        prefill_tp,
        prefill_pp,
        batch_size=prefill_batch_size,
    )
    decode = check_model_parallelism(
        model,
        decode_tp,
        decode_pp,
        batch_size=decode_batch_size,
    )
    label = prefill.model if prefill.code != "unknown_model" else decode.model

    if not prefill.feasible:
        return DisaggregatedParallelismCheck(
            feasible=False,
            code=f"prefill_{prefill.code}",
            reason=f"Invalid prefill parallelism: {prefill.reason}",
            model=label,
            prefill=prefill,
            decode=decode,
            capabilities=capabilities,
        )
    if not decode.feasible:
        return DisaggregatedParallelismCheck(
            feasible=False,
            code=f"decode_{decode.code}",
            reason=f"Invalid decode parallelism: {decode.reason}",
            model=label,
            prefill=prefill,
            decode=decode,
            capabilities=capabilities,
        )

    low_tp = min(prefill.tensor_parallel, decode.tensor_parallel)
    high_tp = max(prefill.tensor_parallel, decode.tensor_parallel)
    integral_ratio = high_tp % low_tp == 0
    tp_ratio = high_tp // low_tp if integral_ratio else None
    if capabilities.requires_integral_tp_ratio and not integral_ratio:
        return DisaggregatedParallelismCheck(
            feasible=False,
            code="non_integral_tp_ratio",
            reason=(
                f"KV transfer {capabilities.name!r} requires an integral TP "
                f"ratio, got prefill_tp={prefill.tensor_parallel} and "
                f"decode_tp={decode.tensor_parallel}"
            ),
            model=label,
            prefill=prefill,
            decode=decode,
            capabilities=capabilities,
        )

    config = _resolve_model_config(model)
    heterogeneous_tp = prefill.tensor_parallel != decode.tensor_parallel
    if (
        heterogeneous_tp
        and bool(getattr(config, "is_mamba", False))
        and not capabilities.supports_mamba_heterogeneous_tp
    ):
        return DisaggregatedParallelismCheck(
            feasible=False,
            code="mamba_heterogeneous_tp_unsupported",
            reason=(
                f"KV transfer {capabilities.name!r} does not support "
                f"heterogeneous TP for Mamba model {label!r}"
            ),
            model=label,
            prefill=prefill,
            decode=decode,
            capabilities=capabilities,
            tp_ratio=tp_ratio,
        )

    prefill_layout = prefill.kv_layout
    assert prefill_layout is not None
    if (
        prefill.tensor_parallel > decode.tensor_parallel
        and prefill_layout.head_partition_saturated
        and not capabilities.supports_saturated_producer_fan_in
    ):
        return DisaggregatedParallelismCheck(
            feasible=False,
            code="saturated_producer_fan_in_unsupported",
            reason=(
                f"KV transfer {capabilities.name!r} cannot fan in from "
                f"prefill_tp={prefill.tensor_parallel} to "
                f"decode_tp={decode.tensor_parallel} after the producer TP "
                f"has reached/exceeded model {label!r}'s "
                f"{prefill_layout.total_kv_heads} KV heads"
            ),
            model=label,
            prefill=prefill,
            decode=decode,
            capabilities=capabilities,
            tp_ratio=tp_ratio,
        )

    return DisaggregatedParallelismCheck(
        feasible=True,
        code="ok",
        reason="",
        model=label,
        prefill=prefill,
        decode=decode,
        capabilities=capabilities,
        tp_ratio=tp_ratio,
    )


def valid_parallelism_configs(
    model: Any,
    total_chips: int,
    *,
    exact_chips: bool = True,
    batch_size: int | None = None,
) -> tuple[ModelParallelismCheck, ...]:
    """Enumerate feasible TP/PP configurations within a chip budget.

    By default, every returned pair uses exactly ``total_chips``.  Passing
    ``exact_chips=False`` enumerates all products up to that budget.  Results
    are structured checks so callers retain head/KV-layout provenance.
    """

    chips, chips_error = _as_positive_int(total_chips, "total_chips")
    if chips_error:
        raise ValueError(chips_error)
    assert chips is not None

    try:
        resolved_model = _resolve_model_config(model)
    except Exception:  # noqa: BLE001 - unknown models have no valid configs
        return ()

    results: list[ModelParallelismCheck] = []
    for tp in range(1, chips + 1):
        for pp in range(1, chips // tp + 1):
            used = tp * pp
            if exact_chips and used != chips:
                continue
            result = check_model_parallelism(
                resolved_model,
                tensor_parallel=tp,
                pipeline_parallel=pp,
                batch_size=batch_size,
            )
            if result.feasible:
                results.append(result)
    return tuple(results)


class ParallelismConfig:
    r"""
    This is the configuration class to store the configuration of a Model Splitting.
    It is used to instantiate an LLM into multiple parallel units
    according to the specified arguments, defining the degree of various parallelism.
    Args:

    """
    def __init__(
        self,
        tensor_parallel=1,
        pipeline_parallel=1,
        data_parallel=1,
        expert_parallel=1,
        sequence_parallel=1,
        **kwargs,
    ):
        self.tensor_parallel = tensor_parallel
        self.pipeline_parallel = pipeline_parallel
        self.data_parallel = data_parallel
        self.expert_parallel = expert_parallel
        self.sequence_parallel = sequence_parallel
        self.total_chips = np.prod([
                            self.data_parallel,
                            self.expert_parallel,
                            self.sequence_parallel,
                            self.pipeline_parallel,
                            self.tensor_parallel])

        super().__init__(**kwargs)

    def __str__(self):
        return str(vars(self))
