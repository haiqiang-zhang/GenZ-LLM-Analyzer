import pytest

from GenZ import (
    GENERIC_KV_TRANSFER_CAPABILITIES,
    NIXL_KV_TRANSFER_CAPABILITIES,
    ParallelismValidationError,
    check_disaggregated_parallelism,
    check_model_parallelism,
    valid_parallelism_configs,
)
from GenZ.LLM_inference.best_parallelization import get_various_parallization
from GenZ.Models import ModelConfig, get_configs
from GenZ.Models.get_language_model import (
    create_full_chunked_model,
    create_full_decode_model,
    create_full_prefill_model,
    create_inference_moe_prefill_layer,
)


QWEN_ARCHITECTURES = {
    "Qwen/Qwen2.5-1.5B-Instruct": (12, 2, 28),
    "Qwen/Qwen2.5-3B-Instruct": (16, 2, 36),
    "Qwen/Qwen2.5-7B-Instruct": (28, 4, 28),
    "Qwen/Qwen2.5-14B-Instruct": (40, 8, 48),
}


@pytest.mark.parametrize("alias,expected", QWEN_ARCHITECTURES.items())
def test_qwen_instruct_aliases_resolve_architecture(alias, expected):
    config = get_configs(alias)

    assert (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.num_decoder_layers,
    ) == expected
    # ModelCollection also registers the unqualified alias generically.
    assert get_configs(alias.split("/", 1)[1]) is config


@pytest.mark.parametrize("alias", QWEN_ARCHITECTURES)
@pytest.mark.parametrize("tp", [1, 2, 4])
@pytest.mark.parametrize("pp", [1, 2, 4])
def test_qwen_ladder_supports_expected_single_engine_tp_pp(alias, tp, pp):
    result = check_model_parallelism(alias, tp, pp)

    assert result.feasible, result.reason
    assert result.code == "ok"
    assert result.parallelism == (tp, pp)
    assert result.attention_heads_per_rank == QWEN_ARCHITECTURES[alias][0] // tp
    assert result.require() is result


def test_pipeline_parallel_allows_uneven_layers_but_not_empty_stages():
    uneven = check_model_parallelism(
        "Qwen/Qwen2.5-1.5B-Instruct", tensor_parallel=1, pipeline_parallel=3
    )
    too_wide = check_model_parallelism(
        "Qwen/Qwen2.5-1.5B-Instruct", tensor_parallel=1, pipeline_parallel=29
    )

    assert uneven.feasible
    assert not too_wide.feasible
    assert too_wide.code == "pipeline_parallel_exceeds_layers"


def test_attention_and_kv_partition_fail_closed_independently():
    config = ModelConfig(
        model="test/partition-geometry",
        num_attention_heads=24,
        num_key_value_heads=6,
        num_decoder_layers=8,
    )

    attention = check_model_parallelism(config, tensor_parallel=5)
    kv_partition = check_model_parallelism(config, tensor_parallel=4)
    kv_replication = check_model_parallelism(config, tensor_parallel=8)

    assert attention.code == "attention_heads_not_divisible"
    assert kv_partition.code == "kv_heads_not_divisible"
    assert kv_replication.code == "kv_replication_not_divisible"


def test_kv_layout_distinguishes_partition_boundary_and_replication():
    partitioned = check_model_parallelism(
        "Qwen/Qwen2.5-1.5B-Instruct", tensor_parallel=2
    ).require()
    replicated = check_model_parallelism(
        "Qwen/Qwen2.5-1.5B-Instruct", tensor_parallel=4
    ).require()

    assert partitioned.kv_layout is not None
    assert partitioned.kv_layout.kv_heads_per_rank == 1
    assert partitioned.kv_layout.replicas_per_head == 1
    assert not partitioned.kv_layout.replicated
    assert partitioned.kv_layout.head_partition_saturated

    assert replicated.kv_layout is not None
    assert replicated.kv_layout.kv_heads_per_rank == 1
    assert replicated.kv_layout.replicas_per_head == 2
    assert replicated.kv_layout.replicated
    assert replicated.kv_layout.head_partition_saturated


def test_pipeline_parallel_accepts_batch_smaller_than_stage_count():
    result = check_model_parallelism(
        "Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel=1,
        pipeline_parallel=4,
        batch_size=2,
    )

    assert result.feasible
    assert result.code == "ok"


def test_unknown_model_fails_closed_and_require_raises():
    result = check_model_parallelism("not-a-real/model", 1, 1)

    assert not result.feasible
    assert result.code == "unknown_model"
    with pytest.raises(ParallelismValidationError, match="Cannot resolve GenZ model"):
        result.require()


def test_model_config_can_be_supplied_without_registry_entry():
    config = ModelConfig(
        model="private/offline-model",
        num_attention_heads=18,
        num_key_value_heads=3,
        num_decoder_layers=7,
    )

    result = check_model_parallelism(config, tensor_parallel=3, pipeline_parallel=2)

    assert result.feasible
    assert result.model == "private/offline-model"


def test_valid_parallelism_configs_exact_budget_and_batch_filter():
    all_configs = valid_parallelism_configs(
        "Qwen/Qwen2.5-7B-Instruct", total_chips=4
    )
    batch_filtered = valid_parallelism_configs(
        "Qwen/Qwen2.5-7B-Instruct", total_chips=4, batch_size=2
    )

    assert [result.parallelism for result in all_configs] == [
        (1, 4),
        (2, 2),
        (4, 1),
    ]
    assert [result.parallelism for result in batch_filtered] == [
        (1, 4),
        (2, 2),
        (4, 1),
    ]


def test_valid_parallelism_configs_can_enumerate_up_to_budget():
    results = valid_parallelism_configs(
        "Qwen/Qwen2.5-1.5B-Instruct",
        total_chips=3,
        exact_chips=False,
    )

    assert {result.parallelism for result in results} == {
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
    }


@pytest.mark.parametrize(
    "model,prefill_tp,decode_tp,feasible",
    [
        ("Qwen/Qwen2.5-1.5B-Instruct", 2, 1, False),
        ("Qwen/Qwen2.5-1.5B-Instruct", 1, 2, True),
        ("Qwen/Qwen2.5-3B-Instruct", 4, 2, False),
        ("Qwen/Qwen2.5-7B-Instruct", 2, 1, True),
        ("Qwen/Qwen2.5-7B-Instruct", 4, 2, False),
        ("Qwen/Qwen2.5-14B-Instruct", 4, 1, True),
    ],
)
def test_nixl_qwen_heterogeneous_tp_matrix(
    model, prefill_tp, decode_tp, feasible
):
    result = check_disaggregated_parallelism(
        model,
        prefill_tp=prefill_tp,
        decode_tp=decode_tp,
        capabilities=NIXL_KV_TRANSFER_CAPABILITIES,
    )

    assert result.feasible is feasible
    if feasible:
        assert result.code == "ok"
        assert result.require() is result
    else:
        assert result.code == "saturated_producer_fan_in_unsupported"
        with pytest.raises(ParallelismValidationError):
            result.require()


def test_disaggregated_tp_ratio_must_be_integral():
    config = ModelConfig(
        model="test/non-integral-ratio",
        num_attention_heads=12,
        num_key_value_heads=6,
        num_decoder_layers=8,
    )

    result = check_disaggregated_parallelism(
        config, prefill_tp=3, decode_tp=2
    )

    assert not result.feasible
    assert result.code == "non_integral_tp_ratio"
    assert result.tp_ratio is None


def test_transfer_capability_can_allow_saturated_producer_fan_in():
    result = check_disaggregated_parallelism(
        "Qwen/Qwen2.5-1.5B-Instruct",
        prefill_tp=2,
        decode_tp=1,
        capabilities=GENERIC_KV_TRANSFER_CAPABILITIES,
    )

    assert result.feasible
    assert result.tp_ratio == 2


def test_nixl_rejects_mamba_heterogeneous_tp_from_model_metadata():
    config = ModelConfig(
        model="test/mamba",
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_decoder_layers=8,
        mamba_d_state=16,
        mamba_d_conv=4,
    )

    nixl = check_disaggregated_parallelism(
        config, prefill_tp=2, decode_tp=1
    )
    generic = check_disaggregated_parallelism(
        config,
        prefill_tp=2,
        decode_tp=1,
        capabilities=GENERIC_KV_TRANSFER_CAPABILITIES,
    )

    assert not nixl.feasible
    assert nixl.code == "mamba_heterogeneous_tp_unsupported"
    assert generic.feasible


@pytest.mark.parametrize(
    "builder,kwargs",
    [
        (create_full_prefill_model, {"input_sequence_length": 8}),
        (
            create_full_decode_model,
            {"input_sequence_length": 8, "output_gen_tokens": 1},
        ),
        (
            create_full_chunked_model,
            {"prefill_kv_sizes": [(0, 8)], "decode_kv_sizes": []},
        ),
    ],
)
def test_main_model_construction_entrypoints_fail_fast(builder, kwargs):
    config = ModelConfig(
        model="test/invalid-entrypoint-tp",
        num_attention_heads=12,
        num_key_value_heads=2,
        num_decoder_layers=8,
    )

    with pytest.raises(ParallelismValidationError, match="attention heads"):
        builder(
            name=config,
            tensor_parallel=5,
            pipeline_parallel=1,
            **kwargs,
        )


def test_low_level_attention_construction_cannot_bypass_validation():
    config = ModelConfig(
        model="test/invalid-low-level-tp",
        num_attention_heads=12,
        num_key_value_heads=2,
        num_decoder_layers=8,
    )

    with pytest.raises(ParallelismValidationError, match="attention heads"):
        create_inference_moe_prefill_layer(
            input_sequence_length=8,
            name=config,
            tensor_parallel=5,
        )


def test_legacy_parallelism_enumerator_reuses_model_aware_rules():
    # Preserve the legacy API's [total_nodes/2, total_nodes) chip window.
    assert get_various_parallization(
        model="Qwen/Qwen2.5-7B-Instruct", total_nodes=1
    ) == {(1, 1)}
    assert get_various_parallization(
        model="Qwen/Qwen2.5-7B-Instruct", total_nodes=8
    ) == {
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (2, 2),
        (2, 3),
        (4, 1),
    }
