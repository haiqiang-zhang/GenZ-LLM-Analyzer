import pytest

from GenZ.LLM_inference import decode_moddeling, prefill_moddeling
from GenZ.LLM_inference.utils import RuntimeBreakdown
from GenZ import analyse_model
from GenZ.unit import Unit


def _legacy_runtime_breakdown(model_df):
    """The pre-fast-path implementation, retained as an exact oracle."""
    simplified = analyse_model.simplify_df(model_df)
    latency_column = f"Latency ({Unit().unit_time})"
    result = RuntimeBreakdown()
    for i in range(len(simplified)):
        analyse_model._add_runtime_breakdown_layer(
            result,
            simplified.loc[i, "Layer Name"],
            simplified.loc[i, latency_column],
        )
    return result


@pytest.mark.parametrize(
    "model,batch,input_tokens,output_tokens,tp,pp",
    [
        ("Qwen/Qwen2.5-1.5B", 1, 128, 32, 1, 1),
        ("Qwen/Qwen2.5-3B", 8, 512, 96, 2, 1),
        ("Qwen/Qwen2.5-7B", 64, 1024, 288, 2, 2),
        ("Qwen/Qwen2.5-14B", 128, 2048, 512, 4, 1),
    ],
)
def test_runtime_breakdown_fastpath_matches_expanded_frame_exactly(
    model, batch, input_tokens, output_tokens, tp, pp,
):
    output = decode_moddeling(
        model=model,
        batch_size=batch,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        Bb=1,
        system_name="A100_40GB_GPU",
        bits="bf16",
        tensor_parallel=tp,
        pipeline_parallel=pp,
        model_offload=True,
        debug=False,
    )

    expected = _legacy_runtime_breakdown(output["model_df"])
    actual = analyse_model.get_runtime_breakdown(output["model_df"])
    assert actual.to_dict() == expected.to_dict()
    assert output["Runtime_breakdown"].to_dict() == expected.to_dict()


@pytest.mark.parametrize(
    "model,batch,input_tokens,tp,pp",
    [
        ("Qwen/Qwen2.5-1.5B", 8, 512, 1, 1),
        ("Qwen/Qwen2.5-7B", 32, 2048, 2, 2),
    ],
)
def test_prefill_runtime_breakdown_fastpath_matches_expanded_frame_exactly(
    model, batch, input_tokens, tp, pp,
):
    output = prefill_moddeling(
        model=model,
        batch_size=batch,
        input_tokens=input_tokens,
        system_name="A100_40GB_GPU",
        bits="bf16",
        tensor_parallel=tp,
        pipeline_parallel=pp,
        model_offload=True,
        debug=False,
    )

    expected = _legacy_runtime_breakdown(output["model_df"])
    actual = analyse_model.get_runtime_breakdown(output["model_df"])
    assert actual.to_dict() == expected.to_dict()
    assert output["Runtime_breakdown"].to_dict() == expected.to_dict()


def test_runtime_breakdown_fastpath_does_not_expand_dataframe(monkeypatch):
    output = decode_moddeling(
        model="Qwen/Qwen2.5-1.5B",
        batch_size=2,
        input_tokens=256,
        output_tokens=64,
        Bb=1,
        system_name="A100_40GB_GPU",
        bits="bf16",
        model_offload=True,
        debug=False,
    )
    expected = _legacy_runtime_breakdown(output["model_df"]).to_dict()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("get_runtime_breakdown called simplify_df")

    monkeypatch.setattr(analyse_model, "simplify_df", fail_if_called)
    assert analyse_model.get_runtime_breakdown(output["model_df"]).to_dict() == expected
