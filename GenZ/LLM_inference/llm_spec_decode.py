from .utils import (
    ModdelingOutput,
    RuntimeBreakdown,
    get_inference_system,
    get_offload_system,
)
from GenZ.unit import Unit
from GenZ.operators import *

from GenZ.analyse_model import *
import warnings
from GenZ.collective_times import *
from GenZ.utils.plot_rooflines import *
from GenZ.Models import (
    create_full_chunked_model,
    create_full_prefill_model,
    create_full_decode_model,
    remove_layer_file,
)
import numpy as np
import pandas as pd

unit = Unit()


def _weighted_pipeline_rank_sums(*terms) -> tuple[float, ...]:
    """Combine deterministic per-rank work from sequential model passes."""
    if not terms:
        raise ValueError("at least one pipeline rank vector is required")
    expected_width = len(terms[0][0])
    if expected_width < 1:
        raise ValueError("pipeline rank vectors must not be empty")
    totals = [0.0] * expected_width
    for values, repetitions in terms:
        if len(values) != expected_width:
            raise ValueError("pipeline rank vectors must have equal width")
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or repetitions < 1
        ):
            raise ValueError("pipeline pass repetitions must be a positive integer")
        for rank, value in enumerate(values):
            contribution = float(value) * repetitions
            if not np.isfinite(contribution) or contribution < 0.0:
                raise ValueError(
                    "pipeline rank contribution must be finite/non-negative"
                )
            totals[rank] += contribution
    return tuple(totals)


def _weighted_runtime_breakdown(*terms) -> RuntimeBreakdown:
    """Combine traversal diagnostics for deterministic sequential passes."""
    combined = RuntimeBreakdown()
    fields = tuple(vars(combined))
    for breakdown, repetitions in terms:
        if (
            isinstance(repetitions, bool)
            or not isinstance(repetitions, int)
            or repetitions < 1
        ):
            raise ValueError("runtime pass repetitions must be a positive integer")
        values = breakdown.to_dict()
        if set(values) != set(fields):
            raise ValueError("runtime breakdown fields do not match")
        for field in fields:
            contribution = float(values[field]) * repetitions
            if not np.isfinite(contribution) or contribution < 0.0:
                raise ValueError(
                    "runtime breakdown contribution must be finite/non-negative"
                )
            setattr(combined, field, getattr(combined, field) + contribution)
    return combined


def spec_prefill_modeling(model = 'meta-llama/Llama-3.1-70B', draft_model = 'meta-llama/meta-llama-3.1-8b',
    batch_size = 1, 
    input_tokens = 1024,       # Input context tokens
    system_name = 'A100_40GB_GPU', system_eff = None, bits=None, debug= False, model_profilling = False,
    tensor_parallel = 1, pipeline_parallel = 1,
    expert_parallel = 1,
    collective_strategy=None, network_config=None,
    parallelism_hierarchy=None,
    model_offload = False, ceff = None, meff = None):

    # Match the vLLM scheduler-step boundary: PP ranks forward one complete
    # resident batch through the layer partitions, not per-rank micro-batches.
    ub = batch_size
    ##################################################################################################
    ### System Declaration
    ##################################################################################################

    _ceff = ceff if ceff is not None else system_eff
    _meff = meff if meff is not None else system_eff
    system = get_inference_system(system_name = system_name, bits = bits, ceff=_ceff, meff=_meff,
                                network_config=network_config,
                                collective_strategy=collective_strategy,
                                parallelism_hierarchy=parallelism_hierarchy )

    ##################################################################################################
    ### Model Characterization Calculation
    ##################################################################################################
    model_full_created = create_full_prefill_model(name=model,
                                            input_sequence_length= input_tokens,
                                            tensor_parallel = tensor_parallel,
                                            pipeline_parallel = pipeline_parallel,
                                            expert_parallel=expert_parallel)

    full_model_df = get_model_df(model_full_created, system=system, batch_size = ub, intermediate_on_chip=True , beam_merge= True, beam_size= 1, model_characterstics = True)
    remove_layer_file(model_full_created)
    full_summary_table = get_summary_table(full_model_df, unit, model_characterstics = True)
    full_stage_memory = get_pipeline_stage_memory_requirements(
        full_model_df, pipeline_parallel, unit,
    )
    
    # While draft model can be fit with a different parallelism, right now we are assuming the same parallelism for both draft and full model.
    model_draft_created = create_full_prefill_model(name=draft_model,
                                            input_sequence_length= input_tokens,
                                            tensor_parallel = tensor_parallel,
                                            pipeline_parallel = pipeline_parallel,
                                            expert_parallel=expert_parallel)

    draft_model_df = get_model_df(model_draft_created, system=system, batch_size = ub, intermediate_on_chip=True , beam_merge= True, beam_size= 1, model_characterstics = True)
    remove_layer_file(model_draft_created)
    draft_summary_table = get_summary_table(draft_model_df, unit, model_characterstics = True)
    draft_stage_memory = get_pipeline_stage_memory_requirements(
        draft_model_df, pipeline_parallel, unit,
    )
    
    pipeline_stage_memory_requirements = _weighted_pipeline_rank_sums(
        (full_stage_memory, 1),
        (draft_stage_memory, 1),
    )
    max_rank_memory_req = max(pipeline_stage_memory_requirements)

    num_nodes = pipeline_parallel * tensor_parallel * expert_parallel

    #################################################################################
    ### Offloading calculations
    #################################################################################
    is_offloaded = False
    per_chip_memory = system.get_off_chip_mem_size()   ## MB
    if per_chip_memory < max_rank_memory_req:
        if model_offload:
            system = get_offload_system(
                system=system,
                total_memory_req=max_rank_memory_req,
                debug=debug,
            )
            warnings.warn(f"Some Parameter offloaded, effective Memory BW:{unit.raw_to_unit(system.offchip_mem_bw, type='BW')} ")
            is_offloaded = True
        elif model_profilling:
            warnings.warn(
                f"All params would not fit on the largest PP rank. System "
                f"Memory Cap:{per_chip_memory/1024} GB, Max Rank Resident "
                f"Memory:{max_rank_memory_req/1024} GB"
            )
        else:
            raise ValueError(
                f"All params would not fit on the largest PP rank. System "
                f"Memory Cap:{per_chip_memory/1024} GB, Max Rank Resident "
                f"Memory:{max_rank_memory_req/1024} GB.\n System:{system_name}"
            )

    ## for tensor shareding per layer.
    assert pipeline_parallel >= 1, "Pipeline parallel must be >= 1"
    assert tensor_parallel >= 1, f"Tensor parallel must be >= 1, {tensor_parallel}"

    if model_profilling:
        return pd.concat([full_model_df, draft_model_df]), pd.concat([full_summary_table, draft_summary_table])

    ##################################################################################################
    ### Initial prefill times
    ##################################################################################################
    model_prefill = create_full_prefill_model(  name=model,
                                            input_sequence_length=input_tokens,
                                            tensor_parallel=tensor_parallel,
                                            pipeline_parallel=pipeline_parallel,
                                            expert_parallel=expert_parallel)

    full_runtime_df = get_model_df(
        model_prefill, system, unit, ub, intermediate_on_chip=True,
    )
    remove_layer_file(model_prefill)
    full_runtime_summary = get_summary_table(full_runtime_df, unit)
    full_stage_latencies = get_pipeline_stage_latencies(
        full_runtime_df, pipeline_parallel, unit,
    )
    full_runtime_breakdown = get_runtime_breakdown(full_runtime_df)
    if debug:
        print("Full Model Prefill")
        display_df(simplify_df(full_runtime_df))
        display(full_runtime_summary)
    
    model_draft_prefill = create_full_prefill_model(  name=draft_model,
                                            input_sequence_length=input_tokens,
                                            tensor_parallel=tensor_parallel,
                                            pipeline_parallel=pipeline_parallel,
                                            expert_parallel=expert_parallel)

    draft_runtime_df = get_model_df(
        model_draft_prefill, system, unit, ub, intermediate_on_chip=True,
    )
    remove_layer_file(model_draft_prefill)
    draft_runtime_summary = get_summary_table(draft_runtime_df, unit)
    draft_stage_latencies = get_pipeline_stage_latencies(
        draft_runtime_df, pipeline_parallel, unit,
    )
    draft_runtime_breakdown = get_runtime_breakdown(draft_runtime_df)
    if debug:
        print("Draft Model Prefill")
        display_df(simplify_df(draft_runtime_df))
        display(draft_runtime_summary)
    
    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    pipeline_stage_latencies = _weighted_pipeline_rank_sums(
        (full_stage_latencies, 1),
        (draft_stage_latencies, 1),
    )
    traversal_latency = sum(pipeline_stage_latencies)
    prefill_latency = max(pipeline_stage_latencies)
    thrpt = 1000 * batch_size / prefill_latency
    summary_table = full_runtime_summary + draft_runtime_summary
    runtime_breakdown = _weighted_runtime_breakdown(
        (full_runtime_breakdown, 1),
        (draft_runtime_breakdown, 1),
    )
    ##################################################################################################
    ### Output Generation
    ##################################################################################################

    return ModdelingOutput(
                        Latency=prefill_latency,
                        SaturatedServiceLatency=prefill_latency,
                        TraversalLatency=traversal_latency,
                        PipelineStageLatencies=pipeline_stage_latencies,
                        Throughput=thrpt,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                        model_df = draft_runtime_df,
                        summary_table = summary_table,
                )
    
def spec_decode_modeling(model = 'meta-llama/Llama-3.1-70B', draft_model = 'meta-llama/meta-llama-3.1-8b',
    batch_size = 1, 
    input_tokens = 1024,       # Input context tokens
    output_tokens = 1024,      # Output context to be generated
    token_acceptance_rate = 0.7,            # Probability of accepting a token from draft model decoding.
    # This gamma parameter in the paper: arxiv.org/pdf/2211.17192
    num_parallel_tokens = 8,    # Number of tokens to be decoded in parallel by the full model.
                                # This means after num_parallel_tokens decode steps of the draft model, num_parallel_tokens tokens are checked in parallel by the full model.
    system_name = 'A100_40GB_GPU', system_eff = None, bits=None, debug= False, model_profilling = False,
    tensor_parallel = 1, pipeline_parallel = 1,
    expert_parallel = 1,
    collective_strategy=None, network_config=None,
    parallelism_hierarchy=None,
    model_offload = False, ceff = None, meff = None):

    # Match the vLLM scheduler-step boundary: PP ranks forward one complete
    # resident batch through the layer partitions, not per-rank micro-batches.
    ub = batch_size

    assert num_parallel_tokens > 1, "Number of parallel tokens must be > 1, for the full model to be useful"
    ##################################################################################################
    ### System Declaration
    ##################################################################################################

    _ceff = ceff if ceff is not None else system_eff
    _meff = meff if meff is not None else system_eff
    system = get_inference_system(system_name = system_name, bits = bits, ceff=_ceff, meff=_meff,
                                network_config=network_config,
                                collective_strategy=collective_strategy,
                                parallelism_hierarchy=parallelism_hierarchy )

    ##################################################################################################
    ### Model Characterization Calculation
    ##################################################################################################
    model_full_created = create_full_decode_model(name=model,
                                            input_sequence_length= input_tokens,
                                            output_gen_tokens = output_tokens,
                                            tensor_parallel = tensor_parallel,
                                            pipeline_parallel = pipeline_parallel,
                                            expert_parallel=expert_parallel)

    full_model_df = get_model_df(model_full_created, system=system, batch_size = ub, intermediate_on_chip=True , beam_merge= True, beam_size= 1, model_characterstics = True)
    remove_layer_file(model_full_created)
    full_summary_table = get_summary_table(full_model_df, unit, model_characterstics = True)
    full_stage_memory = get_pipeline_stage_memory_requirements(
        full_model_df, pipeline_parallel, unit,
    )
    
    # While draft model can be fit with a different parallelism, right now we are assuming the same parallelism for both draft and full model.
    model_draft_created = create_full_decode_model(name=draft_model,
                                            input_sequence_length= input_tokens,
                                            output_gen_tokens= output_tokens,
                                            tensor_parallel = tensor_parallel,
                                            pipeline_parallel = pipeline_parallel,
                                            expert_parallel=expert_parallel)

    draft_model_df = get_model_df(model_draft_created, system=system, batch_size = ub, intermediate_on_chip=True , beam_merge= True, beam_size= 1, model_characterstics = True)
    remove_layer_file(model_draft_created)
    draft_summary_table = get_summary_table(draft_model_df, unit, model_characterstics = True)
    draft_stage_memory = get_pipeline_stage_memory_requirements(
        draft_model_df, pipeline_parallel, unit,
    )
    
    pipeline_stage_memory_requirements = _weighted_pipeline_rank_sums(
        (full_stage_memory, 1),
        (draft_stage_memory, 1),
    )
    max_rank_memory_req = max(pipeline_stage_memory_requirements)

    num_nodes = pipeline_parallel * tensor_parallel * expert_parallel

    #################################################################################
    ### Offloading calculations
    #################################################################################
    is_offloaded = False
    per_chip_memory = system.get_off_chip_mem_size()   ## MB
    if per_chip_memory < max_rank_memory_req:
        if model_offload:
            system = get_offload_system(
                system=system,
                total_memory_req=max_rank_memory_req,
                debug=debug,
            )
            warnings.warn(f"Some Parameter offloaded, effective Memory BW:{unit.raw_to_unit(system.offchip_mem_bw, type='BW')} ")
            is_offloaded = True
        elif model_profilling:
            warnings.warn(
                f"All params would not fit on the largest PP rank. System "
                f"Memory Cap:{per_chip_memory/1024} GB, Max Rank Resident "
                f"Memory:{max_rank_memory_req/1024} GB"
            )
        else:
            raise ValueError(
                f"All params would not fit on the largest PP rank. System "
                f"Memory Cap:{per_chip_memory/1024} GB, Max Rank Resident "
                f"Memory:{max_rank_memory_req/1024} GB.\n System:{system_name}"
            )

    ## for tensor shareding per layer.
    assert pipeline_parallel >= 1, "Pipeline parallel must be >= 1"
    assert tensor_parallel >= 1, f"Tensor parallel must be >= 1, {tensor_parallel}"

    if model_profilling:
        return pd.concat([full_model_df, draft_model_df]), pd.concat([full_summary_table, draft_summary_table])

    ##################################################################################################
    ### Model decode times
    ##################################################################################################
    model_draft_decode = create_full_decode_model(  name=draft_model,
                                            input_sequence_length=input_tokens,
                                            output_gen_tokens = output_tokens ,
                                            tensor_parallel=tensor_parallel,
                                            pipeline_parallel=pipeline_parallel,
                                            expert_parallel=expert_parallel)

    draft_runtime_df = get_model_df(
        model_draft_decode, system, unit, ub, intermediate_on_chip=True,
    )
    remove_layer_file(model_draft_decode)
    draft_runtime_summary = get_summary_table(draft_runtime_df, unit)
    draft_stage_latencies = get_pipeline_stage_latencies(
        draft_runtime_df, pipeline_parallel, unit,
    )
    draft_runtime_breakdown = get_runtime_breakdown(draft_runtime_df)
    if debug:
        print("Draft Model Decode")
        display_df(simplify_df(draft_runtime_df))
        display(draft_runtime_summary)

    # For full model, the exsisting KV cache is input tokens + output tokens, and
    # we are checking num_parallel_tokens tokens in parallel.
    model_decode = create_full_chunked_model(  name=model,
                                            prefill_kv_sizes = [(input_tokens+output_tokens, num_parallel_tokens)],
                                            decode_kv_sizes = [] ,
                                            tensor_parallel=tensor_parallel,
                                            pipeline_parallel=pipeline_parallel,
                                            expert_parallel=expert_parallel)

    full_runtime_df = get_model_df(
        model_decode, system, unit, ub, intermediate_on_chip=True,
    )
    remove_layer_file(model_decode)
    full_runtime_summary = get_summary_table(full_runtime_df, unit)
    full_stage_latencies = get_pipeline_stage_latencies(
        full_runtime_df, pipeline_parallel, unit,
    )
    full_runtime_breakdown = get_runtime_breakdown(full_runtime_df)
    if debug:
        print("Full Model Decode")
        display_df(simplify_df(full_runtime_df))
        display(full_runtime_summary)
    
        
    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    # num_parallel_tokens = 4
    # token_acceptance_rate = 0.7
    # 4 draft token generated.
    # Chance of 1 token accepted = 0.7     , Chance of reject = 1-0.7
    # Chances of 2 token accepted = 0.7**2 , Chance of 2 token rejected = 1-0.7**2
    # Chances of 3 token accepted = 0.7**3 , Chance of 3 token rejected = 0.3*0.7**2
    # Chance of 4 token accepeted= 0.7**4 , Chance of 4 token rejected = 0.3*0.7**3
    # Latency = full-model verification + N sequential draft-model passes.
    # Number of tokens generated = 4*(x**4) + 3*(x**3) + 2*(x**2) + 1*(x**1) = 4*(1-x)**3

    pipeline_stage_latencies = _weighted_pipeline_rank_sums(
        (full_stage_latencies, 1),
        (draft_stage_latencies, num_parallel_tokens),
    )
    traversal_latency = sum(pipeline_stage_latencies)
    total_latency = max(pipeline_stage_latencies)
    tokens_generated = expected_tokens(num_parallel_tokens, token_acceptance_rate)
    thrpt = 1000 * batch_size * tokens_generated / total_latency

    summary_table = (
        full_runtime_summary
        + draft_runtime_summary * num_parallel_tokens
    )
    runtime_breakdown = _weighted_runtime_breakdown(
        (full_runtime_breakdown, 1),
        (draft_runtime_breakdown, num_parallel_tokens),
    )
    ##################################################################################################
    ### Output Generation
    ##################################################################################################

    return ModdelingOutput(
                        Latency=total_latency,
                        SaturatedServiceLatency=total_latency,
                        TraversalLatency=traversal_latency,
                        PipelineStageLatencies=pipeline_stage_latencies,
                        Throughput=thrpt,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                        model_df = full_runtime_df,
                        summary_table = summary_table,
                        tokens_generated=tokens_generated
                )
    
def expected_tokens(N: int, x: float) -> float:
    """
    Calculate expected number of accepted tokens in speculative decoding.
    
    Args:
        N (int): Number of tokens generated speculatively
        x (float): Probability of token acceptance (between 0 and 1)
    
    Returns:
        float: Expected number of accepted tokens
    """
    if not 0 <= x <= 1:
        raise ValueError("Probability x must be between 0 and 1")
    if N < 1:
        raise ValueError("N must be positive")
    
    # For k < N: k tokens accepted with prob x^k * (1-x)
    # For k = N: N tokens accepted with prob x^N
    
    # We start from 1 because, the full model will correct 1 token if it is wrong.
    expected = 1
    for k in range(1, N):
        expected += k * (x**k) * (1-x)
    expected += N * (x**N)
    
    return min(expected, N)
