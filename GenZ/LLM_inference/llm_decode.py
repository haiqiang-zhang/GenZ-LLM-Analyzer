from .utils import ModdelingOutput, get_inference_system, get_offload_system
from GenZ.unit import Unit
from GenZ.operators import *

from GenZ.analyse_model import *
import warnings
from GenZ.collective_times import *
from GenZ.utils.plot_rooflines import *
from GenZ.Models import create_full_decode_model, remove_layer_file
from math import ceil

unit = Unit()

def decode_moddeling(model = 'BERT', batch_size = 1, input_tokens = 4096,
    output_tokens = 0,   Bb = 4 ,           ## Only for Decode
    system_name = 'A100_40GB_GPU', system_eff = None, bits=None, debug= False, model_profilling = False,
    tensor_parallel = 1, pipeline_parallel = 1,
    expert_parallel = 1,
    collective_strategy=None, network_config=None,
    parallelism_hierarchy=None,
    model_offload = False, ceff = None, meff = None,
    pipeline_layer_partition = None):

    # vLLM pipeline parallelism forwards one complete scheduler-step tensor
    # through every layer partition.  Resident sequences are not split into
    # one micro-batch per PP rank.  The model graph below already contains the
    # full layer traversal plus PP message passes, so price the full resident
    # batch at every stage.
    ub = batch_size

    ##################################################################################################
    ### System Declaration
    ##################################################################################################

    # If caller passes explicit ceff/meff, they independently override
    # the single-scalar ``system_eff`` for compute and memory efficiency.
    _ceff = ceff if ceff is not None else system_eff
    _meff = meff if meff is not None else system_eff
    system = get_inference_system(system_name = system_name, bits = bits, ceff=_ceff, meff=_meff,
                                network_config=network_config,
                                collective_strategy=collective_strategy,
                                parallelism_hierarchy=parallelism_hierarchy )

    ##################################################################################################
    ### Model Characterization Calculation
    ##################################################################################################
    # if is_moe:
    model_decode = create_full_decode_model(name=model,
                                            input_sequence_length=input_tokens,
                                            output_gen_tokens = output_tokens ,
                                            tensor_parallel=tensor_parallel,
                                            pipeline_parallel=pipeline_parallel,
                                            expert_parallel=expert_parallel,
                                            pipeline_layer_partition=pipeline_layer_partition)

    model_df = get_model_df(model_decode, system=system, batch_size= ub*Bb, intermediate_on_chip=True , beam_merge= (Bb > 1), beam_size= Bb, model_characterstics = True)
    summary_table = get_summary_table(model_df, unit, model_characterstics = True)

    pipeline_stage_memory_requirements = get_pipeline_stage_memory_requirements(
        model_df, pipeline_parallel, unit,
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
        return model_df, summary_table

    ##################################################################################################
    ### Token generation time
    ##################################################################################################
    # model_decode = create_full_decode_model(name=model,
    #                                         input_sequence_length=input_tokens,
    #                                         output_gen_tokens = output_tokens ,
    #                                         tensor_parallel=tensor_parallel,
    #                                         pipeline_parallel=pipeline_parallel,
    #                                         expert_parallel=expert_parallel)

    model_df = get_model_df(model_decode, system, unit, ub*Bb,  intermediate_on_chip=True , beam_merge= (Bb > 1), beam_size= Bb)
    remove_layer_file(model_decode)   # temp CSV: final read done (07-06 leak fix)
    summary_table = get_summary_table(model_df, unit)

    if debug:
        display_df(simplify_df(model_df))
        display(summary_table)
    traversal_latency = summary_table[f'Latency ({unit.unit_time})'].values[0]  # Latency in msec
    pipeline_stage_latencies = get_pipeline_stage_latencies(
        model_df, pipeline_parallel, unit,
    )
    decode_latency = (
        traversal_latency
        if pipeline_parallel == 1
        else max(pipeline_stage_latencies)
    )

    ##################################################################################################
    ### Final Latency and Thrpt Calculation
    ##################################################################################################

    # vLLM keeps up to PP complete scheduler batches in ``batch_queue``.  At
    # saturation, interdeparture service is the slowest physical PP rank;
    # one request's full traversal remains available as a diagnostic below.
    thrpt = 1000 * batch_size / decode_latency


    linear_time = summary_table[f'Linear Latency ({unit.unit_time})'].values[0]                ## In milliseconds
    attn_time = summary_table[f'Attn Latency ({unit.unit_time})'].values[0]                    ## In milliseconds
    total_communication_delay = summary_table[f'Comm Latency ({unit.unit_time})'].values[0]    ## In milliseconds
    total_time = linear_time + attn_time + total_communication_delay
    # runtime_breakdown = [linear_time, attn_time, total_communication_delay]
    runtime_breakdown = get_runtime_breakdown(model_df)
    ##################################################################################################
    ### Output Generation
    ##################################################################################################

    return ModdelingOutput(
                        Latency=decode_latency,
                        SaturatedServiceLatency=decode_latency,
                        TraversalLatency=traversal_latency,
                        PipelineStageLatencies=pipeline_stage_latencies,
                        Throughput=thrpt,
                        Runtime_breakdown=runtime_breakdown,
                        is_offload=is_offloaded,
                        model_df = model_df,
                        summary_table = summary_table,
                )
