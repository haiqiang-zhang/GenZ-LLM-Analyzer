from GenZ.unit import Unit
import GenZ.operators as operators
from GenZ.operator_base import op_type_dicts
from GenZ.system import System
import pandas as pd
import numpy as np
import os
from GenZ.Models import OpType, ResidencyInfo

from GenZ.LLM_inference.utils import RuntimeBreakdown

def get_attn_index(df:pd.DataFrame):
    ret = []
    for idx in range(len(df)):
        if 'Attend' in df.loc[idx, 'Op Type'] or 'Logit' in df.loc[idx, 'Op Type']:
            ret.append(idx)
    return ret

def get_mamba_index(df:pd.DataFrame):
    ret = []
    for idx in range(len(df)):
        if 'CONV1D' in df.loc[idx, 'Op Type'] or 'x calc' in df.loc[idx, 'Layer Name']:
            ret.append(idx)
    return ret


def get_summary_table(df:pd.DataFrame, unit = Unit(), model_characterstics:bool=False):

    attn_idx = get_attn_index(df)
    mamba_idx = get_mamba_index(df)

    total_macs = 0
    total_data = 0
    kv_cache = 0
    total_weights = 0
    unused_weights = 0
    total_latencies = 0
    total_cycles = 0
    total_attn_latencies = 0
    total_linear_latencies = 0
    total_comm_latencies = 0

    multiplier = 1
    for i in range(len(df)):
        if df.loc[i,'Op Type'] == 'Repeat':
            multiplier *= df.loc[i,'Dimension']
        elif df.loc[i,'Op Type'] == 'EndRepeat':
            multiplier /= df.loc[i,'Dimension']
        else:
            total_macs += df.loc[i,f'Num ops ({unit.unit_flop})'] * multiplier
            total_data += (df.loc[i,f'Input_a ({unit.unit_mem})'] + df.loc[i,f'Input_w ({unit.unit_mem})'] + df.loc[i,f'Output ({unit.unit_mem})']) * multiplier
            if i in attn_idx:
                kv_cache += df.loc[i,f'Input_w ({unit.unit_mem})'] * multiplier
            elif i in mamba_idx:
                if 'CONV1D' in df.loc[i, 'Op Type']:
                    kv_cache += df.loc[i,f'Input_w ({unit.unit_mem})'] * multiplier
                elif 'x calc' in df.loc[i, 'Layer Name']:
                    kv_cache += df.loc[i,f'Input_w ({unit.unit_mem})'] * multiplier / df.loc[i,'Dimension'][0][0]
            elif 'GEMM' in df.loc[i, 'Op Type']:
                total_weights += df.loc[i,f'Input_w ({unit.unit_mem})'] * multiplier
                if df.loc[i, f'Num ops ({unit.unit_flop})'] == 0:
                    unused_weights += df.loc[i,f'Input_w ({unit.unit_mem})'] * multiplier

            if model_characterstics == False:
                total_latencies += df.loc[i,f'Latency ({unit.unit_time})'] * multiplier
                total_cycles += df.loc[i,'Cycles'] * multiplier
                if i in attn_idx:
                    total_attn_latencies += df.loc[i,f'Latency ({unit.unit_time})'] * multiplier
                elif 'GEMM' in df.loc[i, 'Op Type']:
                    total_linear_latencies += df.loc[i,f'Latency ({unit.unit_time})'] * multiplier
                elif 'Sync' in df.loc[i, 'Op Type']:
                    total_comm_latencies += df.loc[i,f'Latency ({unit.unit_time})'] * multiplier

    max_memory_footprint = max([df.loc[i, f'Input_a ({unit.unit_mem})'] + df.loc[i, f'Input_w ({unit.unit_mem})'] + df.loc[i, f'Output ({unit.unit_mem})'] for i in range(len(df))])


    ret = {
            f'MACs ({unit.unit_flop})': [total_macs],
            f'Total Data ({unit.unit_mem})': [total_data],
            f'Total Weights ({unit.unit_mem})': [total_weights],
            f'Unused Weights ({unit.unit_mem})': [unused_weights],
            f'KV Cache ({unit.unit_mem})': [kv_cache],
            f'On-chip Memory Footprint ({unit.unit_mem})': [max_memory_footprint],
        }
    if model_characterstics == False:
        ret.update({
            f'Latency ({unit.unit_time})': [total_latencies],
            'Cycles': [total_cycles],
            f'Attn Latency ({unit.unit_time})': [total_attn_latencies],
            f'Linear Latency ({unit.unit_time})': [total_linear_latencies],
            f'Comm Latency ({unit.unit_time})': [total_comm_latencies]
        })


    return pd.DataFrame.from_dict(ret)

def simplify_df(df:pd.DataFrame):
    unit = Unit()
    column_to_update = [f'Latency ({unit.unit_time})',
                f'Compute time ({unit.unit_time})', f'Memory time ({unit.unit_time})', f'Communication time ({unit.unit_time})',
                f'Cycles', f'Compute cycle', f'Memory cycle', f'Communication cycle',
                f'Num ops ({unit.unit_flop})', f'Input_a ({unit.unit_mem})', f'Input_w ({unit.unit_mem})', f'Output ({unit.unit_mem})', f'Total Data ({unit.unit_mem})',
                ]
    column_change = [col for col in df.columns if col in column_to_update]
    column_no_change = df.columns.difference(column_to_update).tolist()

    multiplier = 1
    new_df = pd.DataFrame(columns=df.columns)
    for i in range(len(df)):
        if df.loc[i,'Op Type'] == 'Repeat':
            multiplier *= df.loc[i,'Dimension']
        elif df.loc[i,'Op Type'] == 'EndRepeat':
            multiplier /= df.loc[i,'Dimension']
        else:
            new_row = df.loc[i, column_no_change].copy()
            for col in column_change:
                new_row[col] = df.loc[i, col] * multiplier
            if len(new_df) == 0:
                new_df = pd.DataFrame([new_row])
            else:
                new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    return new_df


def _pipeline_stage_frames(
    df: pd.DataFrame,
    pipeline_parallel: int,
) -> tuple[pd.DataFrame, ...]:
    """Split an explicit PP graph into physical-rank dataframes.

    The graph boundary is the one logical ``Message Pass`` emitted after each
    non-last rank.  Repeat scopes must be wholly contained by one rank; a
    boundary inside a repeat would make both latency and resident-memory
    attribution ambiguous and therefore fails closed.
    """
    if (
        isinstance(pipeline_parallel, bool)
        or not isinstance(pipeline_parallel, int)
        or pipeline_parallel < 1
    ):
        raise ValueError("pipeline_parallel must be a positive integer")
    required = {"Layer Name", "Op Type", "Dimension"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"pipeline graph dataframe missing columns: {sorted(missing)}"
        )

    stage_start = 0
    repeat_stack: list[int] = []
    stage_frames: list[pd.DataFrame] = []
    for position in range(len(df)):
        row = df.iloc[position]
        op_type = row["Op Type"]
        if op_type == "Repeat":
            repeat = row["Dimension"]
            if isinstance(repeat, bool):
                raise ValueError("repeat count must be a positive integer")
            try:
                normalized = int(repeat)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("repeat count must be a positive integer") from exc
            if normalized < 1 or repeat != normalized:
                raise ValueError("repeat count must be a positive integer")
            repeat_stack.append(normalized)
            continue
        if op_type == "EndRepeat":
            if not repeat_stack:
                raise ValueError("unmatched EndRepeat in pipeline graph")
            repeat = repeat_stack.pop()
            if row["Dimension"] != repeat:
                raise ValueError("mismatched Repeat/EndRepeat in pipeline graph")
            continue
        if row["Layer Name"] == "Message Pass":
            if repeat_stack:
                raise ValueError(
                    "pipeline Message Pass appears inside a Repeat scope; "
                    "physical PP stage attribution is ambiguous"
                )
            stage_frames.append(
                df.iloc[stage_start : position + 1].reset_index(drop=True)
            )
            stage_start = position + 1

    if repeat_stack:
        raise ValueError("unclosed Repeat in pipeline graph")
    stage_frames.append(df.iloc[stage_start:].reset_index(drop=True))
    if len(stage_frames) != pipeline_parallel:
        raise ValueError(
            f"pipeline graph contains {len(stage_frames)} physical stages; "
            f"expected pipeline_parallel={pipeline_parallel}"
        )
    if any(frame.empty for frame in stage_frames):
        raise ValueError("every physical PP stage must contain graph operators")
    return tuple(stage_frames)


def get_pipeline_stage_latencies(
    df: pd.DataFrame,
    pipeline_parallel: int,
    unit=None,
) -> tuple[float, ...]:
    """Decompose one PP traversal into per-rank service times.

    ``create_full_*_model`` emits one logical ``Message Pass`` after every
    non-last PP rank.  The message is charged to its sending rank.  Repeat
    markers for transformer layers are expanded locally, and a repeat scope
    crossing a PP boundary fails closed because such a graph cannot identify
    a physical rank bottleneck unambiguously.

    The sum is one scheduler batch's end-to-end traversal latency.  Under
    vLLM's saturated ``batch_queue`` (up to ``pipeline_parallel`` scheduler
    batches in flight), the steady-state interdeparture service is the maximum
    rank latency, not this sum.
    """
    unit = Unit() if unit is None else unit
    latency_column = f"Latency ({unit.unit_time})"
    if latency_column not in df.columns:
        raise ValueError(
            f"pipeline stage latency dataframe missing column: {latency_column!r}"
        )
    stage_latencies = []
    for frame in _pipeline_stage_frames(df, pipeline_parallel):
        multiplier = 1
        current_stage_latency = 0.0
        for position in range(len(frame)):
            row = frame.iloc[position]
            op_type = row["Op Type"]
            if op_type == "Repeat":
                multiplier *= int(row["Dimension"])
                continue
            if op_type == "EndRepeat":
                multiplier //= int(row["Dimension"])
                continue
            try:
                contribution = float(row[latency_column]) * multiplier
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("pipeline operator latency must be numeric") from exc
            if not np.isfinite(contribution) or contribution < 0.0:
                raise ValueError(
                    "pipeline operator latency must be finite/non-negative"
                )
            current_stage_latency += contribution
        stage_latencies.append(current_stage_latency)

    if any(latency <= 0.0 for latency in stage_latencies):
        raise ValueError("every physical PP stage must have positive latency")
    return tuple(stage_latencies)


def get_pipeline_stage_memory_requirements(
    df: pd.DataFrame,
    pipeline_parallel: int,
    unit=None,
) -> tuple[float, ...]:
    """Return each PP rank's resident weights + KV cache requirement.

    This preserves GenZ's existing fit/offload memory definition while making
    its physical ownership exact.  Uneven vLLM layer partitions and structural
    ``VLLM_PP_LAYER_PARTITION`` overrides must be checked against the largest
    rank footprint; dividing aggregate memory by PP can admit a rank that does
    not fit.  No fitted coefficient or topology residual is involved.
    """
    unit = Unit() if unit is None else unit
    weights_column = f"Total Weights ({unit.unit_mem})"
    kv_column = f"KV Cache ({unit.unit_mem})"
    requirements: list[float] = []
    for frame in _pipeline_stage_frames(df, pipeline_parallel):
        summary = get_summary_table(frame, unit, model_characterstics=True)
        try:
            requirement = float(summary[weights_column].values[0]) + float(
                summary[kv_column].values[0]
            )
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            raise ValueError(
                "pipeline stage resident memory must contain numeric weights "
                "and KV cache"
            ) from exc
        if not np.isfinite(requirement) or requirement < 0.0:
            raise ValueError(
                "pipeline stage resident memory must be finite/non-negative"
            )
        requirements.append(requirement)
    return tuple(requirements)

def _add_runtime_breakdown_layer(runtime_breakdown, layer_name, layer_latency):
    """Accumulate one already-repeat-scaled operator latency."""
    if layer_name in ['embeddings', 'classifier']:
        runtime_breakdown.Embedding += layer_latency
    elif layer_name in ['QKV', 'Out Proj']:
        runtime_breakdown.MHA += layer_latency
        runtime_breakdown.QKVO_layers += layer_latency
    elif layer_name in ['Logit', 'Attend', 'Logit Pre', 'Logit Suf',
                        'Attend Pre', 'Attend Suf', 'Logit Dec', 'Attend Dec']:
        runtime_breakdown.MHA += layer_latency
        runtime_breakdown.LA_layers += layer_latency
    elif layer_name in ['Gate', 'up+gate', 'down', 'shared up+gate', 'shared down']:
        runtime_breakdown.FFN += layer_latency
        runtime_breakdown.FFN_layers += layer_latency
    elif layer_name in ['Message Pass']:
        runtime_breakdown.Collective += layer_latency
        runtime_breakdown.Send_Recv_time += layer_latency
    elif layer_name in ['MHA AR', 'Mamba AR']:
        runtime_breakdown.MHA += layer_latency
        runtime_breakdown.Collective += layer_latency
        runtime_breakdown.AR_time += layer_latency
    elif layer_name in ['Gate AR', 'FFN AR']:
        runtime_breakdown.FFN += layer_latency
        runtime_breakdown.Collective += layer_latency
        runtime_breakdown.AR_time += layer_latency
    elif layer_name in ['Dispatch A2A', 'Collect A2A']:
        runtime_breakdown.Collective += layer_latency
        runtime_breakdown.A2A_time += layer_latency
        runtime_breakdown.FFN += layer_latency
    elif layer_name in ['Emb_AR', 'classifier_AG']:
        runtime_breakdown.AR_time += layer_latency
        runtime_breakdown.Embedding += layer_latency
        runtime_breakdown.Collective += layer_latency
    elif layer_name in ['Inproj', 'Conv', 'BC proj', 'xt proj', 'deltaA', 'deltaB', 'deltaBu', 'x calc', 'y calc', 'D addition', 'out mult z', 'Out proj', 'Mamba AR']:
        runtime_breakdown.Mamba_time += layer_latency
    else:
        raise ValueError(f'Layer Name:{layer_name} not found in the breakdown function')


def get_runtime_breakdown(df:pd.DataFrame) -> RuntimeBreakdown:
    """Return the runtime diagnostic without materializing an expanded frame.

    ``simplify_df`` used to copy every operator row into a new pandas frame,
    multiplying all numeric columns while doing so.  RuntimeBreakdown consumes
    only the latency column.  Apply the same nested Repeat/EndRepeat multiplier
    directly in the original row order: this preserves the exact arithmetic
    and public result while avoiding the dominant cost in scalar GenZ calls.
    """
    unit = Unit()
    runtime_breakdown = RuntimeBreakdown()
    assert 'Layer Name' in df.columns, "Layer Name not found in the dataframe"
    assert 'Op Type' in df.columns, "Op Type not found in the dataframe"
    assert 'Dimension' in df.columns, "Dimension not found in the dataframe"
    latency_column = f'Latency ({unit.unit_time})'
    assert latency_column in df.columns, "Latency (ms) not found in the dataframe"

    multiplier = 1
    for i in range(len(df)):
        op_type = df.loc[i, 'Op Type']
        if op_type == 'Repeat':
            multiplier *= df.loc[i, 'Dimension']
            continue
        if op_type == 'EndRepeat':
            multiplier /= df.loc[i, 'Dimension']
            continue
        _add_runtime_breakdown_layer(
            runtime_breakdown,
            df.loc[i, 'Layer Name'],
            df.loc[i, latency_column] * multiplier,
        )

    return runtime_breakdown


def analysis_model(model_dims, system=None, unit=None, densities = None,intermediate_on_chip=False,
                    beam_size=1, beam_merge=False, model_characterstics=False):
    # THREAD SAFETY (07-06): mutable default arguments (`unit=Unit()`) are
    # evaluated ONCE and shared by every call that omits them — a latent
    # cross-thread coupling. Construct per call instead.
    if unit is None:
        unit = Unit()
    roofline_list = []
    if densities is None:
        densities = np.ones((len(model_dims), 3), dtype=float)
    for i, (dim, density) in enumerate(zip(model_dims, densities)):

        op_type = op_type_dicts[dim[-1]]
        operators_residency = dim[-2]
        operator = getattr(operators, op_type)
        if beam_merge and (dim[-1] == OpType.Logit_BM_PREFILL or dim[-1] == OpType.Attend_BM_PREFILL):
            dim[1] /= beam_size         ## Batch size is divided by beam size
        operator_instance = operator(dim=dim, density=density)
        # print(density[0],density[1],density[2])
        if (intermediate_on_chip):
            if(op_type == 'Logit'):
                operator_instance.set_mem_pin(output='on')
            elif(op_type == 'Attend'):
                operator_instance.set_mem_pin(input_a='on')

        if operators_residency == ResidencyInfo.A_onchip:
            operator_instance.set_mem_pin(input_a='on')
        elif operators_residency == ResidencyInfo.B_onchip:
            operator_instance.set_mem_pin(input_b='on')
        elif operators_residency == ResidencyInfo.C_onchip:
            operator_instance.set_mem_pin(output='on')
        elif operators_residency == ResidencyInfo.AB_onchip:
            operator_instance.set_mem_pin(input_a='on')
            operator_instance.set_mem_pin(input_b='on')
        elif operators_residency == ResidencyInfo.AC_onchip:
            operator_instance.set_mem_pin(input_a='on')
            operator_instance.set_mem_pin(output='on')
        elif operators_residency == ResidencyInfo.BC_onchip:
            operator_instance.set_mem_pin(input_b='on')
            operator_instance.set_mem_pin(output='on')
        elif operators_residency == ResidencyInfo.All_onchip:
            operator_instance.set_mem_pin(input_a='on')
            operator_instance.set_mem_pin(input_b='on')
            operator_instance.set_mem_pin(output='on')

        if model_characterstics:
            roofline = operator_instance.get_model_characterstics(system=system, unit=unit)
        else:
            roofline = operator_instance.get_roofline(system=system, unit=unit)

        if i==0:
            column = roofline.keys()
        roofline_list.append([roofline[c] for c in column])

    df = pd.DataFrame(np.array(roofline_list,dtype=object), columns=column, dtype=object)

    return df


def get_model_df(model, system=None, unit=None, batch_size=1, data_path="/tmp/genz/data", intermediate_on_chip=False,
                    beam_size=1, beam_merge=False, model_characterstics=False):
    # THREAD SAFETY (07-06): the old `system=System(), unit=Unit()` defaults
    # were evaluated once at import and SHARED across every call that omitted
    # them — module-level mutable state coupling unrelated concurrent calls.
    # Construct per call; explicit arguments behave exactly as before.
    if system is None:
        system = System()
    if unit is None:
        unit = Unit()
    m_file_path = os.path.join(data_path,"model")
    sparsity_file_path = os.path.join(data_path,"sparsity")
    m_file = os.path.join(m_file_path, model)
    density_file = os.path.join(sparsity_file_path, model)
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    model_defs = np.insert(model_defs, 1, batch_size, axis=1)
    # model_defs = np.append(batch_sizes, model_defs, axis=1)
    def verify_repeat_pairs(model_defs):
        pairs = []
        stack = []
        for idx, row in enumerate(model_defs):
            Repeat_id = row[2]
            if row[-1] == OpType.REPEAT:
                stack.append((idx, Repeat_id))
            elif row[-1] == OpType.ENDREPEAT:
                if stack and stack[-1][1] == Repeat_id:
                    start_idx, _ = stack.pop()
                    pairs.append((start_idx, idx))
                else:
                    raise ValueError(f"Unmatched Endrepeat found or ID mismatch:{Repeat_id}")

        if stack:
            raise ValueError(f"Unmatched Repeat found: {stack[-1]}")

        return pairs
    pairs = verify_repeat_pairs(model_defs)

    new_model_defs = []
    for layer in model_defs:
            if isinstance(layer, np.ndarray):
                new_layer = [int(x) if isinstance(x, str) and x.isdigit() else float(x) if isinstance(x, str) and x.replace('.', '', 1).isdigit() else x for x in layer]
            new_model_defs.append(new_layer)

    densities = np.ones((len(model_defs), 3), dtype=float)

    return analysis_model(new_model_defs, system, unit, densities, intermediate_on_chip, beam_size, beam_merge, model_characterstics)
