from .utils import ModdelingOutput, get_inference_system, get_offload_system
from GenZ.unit import Unit
from GenZ.operators import *

from GenZ.operator_base import op_type_dicts
from GenZ.system import System
import pandas as pd
from GenZ.analyse_model import *
import warnings
from GenZ.LLM_inference import decode_moddeling, prefill_moddeling
from paretoset import paretoset
from GenZ.parallelism import valid_parallelism_configs

unit = Unit()

def factors(n):
    return [x for tup in ([i, n//i]
                for i in range(1, int(n**0.5)+1) if n % i == 0) for x in tup]

def get_various_parallization(model='llama2_7b', total_nodes=8):
    if total_nodes < 1:
        raise ValueError(f'Num of Nodes:{total_nodes} should be >= 1')

    # Preserve this legacy API's chip-budget window (at least half the
    # available nodes, but strictly fewer than ``total_nodes``) while using
    # the same model-aware feasibility rules as every other GenZ caller.
    checks = valid_parallelism_configs(
        model,
        total_chips=total_nodes,
        exact_chips=False,
    )
    if total_nodes == 1:
        return {check.parallelism for check in checks}
    return {
        check.parallelism
        for check in checks
        if total_nodes // 2 <= np.prod(check.parallelism) < total_nodes
    }

def get_best_parallization_strategy(
        stage='decode', model='llama2_7b', total_nodes=8, batch_size = 1, beam_size = 1,
        input_tokens = 2000, output_tokens = 256,
        system_name = {'Flops': 200, 'Memory_size': 32, 'Memory_BW': 1000, 'ICN': 300 , 'real_values':True},
        bits='bf16', debug=False
        ):

    parallelism_combinations = get_various_parallization(model=model, total_nodes=total_nodes)
    if debug:
        print(f'For model:{model}, number cores:{total_nodes}, system:{system_name}, \n The parallelism combinations are {parallelism_combinations} ')

    data = []
    for TP,PP in parallelism_combinations:
        if stage == 'prefill':
            prefill_outputs = prefill_moddeling(model = model, batch_size = batch_size,
                                    input_tokens = input_tokens,
                                    system_name = system_name,
                                    bits=bits,
                                    tensor_parallel = TP, pipeline_parallel = PP, debug=debug)
            data.append([batch_size, TP, PP , prefill_outputs['Latency'], prefill_outputs['Throughput']])
        elif stage == 'decode':
            decode_outputs = decode_moddeling(model = model, batch_size = batch_size, Bb = beam_size ,
                                input_tokens = input_tokens, output_tokens = output_tokens,
                                system_name = system_name,
                                bits=bits,
                                tensor_parallel = TP, pipeline_parallel =PP, debug=debug)
            data.append([batch_size, TP, PP,  decode_outputs['Latency'], decode_outputs['Throughput']])
        else:
            raise ValueError('Stage should be prefill or decode')

    data_df = pd.DataFrame(data, columns = ['resident batch', 'TP', 'PP', 'Latency(ms)', 'Tokens/s'])
    if debug:
        display(data_df)
    return data_df.sort_values(by='Tokens/s', ascending=False).head(1)

def get_pareto_optimal_performance(
        stage='decode', model='llama2_7b', total_nodes=8, batch_list = 1, beam_size = 1,
        input_tokens = 2000, output_tokens = 256,
        system_name = {'Flops': 200, 'Memory_size': 32, 'Memory_BW': 1000, 'ICN': 300 , 'real_values':True},
        bits='bf16', debug=False
        ):

    parallelism_combinations = get_various_parallization(model=model, total_nodes=total_nodes)

    if debug:
        print(f'For model:{model}, number cores:{total_nodes}, system:{system_name}, \n The parallelism combinatations are {parallelism_combinations} ')
    data = []
    if isinstance(batch_list, int):
        batch_list = [batch_list]
    for batch_size in batch_list:
        for TP,PP in parallelism_combinations:
            if stage == 'prefill':
                prefill_outputs = prefill_moddeling(model = model, batch_size = batch_size,
                                        input_tokens = input_tokens,
                                        system_name = system_name,
                                        bits=bits,
                                        tensor_parallel = TP, pipeline_parallel = PP, debug=False)
                data.append([batch_size, TP, PP , prefill_outputs['Latency'], prefill_outputs['Throughput']])
            elif stage == 'decode':
                decode_outputs = decode_moddeling(model = model, batch_size = batch_size, Bb = beam_size ,
                                    input_tokens = input_tokens, output_tokens = output_tokens,
                                    system_name = system_name,
                                    bits=bits,
                                    tensor_parallel = TP, pipeline_parallel =PP, debug=False)
                data.append([batch_size, TP, PP,  decode_outputs['Latency'], decode_outputs['Throughput']])
            else:
                raise ValueError('Stage should be prefill or decode')

    data_df = pd.DataFrame(data, columns = ['batch', 'TP', 'PP', 'Latency(ms)', 'Tokens/s'])
    datapoints = data_df[['Latency(ms)','Tokens/s']]

    ##  We want a pareto optimal frontier with minimum latency and maximum Throughput.
    mask = paretoset(datapoints, sense=["min", "max"])
    return data_df[mask]
