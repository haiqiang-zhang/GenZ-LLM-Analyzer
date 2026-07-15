from collections import OrderedDict
from typing import Optional
from GenZ.unit import Unit
import warnings
from GenZ.system import System
import pandas as pd
from Systems.system_configs import system_configs

OFFLOAD_BW = 128

unit = Unit()

class RuntimeBreakdown():
    def __init__(self):
        # Blockwise
        self.Embedding: float = 0
        self.MHA: float = 0
        self.FFN: float = 0
        self.Collective: float = 0
        # Smaller Layers
        self.LA_layers: float = 0
        self.QKVO_layers: float = 0
        self.FFN_layers: float = 0
        # Others
        self.Softmax: float = 0
        self.AR_time: float = 0
        self.A2A_time: float = 0
        self.Send_Recv_time: float = 0
        self.Mamba_time: float = 0

    def __repr__(self):
        variables = vars(self)
        return ', '.join(f'{name}: {value}' for name, value in variables.items())

    def to_dict(self):
        return vars(self)

class ModdelingOutput(dict):
    # ``Latency`` is saturated scheduler-batch interdeparture service.  For
    # PP=1 it equals traversal latency; for PP>1 the full single-batch path and
    # its exact rank decomposition remain available separately.
    # ``Runtime_breakdown`` is accumulated over the full traversal path, not
    # bottleneck-rank ``Latency``, when PP>1.  Its diagnostic categories can
    # overlap and must not be blindly summed.
    Latency: float = 0
    SaturatedServiceLatency: float = 0
    TraversalLatency: float = 0
    PipelineStageLatencies: Optional[tuple[float, ...]] = None
    Throughput: float = 0
    Runtime_breakdown: Optional[RuntimeBreakdown] = None
    is_offload: Optional[bool] = False
    model_df: Optional[pd.DataFrame] = None
    summary_table: Optional[pd.DataFrame] = None
    tokens_generated: Optional[int] = 0

def get_offload_system(system, total_memory_req, debug):
    """Create a new system with offloaded memory connections

    Args:
        system (System): System Definition without offload
        total_memory_req (float): Memory required for the model weights and activations in MB
        debug (bool): Debug flag for printing

    Returns:
        System: System Definition with offload
    """
    total_device_memory = unit.raw_to_unit(system.off_chip_mem_size, type='M')/1024 ## GB
    total_memory_req = total_memory_req/1024 ## GB
    memory_offloaded = total_memory_req - total_device_memory

    offload_bw = OFFLOAD_BW if system.external_mem_bw == 0 else system.get_external_mem_bw() 
    if debug:
        print(f'Total Memory Req:{total_memory_req}GB, Total device mem:{total_device_memory}GB, Mem Offload:{memory_offloaded}GB @ {offload_bw} GB/s')

    ###############################################################################################
    ### Memory Time = max(min(size_required,size_hbm)/BW_hbm, (size_required-size_hbm)/BW_offload)
    ###############################################################################################

    if memory_offloaded > 0:
        new_offchip_BW = (total_memory_req ) / max(min(total_memory_req,total_device_memory)/unit.raw_to_unit(system.offchip_mem_bw, type='BW'), memory_offloaded/offload_bw) 
        system.set_offchip_mem_bw(new_offchip_BW)
        if debug:
            print(f'New BW:{new_offchip_BW}')
    return system

def get_inference_system(system_name='A100_40GB_GPU', bits=None, ceff=None, meff=None,
                        collective_strategy=None, network_config=None,
                        parallelism_hierarchy=None,
                         **kwargs):
    """Resolve ``system_name`` into a ``System`` object.

    For the ``bits`` / ``ceff`` / ``meff`` / ``collective_strategy`` /
    ``parallelism_hierarchy`` args, ``None`` means "caller did not set
    this" — when ``system_name`` is a pre-built ``System``, those fields
    are preserved (not overwritten with a default). When a fresh System
    is constructed from a string/dict, ``None`` resolves to the safe
    default (``bf16``, 1, 1, ``GenZ``, etc.).

    ``network_config`` keeps its legacy semantic where ``None`` is a
    meaningful value ("no network config"); to avoid clobbering
    pre-built System.network_config we still only overwrite when the
    existing field is absent / not set.
    """
    ##################################################################################################
    ### System Declaration
    ##################################################################################################
    if isinstance(system_name, str):
        if system_name in system_configs:
            system_name = system_configs[system_name]
        else:
            raise ValueError(f'System mentioned:{system_name} not present in predefined systems. Please use systems from Systems/system_configs')
    if isinstance(system_name, dict):
        if system_name.get('real_values',False) == True:
            NUM_FLOPS = system_name.get('Flops', 320)
            OFFCHIP_MEM_BW = system_name.get('Memory_BW',40)
            per_chip_memory = system_name.get('Memory_size',2000)
            C2C_BW = system_name.get('ICN',150)
            C2C_LL = system_name.get('ICN_LL',1)
    elif isinstance(system_name, System):
        # THREAD SAFETY (07-06): configure a shallow COPY, never the caller's
        # object. The previous in-place writes made every System-instance
        # caller share mutable state across concurrent modeling calls (two
        # threads pricing with different parallelism_hierarchy/ceff corrupt
        # each other mid-computation). The written fields are all
        # scalars/strings, so a shallow copy fully isolates them; heavier
        # attributes (unit, network_config objects) stay shared read-only.
        import copy as _copy
        system_copy = _copy.copy(system_name)
        if bits is not None:
            system_copy.bits = bits
        if ceff is not None:
            system_copy.compute_efficiency = ceff
        if meff is not None:
            system_copy.memory_efficiency = meff
        if collective_strategy is not None:
            system_copy.collective_strategy = collective_strategy
        if parallelism_hierarchy is not None:
            system_copy.parallelism_hierarchy = parallelism_hierarchy
        if network_config is not None:
            system_copy.network_config = network_config
        return system_copy
    else:
        raise TypeError(f'System should be weight str or dict with Flops,Memory, ICN values: System_name: {system_name}')

    # Fresh-System construction: resolve None sentinels to safe defaults.
    _bits = 'bf16' if bits is None else bits
    _ceff = 1 if ceff is None else ceff
    _meff = 1 if meff is None else meff
    _collective_strategy = 'GenZ' if collective_strategy is None else collective_strategy
    _parallelism_hierarchy = "TP{1}_EP{1}_PP{1}" if parallelism_hierarchy is None else parallelism_hierarchy

    return System(unit, frequency=1000, flops=NUM_FLOPS, off_chip_mem_size=(per_chip_memory*1024),
                    compute_efficiency=_ceff, memory_efficiency=_meff,
                    offchip_mem_bw=OFFCHIP_MEM_BW, bits=_bits, external_mem_bw=OFFLOAD_BW,
                    interchip_link_bw=C2C_BW, interchip_link_latency=C2C_LL,
                    collective_strategy=_collective_strategy, network_config=network_config,
                    parallelism_hierarchy=_parallelism_hierarchy)
