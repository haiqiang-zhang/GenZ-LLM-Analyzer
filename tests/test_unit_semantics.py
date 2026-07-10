from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_SPEC = spec_from_file_location("genz_unit", Path(__file__).parents[1] / "GenZ" / "unit.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
Unit = _MODULE.Unit


def test_bandwidth_uses_decimal_si_units():
    unit = Unit(unit_bw="GBsec")
    assert unit.unit_to_raw(1, type="BW") == 1_000_000_000


def test_memory_capacity_keeps_binary_units():
    unit = Unit(unit_mem="MB")
    assert unit.unit_to_raw(1, type="M") == 1_048_576
