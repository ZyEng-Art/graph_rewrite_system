from __future__ import annotations

import argparse
import ctypes
import ctypes.util
from pathlib import Path

_libgomp = ctypes.util.find_library("gomp")
if _libgomp:
    ctypes.CDLL(_libgomp, mode=ctypes.RTLD_GLOBAL)

import quartz


HEADER = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n'


def graph(context, body: str):
    return quartz.PyGraph.from_qasm_str(context=context, qasm_str=HEADER + body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecc-file", type=Path, required=True)
    args = parser.parse_args()
    context = quartz.QuartzContext(
        gate_set=["h", "x", "cx", "rz", "add"],
        filename=str(args.ecc_file),
        no_increase=False,
        include_nop=True,
    )
    independent_left = graph(context, "h q[0];\nx q[2];\n")
    independent_right = graph(context, "x q[2];\nh q[0];\n")
    dependent_left = graph(context, "h q[0];\nx q[0];\n")
    dependent_right = graph(context, "x q[0];\nh q[0];\n")
    wiring_left = graph(context, "cx q[0],q[1];\n")
    wiring_right = graph(context, "cx q[0],q[2];\n")
    parameter_left = graph(context, "rz(pi*0.25) q[0];\n")
    parameter_right = graph(context, "rz(pi*0.5) q[0];\n")

    assert isinstance(independent_left.exact_key(), bytes)
    assert independent_left.exact_key() == independent_right.exact_key()
    assert dependent_left.exact_key() != dependent_right.exact_key()
    assert wiring_left.exact_key() != wiring_right.exact_key()
    assert parameter_left.exact_key() != parameter_right.exact_key()
    print(
        "native exact graph key: independent-order invariant and "
        "order/wiring/parameter safe"
    )


if __name__ == "__main__":
    main()
