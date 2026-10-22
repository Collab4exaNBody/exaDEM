#!/usr/bin/env python3
"""Checks that the damprate: [...] values hardcoded in test3_restitution_sweep.msp
match exaDEM's own damp_rate <-> restitution coefficient formula
(scripts/tools/compute_damp_rate.py), for the target e values documented in
that file's header comment (e = 0.9, 0.7, 0.5, 0.3, in group 0..3 order).

Usage: python3 verify_damp_rate.py
Exits non-zero (with a message) if any value is inconsistent.
"""
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "tools"))
from compute_damp_rate import compute_damp_rate  # noqa: E402

TARGET_E = [0.9, 0.7, 0.5, 0.3]  # group 0, 1, 2, 3, in this order
MSP_FILE = Path(__file__).resolve().parent / "test3_restitution_sweep.msp"
TOLERANCE = 1e-5


def parsed_damprate(msp_text):
    m = re.search(r"damprate:\s*\[([^\]]*)\]", msp_text)
    if not m:
        raise ValueError(f"no 'damprate: [...]' line found in {MSP_FILE}")
    values_part = m.group(1).split("#", 1)[0]
    return [float(v) for v in values_part.split(",")]


def main():
    expected = [compute_damp_rate(e * e) for e in TARGET_E]
    actual = parsed_damprate(MSP_FILE.read_text())

    if len(actual) != len(expected):
        print(f"FAIL: {MSP_FILE.name} has {len(actual)} damprate values, expected {len(expected)}")
        return 1

    ok = True
    for i, (e, exp, act) in enumerate(zip(TARGET_E, expected, actual)):
        diff = abs(exp - act)
        status = "OK" if diff < TOLERANCE else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"group {i}: target e={e:<4} expected damp_rate={exp:.6f} "
              f"got={act:.6f} diff={diff:.2e} [{status}]")

    # Round-trip sanity check: damp_rate -> e via the exact inverse formula
    # e = exp(-zeta*pi / sqrt(1-zeta^2)) should recover the target e.
    for e, zeta in zip(TARGET_E, expected):
        e_back = math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta * zeta))
        diff = abs(e_back - e)
        status = "OK" if diff < TOLERANCE else "FAIL"
        if status == "FAIL":
            ok = False
        print(f"round-trip: zeta={zeta:.6f} -> e={e_back:.6f} (target {e}) diff={diff:.2e} [{status}]")

    if not ok:
        print("VERIFICATION FAILED")
        return 1
    print("All damp_rate values verified OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
