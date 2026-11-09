#!/usr/bin/env python3
"""Check that the damprate values of test3_restitution_sweep.msp (hertz_lib.DAMP_RATES)
give the target restitution coefficients (hertz_lib.TARGET_E) with the damping of the
Hertz law, and that they are the ones written in the .msp file."""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hertz_lib import DAMP_RATES, TARGET_E, damp_rate_for_restitution  # noqa: E402

TOLERANCE = 1e-5


def main():
    ok = True
    msp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test3_restitution_sweep.msp")
    with open(msp) as f:
        line = next(l for l in f if l.strip().startswith("damprate:"))
    in_msp = [float(x) for x in re.search(r"\[(.*)\]", line).group(1).split(",")]
    for e, zeta, zeta_msp in zip(TARGET_E, DAMP_RATES, in_msp):
        z = damp_rate_for_restitution(e)
        good = abs(z - zeta) < TOLERANCE and zeta == zeta_msp
        ok = ok and good
        print("e = %.2f: damp_rate = %.7f (hertz_lib %.7f, msp %.7f) %s" % (e, z, zeta, zeta_msp, "OK" if good else "FAIL"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
