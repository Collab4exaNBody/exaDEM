#!/usr/bin/env python3
"""Checks PolyhedraCompacity.txt written by triperiodic_compression.msp (+ its regression file).

The barostat compresses the periodic cell: the number of particles and their volume do not change, and the
compacity must increase from the one of the initial lattice.
"""
import argparse
import os
import sys

N_PARTICLES = 125
SHAPE_VOLUME = 3.4814549352253138e-01  # volume of the Dodecahedron in dodecahedron.shp
INITIAL_COMPACITY = N_PARTICLES * SHAPE_VOLUME / 6.0**3
TOL = 1e-5  # values are written with 6 decimals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".")
    parser.add_argument("--min-final-compacity", type=float, required=True,
                        help="minimal compacity expected at the end of the run")
    args = parser.parse_args()
    path = os.path.join(args.dir, "PolyhedraTriperiodicCompressionDir", "ExaDEMAnalyses", "PolyhedraCompacity.txt")
    with open(path) as f:
        lines = [l.split() for l in f if l.strip()]

    col = {name: i for i, name in enumerate(lines[0])}
    rows = [[float(x) for x in l] for l in lines[1:]]

    # a single (sequential) run: time and compacity must both increase strictly at every output
    errors = 0
    previous_t, previous_c = 0.0, INITIAL_COMPACITY
    for r in rows:
        t, n, solid, c = r[col["Time"]], r[col["NbParticles"]], r[col["SolidVolume"]], r[col["Compacity"]]
        ok = (t > previous_t and n == N_PARTICLES and abs(solid - N_PARTICLES * SHAPE_VOLUME) < TOL
              and c > previous_c)
        print(f"t = {t:.3f}  compacity = {c:.6f}  {'OK' if ok else 'WRONG'}")
        errors += not ok
        previous_t, previous_c = t, c
    if not rows:
        print("no data")
        errors += 1
    elif previous_c < args.min_final_compacity:
        print(f"final compacity {previous_c:.6f} < {args.min_final_compacity}")
        errors += 1
    if errors:
        print(f"{errors} error(s)")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
