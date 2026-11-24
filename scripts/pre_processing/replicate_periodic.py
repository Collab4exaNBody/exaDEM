#!/usr/bin/env python3
"""
Replicate a periodic fragmented sample (Rockable conf) along x, y and/or z.

Typical workflow (beam periodic along x):

    neper -T -n 1000 -id 2 -domain "cube(2,0.5,0.5)" -periodicity x \
          -transform "translate(-1,-0.25,-0.25)" -o beam_per
    PYTHONPATH=. python3 pre_processing/tess2rockable.py beam_per.tess 1e-3 beam_per.shp
    PYTHONPATH=. python3 pre_processing/replicate_periodic.py beam_per_sticked.conf beam_x3.conf \
          --tess beam_per.tess --nx 3

With a periodic Neper tessellation, cells are not cut at the periodic boundaries:
a cell sticking out at x = xmax exactly fills the hole at x = xmin of the next copy.
Translating the sample by the period therefore gives a seamless tiling.

All copies keep the SAME cluster id: exaDEM (apply_particle_sticking) only creates
inner bonds between particles of the same cluster, so the copies get glued together
across the periodic interfaces. Shapes are reused (no new particle types).
"""

import argparse
import copy
import os

from lib.io_utils import read_rockable_file, write_rockable_file


def read_tess_period(tess_file):
    """Return the period vector (Lx, Ly, Lz) from the **periodicity section of a Neper .tess
    file (0 in non-periodic directions), or None if the tessellation is not periodic."""
    with open(tess_file) as f:
        lines = [l.strip() for l in f]
    for i, line in enumerate(lines):
        if line.lower() == "**periodicity":
            # *general / flags (px py pz) / period lengths (Lx Ly Lz)
            flags = [int(v) for v in lines[i + 2].split()]
            lengths = [float(v) for v in lines[i + 3].split()]
            return [L if p else 0.0 for p, L in zip(flags, lengths)]
    return None


def replicate(data, n, period, center=True):
    """Replicate particles n = (nx, ny, nz) times with translation vector period = (Lx, Ly, Lz)."""
    src = data.particles
    out = []
    shift0 = [-(n[d] - 1) * period[d] / 2.0 if center else 0.0 for d in range(3)]
    for ix in range(n[0]):
        for iy in range(n[1]):
            for iz in range(n[2]):
                idx = (ix, iy, iz)
                offset = [shift0[d] + idx[d] * period[d] for d in range(3)]
                for p in src:
                    q = copy.copy(p)
                    q.pos = tuple(p.pos[d] + offset[d] for d in range(3))
                    out.append(q)
    data.particles = out
    data.n_particles = len(out)
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Replicate a periodic fragmented sample (Rockable conf), keeping one cluster so copies are glued."
    )
    parser.add_argument("input_conf", help="Input Rockable conf (e.g. output of tess2rockable.py)")
    parser.add_argument("output_conf", help="Output Rockable conf")
    parser.add_argument("--tess", help="Neper .tess file: the period is read from its **periodicity section")
    parser.add_argument("--period", type=float, nargs=3, metavar=("LX", "LY", "LZ"),
                        help="Period vector (overrides --tess)")
    parser.add_argument("--nx", type=int, default=1)
    parser.add_argument("--ny", type=int, default=1)
    parser.add_argument("--nz", type=int, default=1)
    parser.add_argument("--no-center", action="store_true",
                        help="Do not recenter the replicated sample (copies are added towards +x/+y/+z)")
    parser.add_argument("--shapefile", help="Value written in the 'shapeFile' line (default: basename of the input one)")
    args = parser.parse_args()

    if args.period is not None:
        period = list(args.period)
    elif args.tess is not None:
        period = read_tess_period(args.tess)
        if period is None:
            parser.error(f"{args.tess} has no **periodicity section (use neper -periodicity)")
    else:
        parser.error("give --tess or --period")

    n = [args.nx, args.ny, args.nz]
    for d, name in enumerate("xyz"):
        if n[d] > 1 and period[d] == 0.0:
            parser.error(f"cannot replicate along {name}: the sample is not periodic in this direction")

    data = read_rockable_file(args.input_conf)
    data = replicate(data, n, period, center=not args.no_center)

    shp = args.shapefile or os.path.basename(data.params.values.get("shapeFile", ""))
    if shp:
        data.params.values["shapeFile"] = shp

    write_rockable_file(args.output_conf, data)
    print(f"period = {period}, copies = {n}")
    print(f"{data.n_particles} particles written to {args.output_conf} (shapeFile {shp})")


if __name__ == "__main__":
    main()
