#!/usr/bin/env python3
"""Check exaDEM outputs of the Hertz verification tests against the exact
(numerical) solution of the Hertz impact (see hertz_lib.HertzContact).

Usage: check_hertz.py test1|test2|test3 [--dir RUN_DIR]
RUN_DIR is where exaDEM was run (it contains VerifHertzTest<N>Dir).
Exit code 0 if every check passes, 1 otherwise.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hertz_lib import TESTS, analytical, read_contact_history, read_velocity_history  # noqa: E402

TOL_FORCE_CURVE = 0.02   # max |F_exaDEM(t) - F_exact(t)| / F_exact_peak
TOL_PEAK = 0.01          # relative error on peak force and max overlap
TOL_DURATION_STEPS = 3   # contact duration error, in time steps
TOL_RESTITUTION = 0.005  # absolute error on e = |v_out| / |v_in|
TOL_PAPER = 0.005        # relative error with the values of Table 3 of Chung & Ooi (2011), rounded to 3-4 digits


def check(test, run_dir):
    cfg = TESTS[test]
    out_dir = os.path.join(run_dir, cfg["dir"])
    dt = cfg["dt"]
    results = []

    def record(label, value, ref, err, tol):
        results.append((label, value, ref, err, tol, err <= tol))

    for c in cfg["contacts"]:
        h = read_contact_history(out_dir, dt, c["i"], c["j"])
        if not h["t"]:
            results.append((c["name"] + ": no interaction found", 0, 0, 1, 0, False))
            continue
        exact = analytical(c)
        f_peak = exact.peak_force()
        curve_err = max(abs(f - exact.force(t)) for t, f in zip(h["t"], h["fn"])) / f_peak
        record(c["name"] + ": force curve F(t)", curve_err, 0.0, curve_err, TOL_FORCE_CURVE)
        record(c["name"] + ": peak force [N]", max(h["fn"]), f_peak,
               abs(max(h["fn"]) / f_peak - 1), TOL_PEAK)
        d_max = exact.max_overlap()
        record(c["name"] + ": max overlap [m]", max(h["overlap"]), d_max,
               abs(max(h["overlap"]) / d_max - 1), TOL_PEAK)
        # every test starts in contact at t = 0
        duration = max(t for t, f in zip(h["t"], h["fn"]) if f > 0)
        record(c["name"] + ": contact duration [s]", duration, exact.t_end,
               abs(duration - exact.t_end) / dt, TOL_DURATION_STEPS)
        if c.get("paper"):
            d_paper, f_paper, t_paper = c["paper"]
            record(c["name"] + ": max overlap vs paper [m]", max(h["overlap"]), d_paper,
                   abs(max(h["overlap"]) / d_paper - 1), TOL_PAPER)
            record(c["name"] + ": peak force vs paper [N]", max(h["fn"]), f_paper,
                   abs(max(h["fn"]) / f_paper - 1), TOL_PAPER)
            record(c["name"] + ": contact time vs paper [s]", duration, t_paper,
                   abs(duration / t_paper - 1), TOL_PAPER)

    vel = read_velocity_history(out_dir, dt)
    for v in cfg["velocities"]:
        if v["id"] not in vel:
            results.append((v["name"] + ": no velocity output", 0, 0, 1, 0, False))
            continue
        exact = analytical(cfg["contacts"][v["contact"]])
        e = -vel[v["id"]]["v"][-1][v["comp"]] / v["v0"]
        record(v["name"] + ": restitution e", e, v["e"], abs(e - v["e"]), TOL_RESTITUTION)
    for p in cfg.get("pairs", []):
        if p["i"] not in vel or p["j"] not in vel:
            results.append((p["name"] + ": no velocity output", 0, 0, 1, 0, False))
            continue
        contact = cfg["contacts"][p["contact"]]
        exact = analytical(contact)
        v_rel = vel[p["j"]]["v"][-1][p["comp"]] - vel[p["i"]]["v"][-1][p["comp"]]
        e = v_rel / contact["v0"]  # i approaches j, then they separate: v_j - v_i > 0
        record(p["name"] + ": restitution e", e, p["e"], abs(e - p["e"]), TOL_RESTITUTION)

    print(cfg["title"])
    print("%-36s %14s %14s %10s %8s  %s" % ("quantity", "exaDEM", "exact", "error", "tol", ""))
    for label, value, ref, err, tol, ok in results:
        print("%-36s %14.6g %14.6g %10.3g %8.3g  %s" % (label, value, ref, err, tol, "OK" if ok else "FAIL"))
    return all(r[-1] for r in results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("test", choices=sorted(TESTS))
    parser.add_argument("--dir", default=".", help="directory where exaDEM was run")
    args = parser.parse_args()
    ok = check(args.test, args.dir)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
