#!/usr/bin/env python3
"""Plot exaDEM results of the Hertz verification tests (Chung & Ooi 2011, tests 1-3)
against the solution of the Hertz impact: velocity vs time, normal force vs overlap,
normal force vs time. exaDEM = crosses, reference = solid line, Table 3 of the paper =
stars (max overlap / max force) and dotted lines (contact time). Requires matplotlib.

Usage: plot_curves.py [test1 test2 test3] [--dir RUN_DIR] [--out OUT_DIR]
RUN_DIR is where exaDEM was run (it contains VerifHertzTest<N>Dir); one PNG per test
is written to OUT_DIR (verif_hertz_<test>.png).
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hertz_lib import TESTS, analytical, read_contact_history, read_velocity_history  # noqa: E402

N_MARKERS = 60


def every(values, n=N_MARKERS):
    step = max(1, len(values) // n)
    return values[::step]


def plot_test(test, run_dir, out_dir):
    cfg = TESTS[test]
    out = os.path.join(run_dir, cfg["dir"])
    dt = cfg["dt"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax_v, ax_fd, ax_ft) = plt.subplots(1, 3, figsize=(16, 4.8))

    vel = read_velocity_history(out, dt)
    for k, v in enumerate(cfg.get("plot_velocities", cfg["velocities"])):
        color = colors[k % len(colors)]
        exact = analytical(cfg["contacts"][v["contact"]])
        h = vel.get(v["id"])
        if h is None:
            continue
        t = h["t"]
        tt = [t[-1] * i / 400 for i in range(401)]
        ax_v.plot([x * 1e6 for x in tt], [v["v0"] * exact.velocity(x) / exact.v0 for x in tt],
                  "-", lw=1, color=color, label=v["name"] + " (reference)")
        ax_v.plot([x * 1e6 for x in every(t)], every([vv[v["comp"]] for vv in h["v"]]),
                  "x", ms=5, color=color, label=v["name"] + " (exaDEM)")
    ax_v.set_xlabel("time [µs]")
    ax_v.set_ylabel("velocity [m/s]")
    ax_v.set_title("Velocity")

    for k, c in enumerate(cfg["contacts"]):
        color = colors[k % len(colors)]
        exact = analytical(c)
        h = read_contact_history(out, dt, c["i"], c["j"])
        tt = [exact.t_end * i / 400 for i in range(401)]
        ax_fd.plot([exact.overlap(x) * 1e6 for x in tt], [exact.force(x) for x in tt],
                   "-", lw=1, color=color, label=c["name"] + " (reference)")
        ax_ft.plot([x * 1e6 for x in tt], [exact.force(x) for x in tt],
                   "-", lw=1, color=color, label=c["name"] + " (reference)")
        keep = [i for i, f in enumerate(h["fn"]) if f > 0]
        ax_fd.plot(every([h["overlap"][i] * 1e6 for i in keep]), every([h["fn"][i] for i in keep]),
                   "x", ms=5, color=color, label=c["name"] + " (exaDEM)")
        ax_ft.plot(every([h["t"][i] * 1e6 for i in keep]), every([h["fn"][i] for i in keep]),
                   "x", ms=5, color=color, label=c["name"] + " (exaDEM)")
        if c.get("paper"):
            d_paper, f_paper, t_paper = c["paper"]
            ax_fd.plot([d_paper * 1e6], [f_paper], "*", ms=12, mec="black", color=color,
                       label=c["name"] + " (paper, Table 3)")
            ax_ft.axvline(t_paper * 1e6, ls=":", color=color, label=c["name"] + " contact time (paper)")
    ax_fd.set_xlabel("overlap [µm]")
    ax_fd.set_ylabel("normal force [N]")
    ax_fd.set_title("Normal force vs overlap")
    ax_ft.set_xlabel("time [µs]")
    ax_ft.set_ylabel("normal force [N]")
    ax_ft.set_title("Normal force vs time")

    for ax in (ax_v, ax_fd, ax_ft):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(cfg["title"])
    fig.tight_layout()
    path = os.path.join(out_dir, "verif_hertz_%s.png" % test)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("wrote", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tests", nargs="*", help="subset of %s (default: all)" % ", ".join(sorted(TESTS)))
    parser.add_argument("--dir", default=".", help="directory where exaDEM was run")
    parser.add_argument("--out", default=".", help="output directory for the PNG files")
    args = parser.parse_args()
    unknown = set(args.tests) - set(TESTS)
    if unknown:
        parser.error("unknown test(s): " + ", ".join(sorted(unknown)))
    for test in args.tests or sorted(TESTS):
        plot_test(test, args.dir, args.out)


if __name__ == "__main__":
    main()
