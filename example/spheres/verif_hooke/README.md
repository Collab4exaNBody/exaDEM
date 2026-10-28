# Hooke contact law verification

Verification of exaDEM's linear (Hooke) spring-dashpot normal contact law
against its exact analytical solution, on single-impact tests.

## Origin of the tests

The test cases are taken from Y. C. Chung, J. Y. Ooi, *"Benchmark tests for
verifying discrete element modelling codes at particle impact level"*,
Granular Matter 13, 643-656 (2011),
https://doi.org/10.1007/s10035-011-0277-0, and **adapted to the Hooke
model**. The paper's reference solutions use Hertz-Mindlin contact theory,
whose normal part (Hertz law, spheres only) is verified in `../verif_hertz`; here the geometry, materials and impact
velocities of the paper are kept, but every reference curve is the exact
solution of the linear contact law that exaDEM actually solves.

Only the paper's normal-impact tests (1, 2, 3) are adapted. Tests 4-8
(oblique impacts, initial spin) check tangential behaviour that is specific
to Mindlin's theory and has no Hooke counterpart.

Adaptations compared to the paper:
- the normal stiffness `kn` is a constant (chosen from the paper's material
  data so that peak force and overlap stay of the same order);
- every particle starts **in contact** (zero gap), so t = 0 is the start of
  the impact;
- tangential stiffness and friction are zero; no gravity (as in the paper).

| Test | File | Description |
|---|---|---|
| 1 | `test1_two_spheres.msp` | elastic (e=1) head-on impact of two identical spheres, Glass and Limestone pairs, 20 m/s closing speed |
| 2 | `test2_sphere_plane.msp` | elastic (e=1) impact of an Al. alloy sphere (R=0.1 m) on a rigid plane, 0.2 m/s |
| 3 | `test3_restitution_sweep.msp` | damped impact of 4 Al. oxide spheres (R=2.5 mm) on a rigid plane, 3.9 m/s, `damp_rate` set for e = 0.9, 0.7, 0.5, 0.3 |

## Analytical solution

With `δ` the overlap, `meff` the effective mass (`m/2` for two identical
spheres, `m` against a rigid plane) and `c = 2 ζ sqrt(kn meff)` (`ζ` =
`damp_rate`, see `compute_damp()` in `common_kernels.hpp`), exaDEM solves

```
meff δ'' = -Fn,    Fn = max(0, kn δ + c δ'),    δ(0) = 0,  δ'(0) = v0
```

As long as `Fn > 0` this is the underdamped oscillator

```
δ(t) = v0/ωd · exp(-ζ ωn t) · sin(ωd t),   ωn = sqrt(kn/meff),   ωd = ωn sqrt(1 - ζ²)
```

- **Elastic (ζ = 0, tests 1-2)**: `Fn = kn δ`, contact duration `tc = π/ωn`,
  peak force `kn v0/ωn`, the normal velocity is exactly reversed (e = 1).
- **Damped (test 3)**: exaDEM clamps `Fn` at zero (the dashpot can not pull
  the particles back), so the contact ends at `t*` where `kn δ + c δ' = 0`,
  i.e. `tan(ωd t*) = -c ωd / (kn - c ζ ωn)`, and `e = -δ'(t*)/v0`. This
  clamped `e` is slightly higher than the textbook value
  `exp(-ζπ/sqrt(1-ζ²))` used to choose `damp_rate`:

  | target e | ζ (`damp_rate`) | exact e (with clamp) | exaDEM |
  |---|---|---|---|
  | 0.9 | 0.033518 | 0.9020 | 0.9023 |
  | 0.7 | 0.112808 | 0.7182 | 0.7189 |
  | 0.5 | 0.215454 | 0.5503 | 0.5511 |
  | 0.3 | 0.357857 | 0.3971 | 0.3980 |

The solution is implemented in `hooke_lib.py` (`HookeContact`).

## Files

- `test*.msp` / `test*.xyz`: simulations (Paraview output for the
  velocities, interaction dump at every step for overlap and force);
- `hooke_lib.py`: test parameters, analytical solution, readers of the
  exaDEM outputs (standard library only);
- `check_hooke.py`: compares a run with the analytical solution, exit code
  1 on failure;
- `plot_curves.py`: velocity / force-overlap / force-time plots, exaDEM
  (crosses) vs. analytical (line), requires matplotlib;
- `verify_damp_rate.py`: checks that the `damprate` values of test 3 match
  `scripts/tools/compute_damp_rate.py` for the target e.

## Running

```
exaDEM test1_two_spheres.msp
exaDEM test2_sphere_plane.msp
exaDEM test3_restitution_sweep.msp
python3 check_hooke.py test1     # test2, test3
python3 plot_curves.py           # writes verif_hooke_test{1,2,3}.png
```

Both scripts take `--dir` (directory where exaDEM was run, default `.`);
`plot_curves.py` also takes `--out` (output directory).

## Non-regression tests (CTest)

For each test, `Example_ExaDEM_VerifHookeTest<N>...` runs exaDEM (serial)
and `..._Check` runs `check_hooke.py` on its output (CTest fixture, so the
check always runs after the simulation). Checked quantities and tolerances:

| Quantity | Tolerance | Typical error |
|---|---|---|
| force curve: max over the dump of \|Fn(t) - Fn_exact(t)\| / Fn_exact peak | 2 % | < 1e-5 (elastic), < 0.7 % (damped) |
| peak force, max overlap | 1 % | < 1e-5 (elastic), < 0.6 % (damped) |
| contact duration | 2 time steps | < 1.1 step |
| restitution coefficient e | 0.005 (absolute) | < 0.001 |

The remaining error of the damped case is the O(dt) time discretisation of
the dashpot force. A 2 % error on `kn` is enough to make the check fail.
`Example_ExaDEM_VerifHookeDampRate` runs `verify_damp_rate.py`.
