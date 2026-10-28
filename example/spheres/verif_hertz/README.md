# Hertz contact law verification

Verification of exaDEM's Hertz normal contact law (spheres only) against the
solution of a single impact.

The tests reproduce the normal impact tests (tests 1, 2 and 3) of Y. C. Chung,
J. Y. Ooi, *"Benchmark tests for verifying discrete element modelling codes at
particle impact level"*, Granular Matter 13, 643-656 (2011),
https://doi.org/10.1007/s10035-011-0277-0, with the materials of Table 2 of the
paper, and compare the results with Table 3 (Hertz theory). Tests 4 to 8
(oblique impacts) are not reproduced: the tangential force of exaDEM is a constant
`kt` spring with a Coulomb limit, not Mindlin's theory (`kt = mu = 0` here).

## The law

With `delta` the overlap, `E*` the reduced Young modulus, `R*` the effective
radius and `m*` the effective mass of the contact:

```
Fe = 4/3 E* sqrt(R*) delta^(3/2)                       (elastic part)
k  = dFe/d(delta) = 2 E* sqrt(R* delta)                (tangent stiffness)
Fn = max(0, Fe + c ddelta/dt),   c = 2 damp_rate sqrt(m* k)
```

- `1/E* = (1 - nu1^2)/E1 + (1 - nu2^2)/E2`, computed from the Young modulus
  and the Poisson ratio **of each material**;
- `damp_rate` is the damping ratio `zeta` at the tangent stiffness (same
  definition as for the Hooke law, with `k` instead of `kn`). The damping
  coefficient grows with `delta^(1/4)`. The restitution coefficient only
  depends on `damp_rate`, with a different relation than for the Hooke law
  (see test 3);
- the Coulomb limit of the friction uses the elastic Hertz force:
  `|ft| <= mu Fe`.

## Operators and parameters

```
contact_sphere_hertz_[none|cohesive|dmt]
```

The contact law is always symetric (each interaction is computed once, for both particles).

The Hertz law has its own parameters (`HertzParams`, without `kn`) and its own
parameter operators, separate from the Hooke ones (`ContactParams`,
`multimat_contact_params`, `drivers_contact_params`). The materials are defined per
group, so there is no single-material version (no `config` / `config_driver`):

| Operator | Material parameters |
|---|---|
| `multimat_contact_params_hertz` | `young`, `poisson` with one value per group |
| `drivers_contact_params_hertz` | `driver_young`, `driver_poisson` with one value per driver (the group materials come from `multimat_contact_params_hertz`) |

The other parameters (`kt`, `kr`, `mu`, `damprate`, `fc`, `dncut`, `gamma`) are
the ones of the Hooke law, one value per listed pair.

### Critical time step

Not estimated yet for the Hertz law (TODO in `scan_hertz_dt`): `check_dt_critical`
only prints a warning. The time step must be chosen by the user, e.g. a small
fraction of the Hertz contact time `2.87 (m*^2 / (R* E*^2 v))^(1/5)`.

## Tests

| Test | File | Description |
|---|---|---|
| 1 | `test1_two_spheres.msp` | elastic head-on impact of two identical spheres (R = 1 cm, +/-10 m/s), Glass pair and Limestone pair, one Young modulus and Poisson ratio per material |
| 2 | `test2_sphere_plane.msp` | elastic impact of an Al. alloy sphere (R = 0.1 m) on a rigid plane, 0.2 m/s |
| 3 | `test3_restitution_sweep.msp` | impacts of 4 Al. oxide spheres (R = 2.5 mm) on a rigid plane, 3.9 m/s, `damp_rate` set for e = 0.9, 0.7, 0.5, 0.3 |

| Material (Table 2) | E [GPa] | nu | rho [kg/m3] |
|---|---|---|---|
| Glass | 48 | 0.20 | 2800 |
| Limestone | 20 | 0.25 | 2500 |
| Al. alloy | 70 | 0.30 | 2699 |
| Al. oxide | 380 | 0.23 | 4000 |

Every particle starts **in contact** (zero gap), so `t = 0` is the start of the
impact; no gravity (as in the paper). The rigid plane is a driver with a very large
Young modulus (`driver_young: [ 1.e40 ]`), i.e. `E* = E / (1 - nu^2)`.

Results (exaDEM vs. Table 3 of the paper):

| Test | Quantity | exaDEM | Paper |
|---|---|---|---|
| 1, Glass | max overlap / max force / contact time | 274.1 um / 10697 N / 40.3 us | 274.1 um / 10697 N / 40.3 us |
| 1, Limestone | max overlap / max force / contact time | 368.3 um / 7108 N / 54.2 us | 368.3 um / 7108 N / 54.2 us |
| 2, Al. alloy | max overlap / max force / contact time | 49.72 um / 11370 N / 731 us | 49.7 um / 11370 N / 731.6 us |
| 3 | restitution coefficient | 0.9000, 0.7000, 0.5000, 0.2999 | 0.9, 0.7, 0.5, 0.3 |

### Restitution coefficient and `damp_rate`

With the damping of the Hertz law, the restitution coefficient only depends on
`damp_rate` (not on the velocity, the masses, the radii or `E*`), but the relation
is not the one of the Hooke law. `hertz_lib.damp_rate_for_restitution(e)` computes
it (bisection on the impact solution):

| e | 0.9 | 0.7 | 0.5 | 0.3 |
|---|---|---|---|---|
| `damp_rate` | 0.0312797 | 0.1115664 | 0.2330340 | 0.4539161 |

## Reference solution

`meff x'' = -Fn` with `x(0) = 0`, `x'(0) = v0`, integrated with a 4th order
Runge-Kutta scheme (`hertz_lib.py`, `HertzContact`, standard library only). For the
elastic case it gives the Hertz results of the paper: `delta_max = (15 m* v0^2 / (16 E* sqrt(R*)))^(2/5)`,
`Fmax = 4/3 E* sqrt(R*) delta_max^(3/2)`, contact time `~ 2.87 (m*^2 / (R* E*^2 v0))^(1/5)`,
restitution `e = 1`. For the damped case the contact ends when the force reaches 0.

## Files

- `test*.msp` / `test*.xyz`: simulations (Paraview output for the velocities,
  interaction dump at every step for overlap and force);
- `hertz_lib.py`: test parameters and reference solution; the readers of the exaDEM
  outputs are the ones of `../verif_hooke/hooke_lib.py`;
- `check_hertz.py`: compares a run with the reference solution and with Table 3 of the
  paper, exit code 1 on failure;
- `verify_damp_rate.py`: checks the `damprate` values of test 3;
- `plot_curves.py`: velocity / force-overlap / force-time plots, exaDEM (crosses) vs.
  reference (line) and Table 3 of the paper (stars, dotted lines), requires matplotlib.

## Running

```
exaDEM test1_two_spheres.msp
exaDEM test2_sphere_plane.msp
exaDEM test3_restitution_sweep.msp
python3 check_hertz.py test1     # test2, test3
python3 plot_curves.py           # writes verif_hertz_test{1,2,3}.png
```

Both scripts take `--dir` (directory where exaDEM was run, default `.`);
`plot_curves.py` also takes `--out` (output directory).

## Non-regression tests (CTest)

For each test, `Example_ExaDEM_VerifHertzTest<N>...` runs exaDEM (serial) and
`..._Check` runs `check_hertz.py` on its output. Checked quantities and tolerances:

| Quantity | Tolerance | Typical error |
|---|---|---|
| force curve: max over the dump of \|Fn(t) - Fn_exact(t)\| / Fn_exact peak | 2 % | < 1e-3 |
| peak force, max overlap | 1 % | < 1e-3 |
| contact duration | 3 time steps | < 1.6 step |
| restitution coefficient e | 0.005 (absolute) | < 1e-4 |
| max overlap, max force, contact time vs. Table 3 of the paper | 0.5 % | < 0.1 % (rounding of the paper) |

`Example_ExaDEM_VerifHertzDampRate` runs `verify_damp_rate.py`. Remove the
`VerifHertzTest<N>Dir` directories before running the tests again with different
settings: the check reads the last velocity output of the directory.
