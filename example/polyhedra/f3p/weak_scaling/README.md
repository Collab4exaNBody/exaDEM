# f3p weak scaling

Three-point bending of a beam made of `N` copies, along its width (y), of a 1000-fragment sample
that is periodic in y. The work per copy is constant, so `N` is the weak-scaling parameter.

```bash
python3 make_f3p_weak_scaling.py N            # creates ./case_yN
cd case_yN && mpirun -n P exaDEM f3p.msp --nogpu
```

Options: `--steps` (default 30000), `--paraview <freq>` (default: no output), `--cell-size` (default 0.2,
must be at least 2 x 0.083 + rcut_inc), `--out <dir>`.
The script needs the exaDEM python scripts (`<exaDEM>/scripts`, found relative to this folder or
through `EXADEM_SCRIPTS`).

Each case contains:

| file | content |
|------|---------|
| `beam.conf` | `N` x 1000 fragments, beam `[-1,1] x [-N/4,N/4] x [-0.25,0.25]` |
| `beam_pery.shp` | the 1000 shapes (identical whatever `N`; exaNBody needs `-DXNB_PARTICLE_TYPE_INT=uint16_t`) |
| `support_*.shp`, `punch.shp` | cylinders spanning the whole width |
| `f3p.msp` | input deck, domain grid sized for `N` |

All fragments are in the same cluster: `apply_particle_sticking` glues the copies together, and the
seams between copies are bonded like the inside of a copy.

## Sample

`beam_pery.shp` / `beam_pery_sticked.conf` come from Neper 7 (centroidal cells, which avoids the very small
fragments of a plain Voronoi tessellation; periodic along y, so cells stick out of the faces y = +/-0.25 and
fit into the next copy):

```bash
neper -T -n 1000 -id 2 -domain "cube(2,0.5,0.5)" -periodicity y -morpho centroidal \
      -transform "translate(-1,-0.25,-0.25)" -o beam_pery
cd <exaDEM>/scripts && PYTHONPATH=. python3 pre_processing/tess2rockable.py beam_pery.tess 1e-3 beam_pery.shp
```

To replicate another periodic sample, see `scripts/pre_processing/replicate_periodic.py`.

## Picture

`render_f3p.py` (ParaView) renders a case run with `--paraview <freq>`:

```bash
pvpython --force-offscreen-rendering render_f3p.py case_yN/F3P_WS_YN <step> image.png [copy|velocity]
```
