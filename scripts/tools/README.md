# compute_damp_rate.py

## Usage

The damping rate can be computed directly from the command line:

```bash
python3 compute_damp_rate.py en2=0.0001
```

# dump_inspector.msp

## Usage

Prints the header of a `.dump` checkpoint file (format version, particle count, timestep,
physical time, field list, domain) without loading any particle data.

**From the command line**, via the `dump_inspector.sh` wrapper (generates the override for you).
CMake also drops a `dump_inspector` launcher in the build directory, right next to `exaDEM`
itself, so it works the same way:

```bash
./scripts/tools/dump_inspector.sh CheckpointFiles/exadem_0000012345.dump   # from the source tree
# or, from the build directory:
./dump_inspector CheckpointFiles/exadem_0000012345.dump
```

# ConvExaDEMToTxt

## Usage

Exports a `.dump` checkpoint file to plain-text column files: `<pattern_name>_particles.txt`
(one row per particle, columns taken from the dump's own header), `<pattern_name>_interactions.txt`
(one row per interaction, only written if the dump has any), and `<pattern_name>_summary.txt`.

Whether the dump was written with `read_dump_particle_interaction` or
`read_dump_particle_fragmentation` (which adds a `cluster` field) is detected automatically from
the dump's own header -- nothing to choose.

```bash
./scripts/tools/ConvExaDEMToTxt CheckpointFiles/exadem_0000012345.dump my_export   # from the source tree
# or, from the build directory:
./ConvExaDEMToTxt CheckpointFiles/exadem_0000012345.dump my_export
```