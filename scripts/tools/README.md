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