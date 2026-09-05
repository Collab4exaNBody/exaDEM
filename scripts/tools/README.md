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
./scripts/tools/dump_inspector.sh ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump   # from the source tree
# or, from the build directory:
./dump_inspector ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump
```

# ConvExaDEMToRockable

## Usage

Exports a `.dump` checkpoint file's particles to a Rockable `.conf` file. Interactions are not
exported yet.

Without a `.shp` shape file, the particle `name` column is the numeric type index from the
`.dump` (a `.dump` alone has no shape-name mapping). Pass one as a third argument to map each
type index to its real shape name (by registration order); it gets copied next to the output
`.conf` as `shape.shp`.

Which of the known `.dump` field-set combinations (interaction/fragmentation, with/without a
`group` field) matches a given dump is auto-detected from its header -- nothing to choose.

`dt` (timestep size) isn't stored in a `.dump` either, so the `.conf`'s `dt` line stays `0`
unless you pass `--dt=VALUE`.

```bash
./scripts/tools/ConvExaDEMToRockable ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump conf0.conf   # from the source tree
./scripts/tools/ConvExaDEMToRockable ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump conf0.conf ExaDEMOutputDir/CheckpointFiles/RestartShapeFile.shp
# or, from the build directory:
./ConvExaDEMToRockable ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump conf0.conf
./ConvExaDEMToRockable ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump conf0.conf ExaDEMOutputDir/CheckpointFiles/RestartShapeFile.shp
```

**`--last`** picks the highest-iteration `exadem_*.dump` (and `RestartShapeFile.shp`, if present)
under `<input-dir>/CheckpointFiles/` for you -- `<input-dir>` defaults to `ExaDEMOutputDir`,
override with `--input-dir=DIR`:

```bash
./scripts/tools/ConvExaDEMToRockable --last conf0.conf
./scripts/tools/ConvExaDEMToRockable --last conf0.conf --input-dir=OtherOutputDir
./scripts/tools/ConvExaDEMToRockable --last conf0.conf --dt=0.0001
# or, from the build directory:
./ConvExaDEMToRockable --last conf0.conf
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

# ConvExaDEMToXYZ

## Usage

Exports a `.dump` checkpoint file's particles to a plain XYZ file: particle count, domain bounds
upper corner (`x y z`), then one `type x y z` row per particle.

Without a `.shp` shape file, the `type` column is the numeric type index from the `.dump` (a
`.dump` alone has no shape-name mapping). Pass one as a third argument to map each type index to
its real shape name (by registration order).

**`--type-map=[S1,S2,S3]`** maps each type index to a name directly (by position), without
needing a `.shp` file -- handy when you know the names but don't have/want the shape geometry. It
takes priority over a `.shp` file's names if both are given.

Which of the known `.dump` field-set combinations (interaction/fragmentation, with/without a
`group` field) matches a given dump is auto-detected from its header -- nothing to choose.

```bash
./scripts/tools/ConvExaDEMToXYZ ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump out.xyz   # from the source tree
./scripts/tools/ConvExaDEMToXYZ ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump out.xyz ExaDEMOutputDir/CheckpointFiles/RestartShapeFile.shp
./scripts/tools/ConvExaDEMToXYZ ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump out.xyz --type-map=[S1,S2,S3]
# or, from the build directory:
./ConvExaDEMToXYZ ExaDEMOutputDir/CheckpointFiles/exadem_0000012345.dump out.xyz
```

**`--last`** picks the highest-iteration `exadem_*.dump` (and `RestartShapeFile.shp`, if present)
under `<input-dir>/CheckpointFiles/` for you -- `<input-dir>` defaults to `ExaDEMOutputDir`,
override with `--input-dir=DIR`:

```bash
./scripts/tools/ConvExaDEMToXYZ --last out.xyz
./scripts/tools/ConvExaDEMToXYZ --last out.xyz --input-dir=OtherOutputDir
# or, from the build directory:
./ConvExaDEMToXYZ --last out.xyz
```