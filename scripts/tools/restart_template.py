import argparse
import os
import re
import sys

# Matches a top-level key ("key:" or "+key:", no leading whitespace) -- the start of a new
# top-level .msp block. Everything up to the next such line (or EOF) belongs to that block.
TOP_LEVEL_KEY_RE = re.compile(r"^\+?([A-Za-z_][A-Za-z0-9_]*):")

# Matches the start of a "- name" / "- name:" sequence item, capturing its indentation and name.
SEQUENCE_ITEM_RE = re.compile(r"^(\s*)-\s*\+?([A-Za-z_][A-Za-z0-9_]*)\s*:?")

# input_data: entries to keep regardless of restart: everything else is assumed to be part of
# building the initial particle set from scratch (grid bootstrap, lattice generation, reading a
# dump/shape/Rockable config, computing initial per-particle fields from it, ...) -- which the
# restart operator now does on its own in one step -- so it gets dropped. There are too many
# such particle-setup operator names across real launch files to list them all (read_conf_rockable,
# density_from_shape, set_velocity, lattice, common_setup_particles, ... the list keeps growing),
# but the contact-law/material parameter setup that must survive a restart consistently follows
# one naming convention across every real file seen so far: a "_params" suffix
# (multimat_contact_params:, drivers_contact_params:, inner_bond_params:, ...).
KEEP_ENTRY_RE = re.compile(r"_params$")

# global: values forced for a restart, regardless of what the original launch file set. Restored
# particles/bonds already carry whatever state these flags would otherwise (re-)compute at
# startup, so re-running that logic on restart would be redundant at best (sticking: the
# restored bonds already encode which particles are stuck -- recomputing it risks wrongly
# re-triggering the initial-contact sticking detection against already-evolved positions).
GLOBAL_OVERRIDES = {
    "apply_particle_sticking": "false",
}
GLOBAL_OVERRIDE_RE = re.compile(
    r"^(\s*)(" + "|".join(re.escape(k) for k in GLOBAL_OVERRIDES) + r")\s*:.*$"
)


def split_top_level_blocks(lines):
    """Splits .msp file lines into (key, block_lines) pairs, in order. key is None for any
    leading lines (comments, blank lines) before the first top-level key."""
    blocks = []
    current_key = None
    current_lines = []
    for line in lines:
        m = TOP_LEVEL_KEY_RE.match(line)
        if m:
            blocks.append((current_key, current_lines))
            current_key = m.group(1)
            current_lines = [line]
        else:
            current_lines.append(line)
    blocks.append((current_key, current_lines))
    return blocks


def split_sequence_items(lines):
    """Splits the lines of a YAML block sequence ("- name", "- name:" + nested lines) into
    (name, item_lines) pairs, in order. Lines before the first "- " item (just the "key:" line
    itself) are returned first as (None, ...)."""
    items = []
    current_name = None
    current_lines = []
    item_indent = None
    for line in lines:
        m = SEQUENCE_ITEM_RE.match(line)
        if m and (item_indent is None or len(m.group(1)) == item_indent):
            item_indent = len(m.group(1))
            items.append((current_name, current_lines))
            current_name = m.group(2)
            current_lines = [line]
        else:
            current_lines.append(line)
    items.append((current_name, current_lines))
    return items


def adapt_input_data(block_lines):
    """Rewrites input_data:'s sequence down to just "- restart" plus whichever entries match
    KEEP_ENTRY_RE (the behavior-law / contact-law parameter setup operators) -- restart first,
    since it's the one populating the particles those parameters apply to, then the kept entries
    in their original order."""
    out = []
    kept = []
    indent = "  "
    for name, item_lines in split_sequence_items(block_lines):
        if name is None:
            out.extend(item_lines)  # the "input_data:" key line itself
        else:
            indent = SEQUENCE_ITEM_RE.match(item_lines[0]).group(1)
            if KEEP_ENTRY_RE.search(name):
                kept.extend(item_lines)
    out.append(indent + "- restart\n")
    out.extend(kept)
    return out


def adapt_global(block_lines):
    """Rewrites global:'s lines, forcing any key in GLOBAL_OVERRIDES to its restart value."""
    out = []
    for line in block_lines:
        m = GLOBAL_OVERRIDE_RE.match(line)
        out.append(line if not m else m.group(1) + m.group(2) + ": " + GLOBAL_OVERRIDES[m.group(2)] + "\n")
    return out


def adapt_for_restart(lines):
    """Rewrites a real .msp launch file for restart: input_data: is cut down to "- restart" plus
    its behavior-law/contact-law parameter entries (see adapt_input_data), setup_drivers: is
    dropped entirely (the restart operator reads drivers back on its own), global: has its
    restart-forced values applied (see GLOBAL_OVERRIDES), and every other block (domain:,
    compute_force:, includes:, ...) is kept byte-for-byte as-is."""
    out = []
    for key, block_lines in split_top_level_blocks(lines):
        if key == "input_data":
            out.extend(adapt_input_data(block_lines))
        elif key == "setup_drivers":
            continue
        elif key == "global":
            out.extend(adapt_global(block_lines))
        else:
            out.extend(block_lines)
    return [line for line in out if not line.lstrip().startswith("#")]


def main():
    parser = argparse.ArgumentParser(
        description="Adapts a real exaDEM launch .msp file for restart: swaps input_data: for "
                     "just the restart operator and drops setup_drivers: (drivers are read back "
                     "by restart itself), keeping everything else -- domain, compute_force and "
                     "its contact-law parameters, global, includes, etc. -- untouched.")
    parser.add_argument("filename", help="Path to the launch .msp file to adapt.")
    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print("[Error] File not found: " + args.filename)
        sys.exit(1)

    with open(args.filename) as f:
        lines = f.readlines()

    out_lines = adapt_for_restart(lines)

    stem, ext = os.path.splitext(args.filename)
    out_filename = stem + "_restart" + ext

    with open(out_filename, "w") as f:
        f.writelines(out_lines)

    print("Wrote " + out_filename)


if __name__ == "__main__":
    main()
