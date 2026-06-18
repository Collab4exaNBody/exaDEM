#!/usr/bin/env python3
"""
broken_interfaces_to_vtp.py
----------------------------

Converts a "BrokenInterfaces" text file into a ParaView file
(.vtp - VTK PolyData XML) to visualize the broken interfaces
between pairs of grains.

Input file format (one header line, then one line per interface):

    iteration id_a id_b nb_vertex [vertex_x vertex_y vertex_z]...

Each line describes a polygon (the broken interface) defined by
nb_vertex consecutive 3D vertices, and carries scalar attributes:
- the iteration at which the break was detected
- the IDs of the two grains (id_a, id_b) involved

The output file is a PolyData (.vtp) containing:
- the points (polygon vertices)
- the "polygon" cells (one per line of the input file)
- per-cell data fields (CellData): iteration, id_a, id_b

Usage:
    python broken_interfaces_to_vtp.py BrokenInterfaces.txt output.vtp

If no arguments are given, the script defaults to "BrokenInterfaces.txt"
as input and "BrokenInterfaces.vtp" as output.

The .vtp file can be opened directly in ParaView (File > Open).
"""

import sys


def read_interfaces(input_path):
    """Reads the text file and returns a list of dictionaries, one
    per line/interface, containing: iteration, id_a, id_b, vertices
    (list of (x, y, z) tuples).
    """
    interfaces = []
    with open(input_path, "r") as f:
        first_line = True
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            # Skip the header line (text, not numbers)
            if first_line:
                first_line = False
                if not line[0].isdigit() and line[0] not in "+-":
                    continue

            tokens = line.split()
            try:
                iteration = int(tokens[0])
                id_a = int(tokens[1])
                id_b = int(tokens[2])
                nb_vertex = int(tokens[3])
            except (ValueError, IndexError):
                print(f"Line {line_number} skipped (unexpected format): {line[:60]}...")
                continue

            coords = tokens[4:]
            expected = nb_vertex * 3
            if len(coords) < expected:
                print(
                    f"Line {line_number} skipped: expected {nb_vertex} "
                    f"vertices ({expected} values) but only "
                    f"{len(coords)} values found."
                )
                continue

            coords = [float(c) for c in coords[:expected]]
            vertices = [
                (coords[3 * i], coords[3 * i + 1], coords[3 * i + 2])
                for i in range(nb_vertex)
            ]

            interfaces.append(
                {
                    "iteration": iteration,
                    "id_a": id_a,
                    "id_b": id_b,
                    "vertices": vertices,
                }
            )

    return interfaces


def write_vtp(interfaces, output_path):
    """Writes the interfaces as VTK PolyData (.vtp, ASCII XML format),
    with one polygon per interface and CellData fields
    (iteration, id_a, id_b, nb_vertex).
    """
    # --- Build the global point list + per-polygon index list ---
    all_points = []  # list of (x, y, z)
    connectivity_lists = []  # list of point-index lists, one per polygon
    current_offset = 0

    for interface in interfaces:
        indices = []
        for v in interface["vertices"]:
            all_points.append(v)
            indices.append(current_offset)
            current_offset += 1
        connectivity_lists.append(indices)

    nb_points = len(all_points)
    nb_polys = len(connectivity_lists)

    # Build the flat "connectivity" and "offsets" arrays expected by VTK
    connectivity_flat = []
    offsets = []
    cursor = 0
    for indices in connectivity_lists:
        connectivity_flat.extend(indices)
        cursor += len(indices)
        offsets.append(cursor)

    iterations = [interface["iteration"] for interface in interfaces]
    ids_a = [interface["id_a"] for interface in interfaces]
    ids_b = [interface["id_b"] for interface in interfaces]

    def format_floats(values, per_line=9):
        lines = []
        for i in range(0, len(values), per_line):
            chunk = values[i : i + per_line]
            lines.append(" ".join(f"{v:.6g}" for v in chunk))
        return "\n".join(lines)

    def format_ints(values, per_line=12):
        lines = []
        for i in range(0, len(values), per_line):
            chunk = values[i : i + per_line]
            lines.append(" ".join(str(v) for v in chunk))
        return "\n".join(lines)

    points_flat = []
    for x, y, z in all_points:
        points_flat.extend([x, y, z])

    with open(output_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write(
            '<VTKFile type="PolyData" version="1.0" '
            'byte_order="LittleEndian" header_type="UInt64">\n'
        )
        f.write("  <PolyData>\n")
        f.write(
            f'    <Piece NumberOfPoints="{nb_points}" '
            f'NumberOfVerts="0" NumberOfLines="0" '
            f'NumberOfStrips="0" NumberOfPolys="{nb_polys}">\n'
        )

        # --- Points ---
        f.write("      <Points>\n")
        f.write(
            '        <DataArray type="Float64" NumberOfComponents="3" '
            'format="ascii">\n'
        )
        f.write(format_floats(points_flat))
        f.write("\n        </DataArray>\n")
        f.write("      </Points>\n")

        # --- Polygons ---
        f.write("      <Polys>\n")
        f.write('        <DataArray type="Int64" Name="connectivity" format="ascii">\n')
        f.write(format_ints(connectivity_flat))
        f.write("\n        </DataArray>\n")
        f.write('        <DataArray type="Int64" Name="offsets" format="ascii">\n')
        f.write(format_ints(offsets))
        f.write("\n        </DataArray>\n")
        f.write("      </Polys>\n")

        # --- Per-cell data (one polygon = one broken interface) ---
        f.write("      <CellData>\n")

        f.write(
            '        <DataArray type="Int64" Name="iteration" format="ascii">\n'
        )
        f.write(format_ints(iterations))
        f.write("\n        </DataArray>\n")

        f.write('        <DataArray type="Int64" Name="id_a" format="ascii">\n')
        f.write(format_ints(ids_a))
        f.write("\n        </DataArray>\n")

        f.write('        <DataArray type="Int64" Name="id_b" format="ascii">\n')
        f.write(format_ints(ids_b))
        f.write("\n        </DataArray>\n")

        # Number of vertices per polygon (useful for filtering/coloring)
        nb_vertex_list = [len(c) for c in connectivity_lists]
        f.write(
            '        <DataArray type="Int64" Name="nb_vertex" format="ascii">\n'
        )
        f.write(format_ints(nb_vertex_list))
        f.write("\n        </DataArray>\n")

        f.write("      </CellData>\n")

        f.write("    </Piece>\n")
        f.write("  </PolyData>\n")
        f.write("</VTKFile>\n")


def main():
    if len(sys.argv) >= 2:
        input_path = sys.argv[1]
    else:
        input_path = "BrokenInterfaces.txt"

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        if input_path.lower().endswith(".txt"):
            output_path = input_path[:-4] + ".vtp"
        else:
            output_path = input_path + ".vtp"

    print(f"Reading: {input_path}")
    interfaces = read_interfaces(input_path)
    print(f"{len(interfaces)} interfaces (polygons) read.")

    if not interfaces:
        print("No valid interface found, aborting.")
        sys.exit(1)

    print(f"Writing ParaView file: {output_path}")
    write_vtp(interfaces, output_path)
    print("Done. Open this .vtp file in ParaView (File > Open).")


if __name__ == "__main__":
    main()
