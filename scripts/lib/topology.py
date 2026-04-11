import numpy as np
from collections import defaultdict
from .geometry import face_area, face_normal
from data_class import Shapes, Shape, CellsData


def count_interfaces(polyhedra):
    """
    Count the number of interfaces between cells in a topology.

    Parameters
    ----------
    polyhedra : dict
        mapping polyhedron ID to its face indices [fid0, fid1, ...]

    Returns
    ----------
    n_interface : int
       number of interfaces
    """

    face_count = defaultdict(int)
    for fids in polyhedra.values():
        for fid in fids:
            face_count[fid] += 1

    n_interface = sum(1 for c in face_count.values() if c == 2)
    return n_interface


def get_interfaces(polyhedra):
    """
    Return a list of interfaces between cells in a topology, defined as faces that are shared by exactly two cells.

    Parameters
    ----------
    polyhedra : dict
        mapping polyhedron ID to its face indices [fid0, fid1, ...]

    Returns
    ----------
    interfaces : list
        tuples (fid, cell1, cell2) where fid is the face ID of the interface, and cell1 and cell2 are the IDs of the two cells sharing the interface.
    """
    face_to_cells = defaultdict(list)
    for pid, fids in polyhedra.items():
        for fid in fids:
            face_to_cells[fid].append(pid)
    return [(fid, c[0], c[1]) for fid, c in face_to_cells.items() if len(c) == 2]


def check_interface(face_i, verts_i, face_j, verts_j):
    """
    Check if two faces (face_i and face_j) from two different cells represent the same interface, by comparing their geometry (area, normal, number of vertices).

    Parameters
    ----------
    face_i : list
        vertex indices of the first face
    verts_i : array of shape (n_vertices_i, 3)
        coordinates of the vertices of the first cell
    face_j : list
        vertex indices of the second face
    verts_j : array of shape (n_vertices_j, 3)
        coordinates of the vertices of the second cell

    Returns
    ----------
    ok: bool
        True if the faces represent the same interface, False otherwise
    reason: str
        reason for failure if ok is False ("nb_vertices", "area", "normal")
        Criteria for matching:
         - number of vertices must be the same
         - area must be within 1% of each other
         - normals must be parallel (cross product close to zero)
    """
    if len(face_i) != len(face_j):
        return False, "nb_vertices"

    Ai = face_area(verts_i, face_i)
    Aj = face_area(verts_j, face_j)

    if not (0.99 < Ai / Aj < 1.01):
        return False, "area"

    ni = face_normal(verts_i, face_i)
    nj = face_normal(verts_j, face_j)

    if np.linalg.norm(np.cross(ni, nj)) > 1e-10:
        return False, "normal"

    return True, "ok"


# =========================
# CHECK INTERFACES NEPER
# ---------------------------------------------------------------------------
def check_interfaces(cell_shapes: Shapes, polyhedra: dict):
    """
    Check that the interfaces defined by Neper (from the polyhedra definitions) are correctly represented in the computed cell shapes.
    For each interface (face shared by two cells), we check if there are corresponding faces in the shapes of the two cells that match in terms of geometry (area, normal, number of vertices).

    Parameters
    ----------
    cell_shapes : Shapes
        data class containing the geometric data of the cells (vertices, faces, edges, volume, inertia tensor)
    polyhedra : dict
        mapping polyhedron ID to its face indices [face_id0, face_id1, ...] (from Neper)

    Returns
    ----------
    None
    """
    interfaces = get_interfaces(polyhedra)

    missing = []

    for fid, c1, c2 in interfaces:
        if c1 not in cell_shapes or c2 not in cell_shapes:
            continue

        shape1 = cell_shapes[c1]
        shape2 = cell_shapes[c2]

        v1, f1 = shape1.vertices, shape1.faces
        v2, f2 = shape2.vertices, shape2.faces

        found = False

        for face_i in f1:
            for face_j in f2:
                ok, _ = check_interface(face_i, v1, face_j, v2)
                if ok:
                    found = True
                    break
            if found:
                break

        if not found:
            best = None
            best_data = None

            for face_i in f1:
                for face_j in f2:
                    Ai = face_area(v1, face_i)
                    Aj = face_area(v2, face_j)
                    rel_diff = abs(Ai - Aj) / max(Ai, Aj)

                    if best is None or rel_diff < best:
                        best = rel_diff
                        best_data = (face_i, face_j, Ai, Aj)

            face_i, face_j, Ai, Aj = best_data

            ni = face_normal(v1, face_i)
            nj = face_normal(v2, face_j)
            normal_error = np.linalg.norm(np.cross(ni, nj))

            # seuils (comme dans check_interface)
            AREA_TOL = 0.01
            NORMAL_TOL = 1e-10

            print(f"\n[INTERFACE {fid}] cellules {c1}-{c2}")
            print(f"  Ai={Ai:.6e} Aj={Aj:.6e}")
            print(f"  rel_diff={best:.6f} (crit < {AREA_TOL})")
            print(f"  nb_vertices={len(face_i)} vs {len(face_j)} (crit = égalité)")
            print(f"  normal_error={normal_error:.3e} (crit < {NORMAL_TOL})")

            fail = []

            if len(face_i) != len(face_j):
                fail.append("nb_vertices")

            if best > 0.01:
                fail.append("area")

            if normal_error > 1e-10:
                fail.append("normal")

            print(f"  => FAIL: {fail}")

            missing.append((fid, c1, c2, fail))

    print("\n=== CHECK INTERFACES ===")
    print(f"Interfaces Neper       : {len(interfaces)}")
    print(f"Interfaces retrouvées  : {len(interfaces)-len(missing)}")
    print(f"Interfaces perdues     : {len(missing)}")

    for m in missing[:20]:
        print("Missing:", m)
