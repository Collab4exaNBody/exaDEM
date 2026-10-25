# Unit tests — `shape` plugin

Unit tests for the `shape` plugin (`src/shape`): oriented bounding boxes, OBB trees, `shape`, `shapes` and basic shapes.
Each `test_<name>.cpp` file builds one executable, registered in CTest as `UnitTest_shape_<name>`.

```bash
ctest -R UnitTest_shape --output-on-failure    # all shape tests
./exaDEM_test_shape_obb                         # one executable, prints [  OK  ] / [FAILED] per test
```

To add a file, create `test_<name>.cpp` (see `tests/common/unit_test.hpp`) and add `exadem_add_unit_test(shape <name>)`
to `CMakeLists.txt`.

## `test_obb.cpp` — `OBB` (`exaDEM/type/OBB.hpp`)

| Test | Checks |
|---|---|
| **Basic operations** | |
| `default_constructor` | Center at the origin, identity axes, null extents |
| `enlarge` | Extents are increased, center unchanged |
| `translate` | Center is moved, axes and extents unchanged |
| `rotate_in_place_rotates_axes_and_center` | 90° rotation around z of the axes **and** of the center |
| `rotate_in_place_keeps_orthonormal_axes` | Axes stay orthonormal after random rotations |
| **Point inclusion** | |
| `intersect_point_axis_aligned` | Inside, on a face (inclusive), just outside along each axis |
| `intersect_point_rotated` | Cube rotated by 45° |
| **OBB / OBB intersection** | |
| `intersect_obb_axis_aligned` | Overlap, containment, separation along x / y / z |
| `intersect_obb_touching_faces` | Boxes sharing a face intersect |
| `intersect_obb_rotated` | Rotated box on both sides of the contact |
| `intersect_obb_is_symmetric` | `A.intersect(B) == B.intersect(A)` on random boxes |
| `intersect_obb_matches_reference_sat` | 20,000 random configurations compared to an independent 15-axis separating axis test |
| `intersect_obb_separated_by_edge_edge_axis` | Two crossed rods only separated by an edge × edge axis |
| `intersect_obb_enlarge_creates_contact` | `enlarge` turns a gap into a contact |
| **AABB conversions** | |
| `conv_to_obb` | Center, extents and axes of the box |
| `conv_to_aabb_round_trip` | AABB → OBB → AABB is the identity |
| `conv_to_aabb_rotated` | AABB of a cube rotated by 45° |
| `conv_to_aabb_contains_obb_corners` | The AABB contains the 8 corners of random OBBs |

## `test_obb_tree.cpp` — `OBBtree` (`exaDEM/type/OBBtree.hpp`)

| Test | Checks |
|---|---|
| **Covariance matrix and OBB fitting** | |
| `covariance_matrix` | Known values (sum of `p·pᵀ`, not the mean) |
| `covariance_matrix_is_symmetric` | Symmetric matrix |
| `fit_obb_axis_aligned_cuboid` | Center, axes sorted by increasing variance, extents |
| `fit_obb_with_radius` | The Minkowski radius is added to the extents |
| `fit_obb_rotated_cuboid` | Recovers orientation and extents of a rotated cuboid |
| **Tree construction** | |
| `recursive_build_structure` | 8 leaves, each one once, balanced tree of depth 4 |
| `recursive_build_boundaries_contain_points` | Each node contains all points of its subtree |
| `recursive_build_single_leaf` | One bundle gives a single leaf with its data |
| `recursive_build_empty` | No bundle gives a null root |
| **Memory management** | |
| `copy_is_deep` | Copied nodes are distinct and survive the destruction of the original |
| `copy_assignment` | Assignment and self-assignment |
| `move` | Move constructor and move assignment transfer the root |
| **Queries** | |
| `obb_intersection_ids` | Leaves intersected by a query box (one, several, all, none, null tree) |
| `obb_intersection_ids_subset_of_brute_force` | Every reported leaf really intersects the query |
| `tree_intersection_ids` | Tree / tree pairs: same position, translation, far away, `enlarge` value |
| `tree_intersection_ids_rotation` | Tree B rotated by 90° |
| `tree_intersection_ids_scale` | Tree B scaled by 2 |
| `tree_intersection_ids_null` | Null trees |

## `test_shape.cpp` — `shape` (`exaDEM/shape.hpp`, `shape_prepro.hpp`, `shape/basic_shape/*.hpp`)

| Test | Checks |
|---|---|
| **Construction** | |
| `default_shape_is_empty` | No vertex, edge or face, name `"undefined"` |
| `add_vertices_and_edges` | Counts, `get_vertex`, `get_edge` |
| `add_faces_of_different_sizes` | Both `add_face` overloads, faces of 3/4/5 vertices, `compute_offset_faces` |
| `clear` | Empty shape, still usable afterwards |
| `visitors` | `for_all_vertices/edges/faces`, extra arguments forwarded |
| **Geometry** | |
| `transformed_vertex` | `get_vertex(i, p, h, Q) = p + Q·(h·v)` |
| `volume_density_and_inertia_with_homothety` | Volume in `h³`, density, `I/m` in `h²` |
| `minkowski_and_rcut` | `minkowski(h)`, `compute_max_rcut(h)` |
| `face_areas` | Triangle areas, `h²` scaling |
| `face_areas_polygons` | Quadrangular faces of a cube |
| `surface_triangles` | Total surface of a tetrahedron |
| `surface_polygons` | Total surface of a cube (quadrangular faces) |
| `shift_vertices` | The shift vector is **subtracted** |
| `rescale` | Vertices, volume, `I/m`, rebuilt OBB, with and without Minkowski rescaling |
| `deform` | Anisotropic deformation: vertices, volume, OBB |
| `identify_face` | Exact match (any permutation) and partial match |
| **OBBs** | |
| `build_obb_from_shape` | Contains all vertices, includes the Minkowski radius |
| `prepro_obb_follow_the_particle` | `compute_prepro_obb` with a rotated and translated particle |
| `obb_edge_and_face_are_moved` | `get_obb_edge` / `get_obb_face` contain the moved vertices |
| `build_obb_tree` | 26 leaves for a cube (6 faces, 12 edges, 8 vertices), root contains the shape, rebuild |
| **Basic shapes** | |
| `cube_topology` | Euler characteristic, each edge shared by 2 faces in opposite directions, outward faces |
| `cube_volume_and_inertia` | `c³` and `c²/6` without rounding, Steiner formula with rounding |
| `tetrahedron_topology` | Same topology checks on the test tetrahedron |
| `sphere_volume_and_inertia` | `V = 4/3 π r³`, `I/m = 2/5 r²` |
| `rice` | Vertices, volume, rolling inertia lower than tumbling inertia |
| `rice_without_length_is_a_sphere` | A rice of length 0 has the volume and inertia of a sphere |

## `test_shapes.cpp` — `shapes` (`exaDEM/shapes.hpp`)

| Test | Checks |
|---|---|
| **Container** | |
| `empty_container` | Size 0, no OBB tree, unknown name gives `nullptr` |
| `add_shape_and_access` | Both `add_shape` overloads, access by index (const and mutable), `data()`, `view()` |
| `max_number_of_vertices` | Updated on insertion, never decreases |
| `access_by_name` | Access by name, `nullptr` if not found |
| `contains` | Lookup by name |
| `add_shape_copies` | The container stores an independent copy |
| `enable_obb_tree` | Flag set |
| **Copies of shapes with an OBB tree** | |
| `obb_tree_is_deep_copied` | The stored tree survives the destruction of the original shape |
| `obb_tree_survives_reallocations` | 20 insertions (reallocations) and copy of the whole container (former double free) |
| **Registration in the particle type map** | |
| `register_shape` | Indexes in the `ParticleTypeMap` |
| `register_shapes_renames_duplicates` | A duplicated name is renamed with an `X` suffix |

## Not tested

Error paths that call `color_log::error` (which calls `std::exit`): unknown face in `identify_face`, name already taken
in `register_shape`, face with less than 3 vertices in `compute_face_areas`.
