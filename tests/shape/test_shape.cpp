/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#include <exaDEM/shape.hpp>
#include <exaDEM/shape/basic_shape/cube.hpp>
#include <exaDEM/shape/basic_shape/rice.hpp>
#include <exaDEM/shape/basic_shape/sphere.hpp>

#include <map>
#include <utility>
#include <vector>

#include "unit_test.hpp"

using exaDEM::shape;
using exaDEM::subBox;
using exanb::Quaternion;
using exanb::Vec3d;

namespace {
constexpr double tol = 1e-12;

// Unit right tetrahedron: (0,0,0), (1,0,0), (0,1,0), (0,0,1), outward oriented triangular faces.
shape make_tetrahedron(double radius = 0.0) {
  shape shp;
  shp.name_ = "tetra";
  shp.set_radius(radius);
  shp.volume_ = 1.0 / 6.0;
  shp.add_vertex({0, 0, 0});
  shp.add_vertex({1, 0, 0});
  shp.add_vertex({0, 1, 0});
  shp.add_vertex({0, 0, 1});
  shp.add_edge(0, 1);
  shp.add_edge(0, 2);
  shp.add_edge(0, 3);
  shp.add_edge(1, 2);
  shp.add_edge(1, 3);
  shp.add_edge(2, 3);
  shp.add_face({0, 2, 1});
  shp.add_face({0, 1, 3});
  shp.add_face({0, 3, 2});
  shp.add_face({1, 2, 3});
  shp.compute_offset_faces();
  shp.compute_face_areas();
  return shp;
}

// Newell's method: area-weighted normal of a planar polygon.
Vec3d face_normal(const shape& shp, int face_idx) {
  auto [ids, n] = shp.get_face(face_idx);
  Vec3d normal = {0, 0, 0};
  for (int i = 0; i < n; i++) {
    const Vec3d& a = shp.get_vertex(ids[i]);
    const Vec3d& b = shp.get_vertex(ids[(i + 1) % n]);
    normal.x += (a.y - b.y) * (a.z + b.z);
    normal.y += (a.z - b.z) * (a.x + b.x);
    normal.z += (a.x - b.x) * (a.y + b.y);
  }
  return normal;
}

Vec3d face_centroid(const shape& shp, int face_idx) {
  auto [ids, n] = shp.get_face(face_idx);
  Vec3d c = {0, 0, 0};
  for (int i = 0; i < n; i++) {
    c += shp.get_vertex(ids[i]);
  }
  return c / double(n);
}

bool contains(const OBB& obb, const Vec3d& p, double eps = 1e-10) {
  const Vec3d v = p - obb.center;
  return std::fabs(exanb::dot(v, obb.e1)) <= obb.extent.x + eps &&
         std::fabs(exanb::dot(v, obb.e2)) <= obb.extent.y + eps &&
         std::fabs(exanb::dot(v, obb.e3)) <= obb.extent.z + eps;
}

// Checks the polyhedron topology: closed, outward oriented, consistent edges.
void check_closed_polyhedron(const shape& shp) {
  const int nv = shp.get_number_of_vertices();
  const int ne = shp.get_number_of_edges();
  const int nf = shp.get_number_of_faces();
  CHECK(nv - ne + nf == 2);  // Euler characteristic of a convex polyhedron

  // each edge is shared by exactly two faces, traversed in opposite directions
  std::map<std::pair<int, int>, int> directed_edges;
  for (int f = 0; f < nf; f++) {
    auto [ids, n] = shp.get_face(f);
    for (int i = 0; i < n; i++) {
      directed_edges[{ids[i], ids[(i + 1) % n]}]++;
    }
  }
  for (int e = 0; e < ne; e++) {
    auto [a, b] = shp.get_edge(e);
    CHECK((directed_edges[{a, b}] == 1));
    CHECK((directed_edges[{b, a}] == 1));
  }
  CHECK(int(directed_edges.size()) == 2 * ne);  // no face edge missing from the edge list

  // faces are oriented outward (the shape is centered on its center of mass)
  for (int f = 0; f < nf; f++) {
    CHECK(exanb::dot(face_normal(shp, f), face_centroid(shp, f)) > 0);
  }
}
}  // namespace

// ============================================================
// === Construction
// ============================================================

UNIT_TEST(default_shape_is_empty) {
  shape shp;
  CHECK(shp.get_number_of_vertices() == 0);
  CHECK(shp.get_number_of_edges() == 0);
  CHECK(shp.get_number_of_faces() == 0);
  CHECK(shp.name_ == "undefined");
  CHECK_NEAR(shp.minkowski(), 0.0, 0.0);
  CHECK_NEAR(shp.volume_, 0.0, 0.0);
}

UNIT_TEST(add_vertices_and_edges) {
  shape shp;
  shp.add_vertex({1, 2, 3});
  shp.add_vertex({4, 5, 6});
  shp.add_edge(0, 1);
  CHECK(shp.get_number_of_vertices() == 2);
  CHECK(shp.get_number_of_edges() == 1);
  CHECK_VEC3_NEAR(shp.get_vertex(1), (Vec3d{4, 5, 6}), 0.0);
  auto [a, b] = shp.get_edge(0);
  CHECK(a == 0 && b == 1);
}

UNIT_TEST(add_faces_of_different_sizes) {
  // faces_ layout: [n_faces, n0, ids..., n1, ids..., ...] and offset_faces_ indexes each face
  shape shp;
  for (int i = 0; i < 6; i++) {
    shp.add_vertex({double(i), 0, 0});
  }
  int quad[4] = {0, 1, 2, 3};
  shp.add_face({0, 1, 2});  // initializer_list overload
  shp.add_face(4, quad);    // pointer overload
  shp.add_face({1, 2, 3, 4, 5});
  shp.compute_offset_faces();

  CHECK(shp.get_number_of_faces() == 3);
  const int expected_sizes[3] = {3, 4, 5};
  const std::vector<std::vector<int>> expected_ids = {{0, 1, 2}, {0, 1, 2, 3}, {1, 2, 3, 4, 5}};
  for (int f = 0; f < 3; f++) {
    auto [ids, n] = shp.get_face(f);
    CHECK(n == expected_sizes[f]);
    CHECK(std::vector<int>(ids, ids + n) == expected_ids[f]);
  }
}

UNIT_TEST(clear) {
  shape shp = make_tetrahedron();
  shp.clear();
  CHECK(shp.get_number_of_vertices() == 0);
  CHECK(shp.get_number_of_edges() == 0);
  CHECK(shp.get_number_of_faces() == 0);
  CHECK(shp.name_ == "undefined");
  shp.add_face({0, 1, 2});  // faces_ is still usable after clear
  CHECK(shp.get_number_of_faces() == 1);
}

UNIT_TEST(visitors) {
  shape shp = make_tetrahedron();
  int nv = 0, ne = 0, nf = 0, n_face_vertices = 0;
  auto count_v = [&nv](Vec3d&) { nv++; };
  auto count_e = [&ne](int a, int b) { ne += (a != b); };
  auto count_f = [&nf, &n_face_vertices](int n, int*) {
    nf++;
    n_face_vertices += n;
  };
  shp.for_all_vertices(count_v);
  shp.for_all_edges(count_e);
  shp.for_all_faces(count_f);
  CHECK(nv == 4);
  CHECK(ne == 6);
  CHECK(nf == 4);
  CHECK(n_face_vertices == 12);

  // extra arguments are forwarded
  auto translate = [](Vec3d& v, const Vec3d& t) { v += t; };
  shp.for_all_vertices(translate, Vec3d{1, 1, 1});
  CHECK_VEC3_NEAR(shp.get_vertex(0), (Vec3d{1, 1, 1}), 0.0);
}

// ============================================================
// === Geometry
// ============================================================

UNIT_TEST(transformed_vertex) {
  shape shp = make_tetrahedron();
  const double s = std::sqrt(0.5);
  const Quaternion rotz90{s, 0, 0, s};
  // p + Q * (h * v)
  CHECK_VEC3_NEAR(shp.get_vertex(1, Vec3d{10, 0, 0}, 2.0, rotz90), (Vec3d{10, 2, 0}), tol);
  CHECK_VEC3_NEAR(shp.get_vertex(3, Vec3d{0, 0, 0}, 3.0, Quaternion{1, 0, 0, 0}), (Vec3d{0, 0, 3}), tol);
}

UNIT_TEST(volume_density_and_inertia_with_homothety) {
  shape shp = make_tetrahedron();
  shp.inertia_on_mass_ = {1, 2, 3};
  CHECK_NEAR(shp.compute_volume(), 1.0 / 6.0, tol);
  CHECK_NEAR(shp.compute_volume(2.0), 8.0 / 6.0, tol);  // h^3
  CHECK_NEAR(shp.compute_density(2.0), 12.0, tol);      // mass / volume
  CHECK_NEAR(shp.compute_density(8.0 / 6.0, 2.0), 1.0, tol);
  CHECK_VEC3_NEAR(shp.compute_Im(3.0), (Vec3d{9, 18, 27}), tol);  // h^2
}

UNIT_TEST(minkowski_and_rcut) {
  shape shp = make_tetrahedron(0.1);
  CHECK_NEAR(shp.minkowski(), 0.1, 0.0);
  CHECK_NEAR(shp.minkowski(3.0), 0.3, tol);
  CHECK_NEAR(shp.compute_max_rcut(), 1.1, tol);     // farthest vertex + radius
  CHECK_NEAR(shp.compute_max_rcut(2.0), 2.2, tol);  // the radius is scaled too
}

UNIT_TEST(face_areas) {
  shape shp = make_tetrahedron();
  CHECK_NEAR(shp.get_face_area(0), 0.5, tol);
  CHECK_NEAR(shp.get_face_area(1), 0.5, tol);
  CHECK_NEAR(shp.get_face_area(2), 0.5, tol);
  CHECK_NEAR(shp.get_face_area(3), std::sqrt(3.0) / 2.0, tol);       // equilateral triangle of side sqrt(2)
  CHECK_NEAR(shp.get_face_area(3, 2.0), 2.0 * std::sqrt(3.0), tol);  // h^2
}

UNIT_TEST(face_areas_polygons) {
  const shape cube = exaDEM::basic_shape::create_cube("cube", 2.0, 0.0);
  for (int f = 0; f < cube.get_number_of_faces(); f++) {
    CHECK_NEAR(cube.get_face_area(f), 4.0, tol);
  }
}

UNIT_TEST(surface_triangles) {
  const shape shp = make_tetrahedron();
  CHECK_NEAR(shp.compute_surface(), 1.5 + std::sqrt(3.0) / 2.0, tol);
}

UNIT_TEST(surface_polygons) {
  // compute_surface must agree with compute_face_areas for non triangular faces
  const shape cube = exaDEM::basic_shape::create_cube("cube", 2.0, 0.0);
  CHECK_NEAR(cube.compute_surface(), 24.0, tol);
}

UNIT_TEST(shift_vertices) {
  shape shp = make_tetrahedron();
  shp.shift_vertices({1, 2, 3});
  CHECK_VEC3_NEAR(shp.get_vertex(0), (Vec3d{-1, -2, -3}), 0.0);  // the shift is subtracted
  CHECK_VEC3_NEAR(shp.get_vertex(1), (Vec3d{0, -2, -3}), 0.0);
}

UNIT_TEST(rescale) {
  shape shp = exaDEM::basic_shape::create_cube("cube", 1.0, 0.1);
  const double volume = shp.volume_;
  const Vec3d Im = shp.inertia_on_mass_;

  shp.rescale(2.0, true);
  CHECK_VEC3_NEAR(shp.get_vertex(6), (Vec3d{1, 1, 1}), tol);
  CHECK_NEAR(shp.minkowski(), 0.2, tol);
  CHECK_NEAR(shp.volume_, 8.0 * volume, 1e-10);
  CHECK_VEC3_NEAR(shp.inertia_on_mass_, 4.0 * Im, 1e-10);
  CHECK_VEC3_NEAR(shp.obb_.extent, (Vec3d{1.2, 1.2, 1.2}), 1e-10);  // OBB rebuilt with the new radius

  shape shp2 = exaDEM::basic_shape::create_cube("cube", 1.0, 0.1);
  shp2.rescale(2.0, false);
  CHECK_NEAR(shp2.minkowski(), 0.1, 0.0);  // the Minkowski radius is kept
  CHECK_VEC3_NEAR(shp2.obb_.extent, (Vec3d{1.1, 1.1, 1.1}), 1e-10);
}

UNIT_TEST(deform) {
  shape shp = exaDEM::basic_shape::create_cube("cube", 1.0, 0.0);
  shp.deform({2, 3, 4});
  CHECK_VEC3_NEAR(shp.get_vertex(6), (Vec3d{1, 1.5, 2}), tol);
  CHECK_NEAR(shp.volume_, 24.0, 1e-10);
  // sorted extents of the rebuilt OBB
  std::vector<double> ext = {shp.obb_.extent.x, shp.obb_.extent.y, shp.obb_.extent.z};
  std::sort(ext.begin(), ext.end());
  CHECK_NEAR(ext[0], 1.0, 1e-10);
  CHECK_NEAR(ext[1], 1.5, 1e-10);
  CHECK_NEAR(ext[2], 2.0, 1e-10);
}

UNIT_TEST(identify_face) {
  const shape cube = exaDEM::basic_shape::create_cube("cube", 1.0, 0.0);
  for (int f = 0; f < cube.get_number_of_faces(); f++) {
    auto [ids, n] = cube.get_face(f);
    std::vector<int> v(ids, ids + n);
    std::reverse(v.begin(), v.end());  // any permutation
    CHECK(cube.identify_face(v) == f);
  }
  // partial match: 3 vertices of the top face {4, 5, 6, 7}
  std::vector<int> partial = {6, 4, 5};
  CHECK(cube.identify_face(partial) == 1);
}

// ============================================================
// === OBBs
// ============================================================

UNIT_TEST(build_obb_from_shape) {
  const shape shp = make_tetrahedron(0.05);
  const OBB obb = exaDEM::build_obb_from_shape(shp);
  for (int i = 0; i < shp.get_number_of_vertices(); i++) {
    CHECK(contains(obb, shp.get_vertex(i)));
  }
  CHECK(std::min({obb.extent.x, obb.extent.y, obb.extent.z}) >= 0.05);  // includes the Minkowski radius
}

UNIT_TEST(prepro_obb_follow_the_particle) {
  shape shp = exaDEM::basic_shape::create_cube("cube", 1.0, 0.1);
  const Vec3d center = {5, -2, 1};
  const double s = std::sqrt(0.5);
  const Quaternion Q{s, s, 0, 0};  // 90 degrees around x
  std::vector<Vec3d> scratch(shp.get_number_of_vertices());
  shp.compute_prepro_obb(scratch.data(), center, Q);

  CHECK(int(shp.obb_vertices_.size()) == shp.get_number_of_vertices());
  CHECK(int(shp.obb_edges_.size()) == shp.get_number_of_edges());
  CHECK(int(shp.obb_faces_.size()) == shp.get_number_of_faces());

  for (int i = 0; i < shp.get_number_of_vertices(); i++) {
    const Vec3d v = shp.get_vertex(i, center, 1.0, Q);
    CHECK_VEC3_NEAR(scratch[i], v, tol);
    CHECK_VEC3_NEAR(shp.obb_vertices_[i].center, v, tol);
    CHECK(shp.obb_vertices_[i].extent.x >= shp.minkowski());
  }
  for (int e = 0; e < shp.get_number_of_edges(); e++) {
    auto [a, b] = shp.get_edge(e);
    CHECK(contains(shp.obb_edges_[e], scratch[a]));
    CHECK(contains(shp.obb_edges_[e], scratch[b]));
  }
  for (int f = 0; f < shp.get_number_of_faces(); f++) {
    auto [ids, n] = shp.get_face(f);
    for (int i = 0; i < n; i++) {
      CHECK(contains(shp.obb_faces_[f], scratch[ids[i]]));
    }
  }
}

UNIT_TEST(obb_edge_and_face_are_moved) {
  // OBBs precomputed at the origin are rotated and translated on demand
  const shape shp = exaDEM::basic_shape::create_cube("cube", 1.0, 0.0);
  const Vec3d position = {1, 2, 3};
  const double s = std::sqrt(0.5);
  const Quaternion Q{s, 0, 0, s};
  for (int e = 0; e < shp.get_number_of_edges(); e++) {
    const OBB obb = shp.get_obb_edge(position, e, Q);
    auto [a, b] = shp.get_edge(e);
    CHECK(contains(obb, shp.get_vertex(a, position, 1.0, Q)));
    CHECK(contains(obb, shp.get_vertex(b, position, 1.0, Q)));
  }
  for (int f = 0; f < shp.get_number_of_faces(); f++) {
    const OBB obb = shp.get_obb_face(position, f, Q);
    auto [ids, n] = shp.get_face(f);
    for (int i = 0; i < n; i++) {
      CHECK(contains(obb, shp.get_vertex(ids[i], position, 1.0, Q)));
    }
  }
}

UNIT_TEST(build_obb_tree) {
  shape shp = exaDEM::basic_shape::create_cube("cube", 1.0, 0.1);
  shp.buildOBBtree();
  CHECK(shp.obbtree_.root != nullptr);

  // one leaf per face, edge and vertex
  std::vector<subBox> leaves;
  std::vector<const OBBnode<subBox>*> stack = {shp.obbtree_.root};
  while (!stack.empty()) {
    const OBBnode<subBox>* node = stack.back();
    stack.pop_back();
    if (node->isLeaf()) {
      leaves.push_back(node->data);
    } else {
      stack.push_back(node->first);
      stack.push_back(node->second);
    }
  }
  CHECK(int(leaves.size()) == 6 + 12 + 8);
  int n_faces = 0, n_edges = 0, n_vertices = 0;
  for (const auto& leaf : leaves) {
    n_faces += (leaf.nbPoints == 4);
    n_edges += (leaf.nbPoints == 2);
    n_vertices += (leaf.nbPoints == 1);
  }
  CHECK(n_faces == 6);
  CHECK(n_edges == 12);
  CHECK(n_vertices == 8);

  // the root contains the whole shape
  for (int i = 0; i < shp.get_number_of_vertices(); i++) {
    CHECK(contains(shp.obbtree_.root->boundary, shp.get_vertex(i)));
  }

  // rebuilding the tree does not leak or keep the previous nodes
  shp.buildOBBtree();
  CHECK(shp.obbtree_.root != nullptr);
}

// ============================================================
// === Basic shapes
// ============================================================

UNIT_TEST(cube_topology) {
  const shape cube = exaDEM::basic_shape::create_cube("cube", 1.5, 0.1);
  CHECK(cube.name_ == "cube");
  CHECK(cube.get_number_of_vertices() == 8);
  CHECK(cube.get_number_of_edges() == 12);
  CHECK(cube.get_number_of_faces() == 6);
  check_closed_polyhedron(cube);
  CHECK_VEC3_NEAR(cube.get_vertex(6), (Vec3d{0.75, 0.75, 0.75}), tol);
}

UNIT_TEST(cube_volume_and_inertia) {
  // without rounding: V = c^3 and I/m = c^2 / 6
  const shape cube = exaDEM::basic_shape::create_cube("cube", 2.0, 0.0);
  CHECK_NEAR(cube.volume_, 8.0, tol);
  CHECK_VEC3_NEAR(cube.inertia_on_mass_, (Vec3d{4.0 / 6.0, 4.0 / 6.0, 4.0 / 6.0}), tol);

  // Steiner formula for the Minkowski sum of a cube and a ball
  const double c = 1.0, r = 0.1;
  const shape rounded = exaDEM::basic_shape::create_cube("cube", c, r);
  CHECK_NEAR(rounded.volume_, c * c * c + 6 * c * c * r + 3 * M_PI * r * r * c + 4.0 / 3.0 * M_PI * r * r * r, tol);
  CHECK_VEC3_NEAR(rounded.obb_.extent, (Vec3d{0.6, 0.6, 0.6}), 1e-10);
}

UNIT_TEST(tetrahedron_topology) {
  shape shp = make_tetrahedron();
  shp.shift_vertices({0.25, 0.25, 0.25});  // recentered on its center of mass for the orientation check
  check_closed_polyhedron(shp);
}

UNIT_TEST(sphere_volume_and_inertia) {
  // a sphere is a single vertex with a Minkowski radius r: V = 4/3 pi r^3, I/m = 2/5 r^2
  const double r = 0.5;
  const shape sphere = exaDEM::basic_shape::create_sphere("sphere", r);
  CHECK(sphere.get_number_of_vertices() == 1);
  CHECK(sphere.get_number_of_edges() == 0);
  CHECK(sphere.get_number_of_faces() == 0);
  CHECK_NEAR(sphere.minkowski(), r, 0.0);
  CHECK_NEAR(sphere.volume_, 4.0 / 3.0 * M_PI * r * r * r, tol);
  CHECK_VEC3_NEAR(sphere.inertia_on_mass_, (Vec3d{0.4 * r * r, 0.4 * r * r, 0.4 * r * r}), tol);
}

UNIT_TEST(rice) {
  const double c = 2.0, r = 0.5;
  const shape rice = exaDEM::basic_shape::create_rice("rice", c, r);
  CHECK(rice.get_number_of_vertices() == 2);
  CHECK(rice.get_number_of_edges() == 1);
  CHECK_VEC3_NEAR(rice.get_vertex(0), (Vec3d{-1, 0, 0}), tol);
  CHECK_VEC3_NEAR(rice.get_vertex(1), (Vec3d{1, 0, 0}), tol);
  CHECK_NEAR(rice.volume_, M_PI * r * r * c + 4.0 / 3.0 * M_PI * r * r * r, tol);
  CHECK(rice.inertia_on_mass_.x < rice.inertia_on_mass_.y);  // easier to roll than to tumble
  CHECK_NEAR(rice.inertia_on_mass_.y, rice.inertia_on_mass_.z, 0.0);
}

UNIT_TEST(rice_without_length_is_a_sphere) {
  const double r = 0.5;
  const shape rice = exaDEM::basic_shape::create_rice("rice", 0.0, r);
  const shape sphere = exaDEM::basic_shape::create_sphere("sphere", r);
  CHECK_NEAR(rice.volume_, sphere.volume_, tol);
  CHECK_VEC3_NEAR(rice.inertia_on_mass_, sphere.inertia_on_mass_, tol);
}

UNIT_TEST_MAIN()
