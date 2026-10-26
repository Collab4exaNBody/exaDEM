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
// RShapeDriver tested with a planar mesh made of 4 rectangles (1 x 0.5) in the plane z = 0:
//
//   y = 0.5   6 ----- 7 ----- 8
//             |  f2   |  f3   |
//   y = 0     3 ----- 4 ----- 5
//             |  f0   |  f1   |
//   y = -0.5  0 ----- 1 ----- 2
//           x = -1   x = 0   x = 1
#include "driver_test.hpp"

#include <exaDEM/shape_detection_core.hpp>

#include <filesystem>
#include <map>
#include <utility>

using namespace exaDEM;
using exaDEM::unit_test::contains;
using exanb::Quaternion;
using exanb::Vec3d;

namespace {
constexpr double tol = 1e-12;
constexpr double minkowski = 0.01;

shape make_mesh(const std::string& name = "mesh4") {
  shape shp;
  shp.name_ = name;
  shp.set_radius(minkowski);
  shp.inertia_on_mass_ = {1, 1, 1};
  for (double y : {-0.5, 0.0, 0.5}) {
    for (double x : {-1.0, 0.0, 1.0}) {
      shp.add_vertex({x, y, 0});
    }
  }
  for (int row : {0, 3, 6}) {  // edges along x
    shp.add_edge(row, row + 1);
    shp.add_edge(row + 1, row + 2);
  }
  for (int col : {0, 1, 2}) {  // edges along y
    shp.add_edge(col, col + 3);
    shp.add_edge(col + 3, col + 6);
  }
  shp.add_face({0, 1, 4, 3});
  shp.add_face({1, 2, 5, 4});
  shp.add_face({3, 4, 7, 6});
  shp.add_face({4, 5, 8, 7});
  shp.compute_offset_faces();
  shp.compute_face_areas();
  shp.obb_ = build_obb_from_shape(shp);
  shp.pre_compute_obb_edges(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
  shp.pre_compute_obb_faces(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
  return shp;
}

RShapeDriver make_driver(MotionType motion_type = STATIONARY, Driver_params motion = {},
                         const Vec3d& center = {0, 0, 0}) {
  RShapeDriver drv;
  drv.shp_ = make_mesh();
  drv.motion_type_ = motion_type;
  drv.fields_.center_ = center;
  drv.initialize(motion);
  return drv;
}

// Faces touched by a vertex of radius r.
std::vector<int> touched_faces(const RShapeDriver& drv, const Vec3d& p, double r) {
  std::vector<int> faces;
  for (int f = 0; f < drv.shp_.get_number_of_faces(); f++) {
    contact c = detection_vertex_face(p, r, drv.fields_.center_, 1.0, f, &drv.shp_, drv.fields_.quat_);
    if (c.is_contact) {
      faces.push_back(f);
    }
  }
  return faces;
}
}  // namespace

// ============================================================
// === Mesh
// ============================================================

UNIT_TEST(mesh_topology) {
  const shape shp = make_mesh();
  CHECK(shp.get_number_of_vertices() == 9);
  CHECK(shp.get_number_of_edges() == 12);
  CHECK(shp.get_number_of_faces() == 4);
  CHECK(shp.get_number_of_vertices() - shp.get_number_of_edges() + shp.get_number_of_faces() == 1);  // a disk

  // 4 interior edges shared by 2 faces, 8 boundary edges
  std::map<std::pair<int, int>, int> count;
  for (int f = 0; f < 4; f++) {
    auto [ids, n] = shp.get_face(f);
    CHECK(n == 4);
    for (int i = 0; i < n; i++) {
      count[std::minmax(ids[i], ids[(i + 1) % n])]++;
    }
  }
  int interior = 0, boundary = 0;
  for (int e = 0; e < 12; e++) {
    auto [a, b] = shp.get_edge(e);
    const int c = count[std::minmax(a, b)];
    interior += (c == 2);
    boundary += (c == 1);
  }
  CHECK(interior == 4);
  CHECK(boundary == 8);
}

UNIT_TEST(mesh_areas) {
  const shape shp = make_mesh();
  double total = 0;
  for (int f = 0; f < 4; f++) {
    CHECK_NEAR(shp.get_face_area(f), 0.5, tol);
    total += shp.get_face_area(f);
  }
  CHECK_NEAR(total, 2.0, tol);
  CHECK_NEAR(shp.compute_surface(), 2.0, tol);  // used by RShapeDriver::initialize in compressive mode
}

// ============================================================
// === Initialization and vertices
// ============================================================

UNIT_TEST(initialize_computes_vertices) {
  RShapeDriver drv = make_driver(STATIONARY, {}, Vec3d{1, 2, 3});
  CHECK(int(drv.vertices_.size()) == 9);
  CHECK_VEC3_NEAR(drv.vertices_[0], (Vec3d{0, 1.5, 3}), tol);
  CHECK_VEC3_NEAR(drv.vertices_[8], (Vec3d{2, 2.5, 3}), tol);
  CHECK(drv.stationary());
  CHECK(!drv.need_moment());
}

UNIT_TEST(initialize_removes_the_path_of_the_name) {
  RShapeDriver drv;
  drv.shp_ = make_mesh("some/directory/mesh4.shp");
  drv.motion_type_ = STATIONARY;
  Driver_params motion;
  drv.initialize(motion);
  CHECK(drv.shp_.name_ == "mesh4.shp");
}

UNIT_TEST(update_vertex_with_orientation) {
  RShapeDriver drv = make_driver();
  const double s = std::sqrt(0.5);
  drv.fields_.quat_ = {s, 0, 0, s};  // 90 degrees around z
  drv.fields_.center_ = {10, 0, 0};
  drv.update_vertex(2);
  CHECK_VEC3_NEAR(drv.vertices_[2], (Vec3d{10.5, 1, 0}), tol);  // (1, -0.5, 0) -> (0.5, 1, 0)
}

// ============================================================
// === Contact detection with the mesh
// ============================================================

UNIT_TEST(vertex_face_contact_above) {
  RShapeDriver drv = make_driver();
  const Vec3d p = {0.3, 0.2, 0.05};  // above f3
  CHECK(touched_faces(drv, p, 0.1) == std::vector<int>{3});
  contact c = detection_vertex_face(p, 0.1, drv.fields_.center_, 1.0, 3, &drv.shp_, drv.fields_.quat_);
  CHECK_NEAR(c.dn, 0.05 - 0.1 - minkowski, tol);
  CHECK_VEC3_NEAR(c.normal, (Vec3d{0, 0, 1}), tol);  // from the mesh to the vertex
  CHECK_VEC3_NEAR(c.position, p - (0.1 + 0.5 * c.dn) * c.normal, tol);
}

UNIT_TEST(vertex_face_contact_below) {
  RShapeDriver drv = make_driver();
  const Vec3d p = {-0.5, -0.25, -0.05};  // below f0
  CHECK(touched_faces(drv, p, 0.1) == std::vector<int>{0});
  contact c = detection_vertex_face(p, 0.1, drv.fields_.center_, 1.0, 0, &drv.shp_, drv.fields_.quat_);
  CHECK_VEC3_NEAR(c.normal, (Vec3d{0, 0, -1}), tol);
}

UNIT_TEST(vertex_face_no_contact) {
  RShapeDriver drv = make_driver();
  CHECK(touched_faces(drv, Vec3d{0.3, 0.2, 0.5}, 0.1).empty());   // too far
  CHECK(touched_faces(drv, Vec3d{1.5, 0.2, 0.05}, 0.1).empty());  // outside of the mesh
}

UNIT_TEST(vertex_face_contact_moved_mesh) {
  // mesh translated and rotated by 90 degrees around x: the mesh lies in the plane y = 0
  RShapeDriver drv = make_driver(STATIONARY, {}, Vec3d{5, 0, 1});
  const double s = std::sqrt(0.5);
  drv.fields_.quat_ = {s, s, 0, 0};
  const Vec3d p = {5.3, 0.05, 1.2};  // local (0.3, 0.2) -> f3
  CHECK(touched_faces(drv, p, 0.1) == std::vector<int>{3});
  contact c = detection_vertex_face(p, 0.1, drv.fields_.center_, 1.0, 3, &drv.shp_, drv.fields_.quat_);
  CHECK_VEC3_NEAR(c.normal, (Vec3d{0, 1, 0}), tol);
}

UNIT_TEST(vertex_edge_contact) {
  // vertex above the interior edge (4, 5)
  RShapeDriver drv = make_driver();
  contact c = detection_vertex_edge_core(Vec3d{0.5, 0, 0.05}, 0.1, drv.vertices_[4], drv.vertices_[5], minkowski);
  CHECK(c.is_contact);
  CHECK_NEAR(c.dn, 0.05 - 0.1 - minkowski, tol);
  CHECK_VEC3_NEAR(c.normal, (Vec3d{0, 0, 1}), tol);
  // beyond the end of the edge
  CHECK(
      !detection_vertex_edge_core(Vec3d{1.5, 0, 0.05}, 0.1, drv.vertices_[4], drv.vertices_[5], minkowski).is_contact);
}

// ============================================================
// === Translation
// ============================================================

UNIT_TEST(stationary) {
  RShapeDriver drv = make_driver();
  drv.fields_.vel_ = {1, 1, 1};
  Driver_params motion;
  drv.push_f_v(motion, 0.1);
  drv.push_f_v_r(motion, 0.0, 0.1);
  CHECK_VEC3_NEAR(drv.fields_.vel_, (Vec3d{0, 0, 0}), 0.0);
  CHECK_VEC3_NEAR(drv.fields_.center_, (Vec3d{0, 0, 0}), 0.0);
}

UNIT_TEST(linear_motion) {
  Driver_params motion;
  motion.motion_vector_ = {1, 0, 0};
  motion.const_vel_ = 2.0;
  RShapeDriver drv = make_driver(LINEAR_MOTION, motion);
  const double dt = 0.01;
  for (int i = 0; i < 100; i++) {
    drv.push_f_v(motion, dt);
    drv.push_f_v_r(motion, i * dt, dt);
  }
  CHECK_VEC3_NEAR(drv.fields_.center_, (Vec3d{2, 0, 0}), 1e-10);
}

UNIT_TEST(particle_motion) {
  // free body: a = F / m, velocity Verlet
  Driver_params motion;
  RShapeDriver drv;
  drv.shp_ = make_mesh();
  drv.motion_type_ = PARTICLE;
  drv.fields_.mass_ = 2.0;
  drv.fields_.inertia_ = {1, 1, 1};
  drv.initialize(motion);
  CHECK(drv.need_moment());

  drv.fields_.forces_ = {0, 0, -4};
  drv.force_to_accel(motion);
  CHECK_VEC3_NEAR(drv.fields_.acc_, (Vec3d{0, 0, -2}), tol);
  drv.push_f_v(motion, 0.1);
  CHECK_VEC3_NEAR(drv.fields_.vel_, (Vec3d{0, 0, -0.2}), tol);
  drv.push_f_v_r(motion, 0.0, 0.1);
  CHECK_VEC3_NEAR(drv.fields_.center_, (Vec3d{0, 0, -0.02 - 0.01}), tol);
}

UNIT_TEST(linear_force_motion) {
  Driver_params motion;
  motion.motion_vector_ = {0, 0, 1};
  motion.const_force_ = 3.0;
  RShapeDriver drv;
  drv.shp_ = make_mesh();
  drv.motion_type_ = LINEAR_FORCE_MOTION;
  drv.fields_.mass_ = 2.0;
  drv.initialize(motion);
  drv.fields_.forces_ = {5, 5, 1};
  drv.force_to_accel(motion);
  CHECK_VEC3_NEAR(drv.fields_.acc_, (Vec3d{0, 0, 2}), tol);  // (F.mv + const_force) mv / m
}

namespace {
// Piston compressing along mv (toward the particles) with sigma * S = 6 and 0.5 * m = 2:
// a = ((F + sigma * S * mv) . mv / (0.5 * m)) mv
Vec3d compressive_accel(const Vec3d& mv, const Vec3d& forces) {
  Driver_params motion;
  motion.motion_vector_ = mv;
  motion.sigma_ = 2.0;
  motion.damprate_ = 0.5;
  RShapeDriver drv;
  drv.shp_ = make_mesh();
  drv.motion_type_ = LINEAR_COMPRESSIVE_MOTION;
  drv.fields_.mass_ = 4.0;
  drv.fields_.surface_ = 3.0;
  drv.initialize(motion);
  drv.fields_.forces_ = forces;
  drv.force_to_accel(motion);
  return drv.fields_.acc_;
}
}  // namespace

UNIT_TEST(compressive_piston_from_above) {
  // the particles (below) push the piston up with 10 > sigma * S: the piston goes back up
  CHECK_VEC3_NEAR(compressive_accel({0, 0, -1}, {0, 0, 10}), (Vec3d{0, 0, 2}), tol);
  // weak contact forces: the stress pushes the piston down
  CHECK_VEC3_NEAR(compressive_accel({0, 0, -1}, {0, 0, 2}), (Vec3d{0, 0, -2}), tol);
}

UNIT_TEST(compressive_piston_from_below) {
  // same configuration upside down: the accelerations are opposite
  CHECK_VEC3_NEAR(compressive_accel({0, 0, 1}, {0, 0, -10}), (Vec3d{0, 0, -2}), tol);
  CHECK_VEC3_NEAR(compressive_accel({0, 0, 1}, {0, 0, -2}), (Vec3d{0, 0, 2}), tol);
}

UNIT_TEST(compressive_equilibrium) {
  // contact forces balance the applied stress, whatever the compression direction
  const double s = std::sqrt(0.5);
  for (const Vec3d& mv : {Vec3d{0, 0, -1}, Vec3d{1, 0, 0}, Vec3d{0, s, -s}}) {
    CHECK_VEC3_NEAR(compressive_accel(mv, -6.0 * mv), (Vec3d{0, 0, 0}), tol);
  }
}

UNIT_TEST(shaker_velocity_follows_the_direction) {
  Driver_params motion;
  motion.amplitude_ = 0.1;
  motion.omega_ = 5.0;
  motion.shaker_dir_ = {std::sqrt(0.75), 0, 0.5};
  RShapeDriver drv = make_driver(SHAKER, motion);
  drv.push_f_v_r(motion, 0.0, 0.01);
  const Vec3d expected = motion.shaker_velocity(0.01);  // already oriented along shaker_dir
  CHECK_VEC3_NEAR(drv.fields_.vel_, expected, tol);
  CHECK_NEAR(exanb::norm(exanb::cross(drv.fields_.vel_, motion.shaker_dir_)), 0.0, tol);
}

UNIT_TEST(tabulated) {
  Driver_params motion;
  motion.tab_time_ = {0.0, 1.0};
  motion.tab_pos_ = {Vec3d{0, 0, 0}, Vec3d{0, 0, 2}};
  RShapeDriver drv = make_driver(TABULATED, motion);
  drv.push_f_v_r(motion, 0.25, 0.01);
  CHECK_VEC3_NEAR(drv.fields_.center_, (Vec3d{0, 0, 0.5}), tol);
  CHECK_VEC3_NEAR(drv.fields_.vel_, (Vec3d{0, 0, 2}), tol);
}

UNIT_TEST(expression) {
  Driver_params motion;
  motion.expr_.expr_use_v_ = true;
  motion.expr_.expr_vx_ = "3";
  motion.expr_.expr_use_vrot_ = true;
  motion.expr_.expr_vrotz_ = "2*t";
  RShapeDriver drv = make_driver(EXPRESSION, motion);
  drv.push_f_v_r(motion, 0.5, 0.1);
  CHECK_VEC3_NEAR(drv.fields_.vel_, (Vec3d{3, 0, 0}), tol);
  CHECK_VEC3_NEAR(drv.fields_.vrot_, (Vec3d{0, 0, 1}), tol);
  CHECK_VEC3_NEAR(drv.fields_.center_, (Vec3d{0.3, 0, 0}), tol);
}

// ============================================================
// === Rotation
// ============================================================

UNIT_TEST(constant_angular_velocity) {
  // vrot = (0, 0, 1) during pi/2: quarter turn around z
  RShapeDriver drv = make_driver();
  drv.fields_.vrot_ = {0, 0, 1};
  Driver_params motion;
  const int n = 10000;
  const double dt = 0.5 * M_PI / n;
  for (int i = 0; i < n; i++) {
    drv.push_av_to_quat(motion, i * dt, dt);
  }
  const double s = std::sqrt(0.5);
  CHECK_NEAR(drv.fields_.quat_.w, s, 1e-6);
  CHECK_NEAR(drv.fields_.quat_.z, s, 1e-6);
  drv.update_vertex(2);
  CHECK_VEC3_NEAR(drv.vertices_[2], (Vec3d{0.5, 1, 0}), 1e-6);
}

UNIT_TEST(driven_by_moment) {
  // at rest, a moment M around z gives vrot = dt * M / I after one step
  RShapeDriver drv = make_driver();
  drv.fields_.drive_by_mom_ = true;
  drv.fields_.applied_mom_ = {0, 0, 2};
  drv.fields_.mom_axis_ = {0, 0, 1};
  drv.fields_.mom_ = {5, 0, 0};  // the component of the contact moment off the axis is ignored
  drv.fields_.inertia_ = {1, 1, 4};
  CHECK(drv.need_moment());
  Driver_params motion;
  drv.push_av_to_quat(motion, 0.0, 0.1);
  CHECK_VEC3_NEAR(drv.fields_.vrot_, (Vec3d{0, 0, 0.1 * 2 / 4}), tol);
  CHECK_VEC3_NEAR(drv.fields_.mom_, (Vec3d{0, 0, 0}), 0.0);  // the contact moment is consumed
}

// ============================================================
// === YAML / dump
// ============================================================

UNIT_TEST(yaml_fields) {
  const RShapeDriverFields f =
      YAML::Load("{ center: [1,2,3], vel: [0,0,1], quat: [1,0,0,0], surface: 2, moment: [0,0,4], inertia: [1,1,1] }")
          .as<RShapeDriverFields>();
  CHECK_VEC3_NEAR(f.center_, (Vec3d{1, 2, 3}), 0.0);
  CHECK_VEC3_NEAR(f.vel_, (Vec3d{0, 0, 1}), 0.0);
  CHECK_NEAR(f.surface_, 2.0, 0.0);
  CHECK(f.drive_by_mom_);
  CHECK_VEC3_NEAR(f.mom_axis_, (Vec3d{0, 0, 1}), tol);  // normalized moment
  CHECK_VEC3_NEAR(f.inertia_, (Vec3d{1, 1, 1}), 0.0);

  const RShapeDriverFields g = YAML::Load("{ center: [0,0,0] }").as<RShapeDriverFields>();
  CHECK(!g.drive_by_mom_);
}

UNIT_TEST(dump_and_read_back) {
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "exadem_unit_test_rshape";
  std::filesystem::create_directories(dir);
  const std::string path = dir.string() + "/";

  Driver_params motion;
  RShapeDriver drv = make_driver(STATIONARY, motion, Vec3d{1, 2, 3});
  std::stringstream stream;
  drv.dump_data(motion, 4, path, stream);
  CHECK(contains(stream.str(), "type: RSHAPE"));

  // driver state
  const YAML::Node node = YAML::Load(stream.str())[0];
  CHECK(node["filename"].as<std::string>() == path + "mesh4.shp");
  const RShapeDriverFields f = node["state"].as<RShapeDriverFields>();
  CHECK_VEC3_NEAR(f.center_, (Vec3d{1, 2, 3}), 1e-6);

  // shape file
  const shape shp = read_shp(path + "mesh4.shp");
  CHECK(shp.name_ == "mesh4");
  CHECK_NEAR(shp.minkowski(), minkowski, 1e-12);
  CHECK(shp.get_number_of_vertices() == 9);
  CHECK(shp.get_number_of_edges() == 12);
  CHECK(shp.get_number_of_faces() == 4);
  for (int i = 0; i < 9; i++) {
    CHECK_VEC3_NEAR(shp.get_vertex(i), drv.shp_.get_vertex(i), 1e-12);
  }
  CHECK_NEAR(shp.get_face_area(3), 0.5, tol);
  CHECK_NEAR(shp.volume_, 0.0, 0.0);  // a mesh has no volume
  std::filesystem::remove_all(dir);
}

// ============================================================
// === Grid cell indexes
// ============================================================

UNIT_TEST(grid_cell_indexes) {
  // cell 0: vertices {4}, edges {2, 3}, faces {0, 1, 2, 3}; cell 1: faces {3}
  RShapeDriverGridCellIndexes grid;
  grid.initialize(2, 1, 2, 5);
  grid.cells_[0].offset_ = 0;
  grid.cells_[1].offset_ = 7;
  const int v0[] = {4}, e0[] = {2, 3}, f0[] = {0, 1, 2, 3}, f1[] = {3};
  grid.fill_cell(0, v0, e0, f0);
  grid.fill_cell(1, {}, {}, f1);

  RShapeDriverCellAccessor c0(0, grid);
  CHECK(c0.rshape_nv_ == 1 && c0.rshape_ne_ == 2 && c0.rshape_nf_ == 4);
  CHECK(c0.grid_id_vertices_[0] == 4);
  CHECK(c0.grid_id_edges_[1] == 3);
  CHECK(c0.grid_id_faces_[3] == 3);

  RShapeDriverCellAccessor c1(1, grid);
  CHECK(c1.rshape_nv_ == 0 && c1.rshape_ne_ == 0 && c1.rshape_nf_ == 1);
  CHECK(c1.grid_id_faces_[0] == 3);

  grid.reset();
  CHECK(grid.cells_.size() == 0);
}

UNIT_TEST(drivers_container) {
  Drivers drvs;
  Driver_params motion;
  RShapeDriver drv = make_driver();
  drvs.add_driver(0, drv, motion);
  CHECK(drvs.type(0) == DRIVER_TYPE::RSHAPE);
  CHECK(drvs.get_typed_driver<RShapeDriver>(0).shp_.get_number_of_faces() == 4);
  DriversGPUAccessor acc(drvs);
  CHECK(int(acc.get_typed_driver<RShapeDriver>(0).vertices_.size()) == 9);
}

UNIT_TEST_MAIN()
