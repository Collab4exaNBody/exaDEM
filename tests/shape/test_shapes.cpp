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
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape/basic_shape/cube.hpp>
#include <exaDEM/shape/basic_shape/rice.hpp>

#include <string>
#include <vector>

#include "unit_test.hpp"

using exaDEM::shape;
using exaDEM::shapes;
using exanb::ParticleTypeMap;
using exanb::Vec3d;

namespace {
shape make_cube(const std::string& name, double length = 1.0) {
  return exaDEM::basic_shape::create_cube(name, length, 0.01);
}

int count_leaves(const OBBnode<exaDEM::subBox>* node) {
  if (node == nullptr) {
    return 0;
  }
  if (node->isLeaf()) {
    return 1;
  }
  return count_leaves(node->first) + count_leaves(node->second);
}
}  // namespace

// ============================================================
// === Container
// ============================================================

UNIT_TEST(empty_container) {
  shapes shps;
  CHECK(shps.size() == 0);
  CHECK(shps.max_number_of_vertices() == 0);
  CHECK(!shps.use_obb_tree());
  CHECK(shps["cube"] == nullptr);
}

UNIT_TEST(add_shape_and_access) {
  shapes shps;
  shape cube = make_cube("cube");
  shape rice = exaDEM::basic_shape::create_rice("rice", 1.0, 0.1);
  shps.add_shape(rice);
  shps.add_shape(&cube);  // pointer overload
  CHECK(shps.size() == 2);
  CHECK(shps[0]->name_ == "rice");
  CHECK(shps[1]->name_ == "cube");
  CHECK(shps.data() == shps[0]);
  CHECK(shps.view().size() == 2);

  const shapes& cshps = shps;
  CHECK(cshps[1]->get_number_of_vertices() == 8);
  CHECK(cshps.size() == 2);
}

UNIT_TEST(max_number_of_vertices) {
  shapes shps;
  shape rice = exaDEM::basic_shape::create_rice("rice", 1.0, 0.1);
  shape cube = make_cube("cube");
  shps.add_shape(rice);
  CHECK(shps.max_number_of_vertices() == 2);
  shps.add_shape(cube);
  CHECK(shps.max_number_of_vertices() == 8);
  shape rice2 = exaDEM::basic_shape::create_rice("rice2", 1.0, 0.1);
  shps.add_shape(rice2);
  CHECK(shps.max_number_of_vertices() == 8);  // never decreases
}

UNIT_TEST(access_by_name) {
  shapes shps;
  shape a = make_cube("A");
  shape b = make_cube("B", 2.0);
  shps.add_shape(a);
  shps.add_shape(b);
  CHECK(shps["B"] == shps[1]);
  CHECK(shps["A"] == shps[0]);
  CHECK(shps["C"] == nullptr);
  CHECK_NEAR(shps["B"]->get_vertex(6).x, 1.0, 1e-14);
}

UNIT_TEST(contains) {
  shapes shps;
  shape a = make_cube("A");
  shape b = make_cube("B");
  shps.add_shape(a);
  CHECK(shps.contains(a));
  CHECK(!shps.contains(b));
}

UNIT_TEST(add_shape_copies) {
  // the container owns a copy: modifying the original does not affect it
  shapes shps;
  shape cube = make_cube("cube");
  shps.add_shape(cube);
  cube.name_ = "modified";
  cube.shift_vertices({10, 10, 10});
  CHECK(shps[0]->name_ == "cube");
  CHECK_NEAR(shps[0]->get_vertex(6).x, 0.5, 1e-14);
}

UNIT_TEST(enable_obb_tree) {
  shapes shps;
  shps.enable_obb_tree();
  CHECK(shps.use_obb_tree());
}

// ============================================================
// === Copies of shapes with an OBB tree
// ============================================================

UNIT_TEST(obb_tree_is_deep_copied) {
  shapes shps;
  {
    shape cube = make_cube("cube");
    cube.buildOBBtree();
    shps.add_shape(cube);
    CHECK(shps[0]->obbtree_.root != nullptr);
    CHECK(shps[0]->obbtree_.root != cube.obbtree_.root);
  }  // the original is destroyed here: the copy must stay valid
  CHECK(count_leaves(shps[0]->obbtree_.root) == 6 + 12 + 8);
}

UNIT_TEST(obb_tree_survives_reallocations) {
  // container growth copies/moves the shapes and their OBB trees (former double free)
  shapes shps;
  for (int i = 0; i < 20; i++) {
    shape cube = make_cube("cube" + std::to_string(i), 1.0 + i);
    cube.buildOBBtree();
    shps.add_shape(cube);
  }
  for (int i = 0; i < 20; i++) {
    CHECK(shps[i]->name_ == "cube" + std::to_string(i));
    CHECK(count_leaves(shps[i]->obbtree_.root) == 26);
  }
  shapes copy = shps;  // copy of the whole container
  CHECK(copy[0]->obbtree_.root != shps[0]->obbtree_.root);
  CHECK(count_leaves(copy[19]->obbtree_.root) == 26);
}

// ============================================================
// === Registration in the particle type map
// ============================================================

UNIT_TEST(register_shape) {
  ParticleTypeMap ptm;
  shapes shps;
  shape a = make_cube("A");
  shape b = make_cube("B");
  exaDEM::register_shape(ptm, shps, a);
  exaDEM::register_shape(ptm, shps, b);
  CHECK(shps.size() == 2);
  CHECK(ptm.size() == 2);
  CHECK(ptm["A"] == 0);
  CHECK(ptm["B"] == 1);
  CHECK(shps[ptm["B"]]->name_ == "B");
}

UNIT_TEST(register_shapes_renames_duplicates) {
  ParticleTypeMap ptm;
  shapes shps;
  shape a = make_cube("A");
  exaDEM::register_shape(ptm, shps, a);

  std::vector<shape> list = {make_cube("A"), make_cube("B")};
  exaDEM::register_shapes(ptm, shps, list);
  CHECK(shps.size() == 3);
  CHECK(list[0].name_ == "AX");  // renamed because "A" is already taken
  CHECK(ptm["AX"] == 1);
  CHECK(ptm["B"] == 2);
  CHECK(shps[1]->name_ == "AX");
}

UNIT_TEST_MAIN()
