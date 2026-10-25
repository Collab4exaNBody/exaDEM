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
#include <iostream>  // OBBtree.hpp uses std::cerr without including <iostream>

#include <exaDEM/type/OBBtree.hpp>

#include <algorithm>
#include <set>
#include <utility>
#include <vector>

#include "unit_test.hpp"

using exanb::Mat3d;
using exanb::Quaternion;
using exanb::Vec3d;

namespace {
constexpr int n_leaves = 8;
constexpr double tol = 1e-10;
const Vec3d leaf_half_size = {0.1, 0.2, 0.3};  // distinct sizes to avoid degenerate eigenvalues

// Corners of an axis-aligned cuboid.
std::vector<Vec3d> cuboid_corners(const Vec3d& center, const Vec3d& half) {
  std::vector<Vec3d> corners;
  for (int c = 0; c < 8; c++) {
    corners.push_back(Vec3d{center.x + ((c & 1) ? half.x : -half.x), center.y + ((c & 2) ? half.y : -half.y),
                            center.z + ((c & 4) ? half.z : -half.z)});
  }
  return corners;
}

// n_leaves cuboids aligned along x: leaf i is centered at (i, 0, 0), its data is i.
std::vector<OBBbundle<int>> make_bundles() {
  std::vector<OBBbundle<int>> bundles;
  for (int i = 0; i < n_leaves; i++) {
    OBBbundle<int> b;
    b.data = i;
    b.points = cuboid_corners(Vec3d{double(i), 0, 0}, leaf_half_size);
    b.obb.center = Vec3d{double(i), 0, 0};
    b.obb.extent = leaf_half_size;
    bundles.push_back(b);
  }
  return bundles;
}

OBBtree<int> make_tree(double radius = 0.0) {
  auto bundles = make_bundles();
  OBBtree<int> tree;
  tree.root = OBBtree<int>::recursiveBuild(tree.root, bundles, radius);
  return tree;
}

// Point inclusion with a tolerance (OBB::intersect(Vec3d) is strict).
bool contains(const OBB& obb, const Vec3d& p, double eps = tol) {
  const Vec3d v = p - obb.center;
  return std::fabs(exanb::dot(v, obb.e1)) <= obb.extent.x + eps &&
         std::fabs(exanb::dot(v, obb.e2)) <= obb.extent.y + eps &&
         std::fabs(exanb::dot(v, obb.e3)) <= obb.extent.z + eps;
}

void collect_leaves(const OBBnode<int>* node, std::vector<int>& leaves) {
  if (node == nullptr) {
    return;
  }
  if (node->isLeaf()) {
    leaves.push_back(node->data);
    return;
  }
  collect_leaves(node->first, leaves);
  collect_leaves(node->second, leaves);
}

int depth(const OBBnode<int>* node) {
  if (node == nullptr) {
    return 0;
  }
  return 1 + std::max(depth(node->first), depth(node->second));
}

// Every point of the subtree leaves must be inside the node boundary.
void check_boundaries_contain_points(const OBBnode<int>* node, const std::vector<OBBbundle<int>>& bundles) {
  if (node == nullptr) {
    return;
  }
  std::vector<int> leaves;
  collect_leaves(node, leaves);
  for (int id : leaves) {
    for (const auto& p : bundles[id].points) {
      CHECK(contains(node->boundary, p));
    }
  }
  check_boundaries_contain_points(node->first, bundles);
  check_boundaries_contain_points(node->second, bundles);
}

void check_orthonormal(const OBB& obb) {
  CHECK_NEAR(exanb::norm(obb.e1), 1.0, tol);
  CHECK_NEAR(exanb::norm(obb.e2), 1.0, tol);
  CHECK_NEAR(exanb::norm(obb.e3), 1.0, tol);
  CHECK_NEAR(exanb::dot(obb.e1, obb.e2), 0.0, tol);
  CHECK_NEAR(exanb::dot(obb.e1, obb.e3), 0.0, tol);
  CHECK_NEAR(exanb::dot(obb.e2, obb.e3), 0.0, tol);
}
}  // namespace

// ============================================================
// === Covariance matrix and OBB fitting
// ============================================================

UNIT_TEST(covariance_matrix) {
  // points (+/-1, 0, 0) and (0, +/-2, 0): mean = 0, the matrix is the sum (not the mean) of p.p^T
  std::vector<OBBbundle<int>> bundles(1);
  bundles[0].points = {Vec3d{1, 0, 0}, Vec3d{-1, 0, 0}, Vec3d{0, 2, 0}, Vec3d{0, -2, 0}};
  const Mat3d C = OBBtree<int>::getCovarianceMatrix(bundles);
  CHECK_NEAR(C.m11, 2.0, tol);
  CHECK_NEAR(C.m22, 8.0, tol);
  CHECK_NEAR(C.m33, 0.0, tol);
  CHECK_NEAR(C.m12, 0.0, tol);
  CHECK_NEAR(C.m13, 0.0, tol);
  CHECK_NEAR(C.m23, 0.0, tol);
}

UNIT_TEST(covariance_matrix_is_symmetric) {
  std::vector<OBBbundle<int>> bundles(1);
  bundles[0].points = {Vec3d{1, 2, 3}, Vec3d{-1, 0.5, 2}, Vec3d{0, -2, 1}, Vec3d{3, 1, -1}};
  const Mat3d C = OBBtree<int>::getCovarianceMatrix(bundles);
  CHECK_NEAR(C.m12, C.m21, 0.0);
  CHECK_NEAR(C.m13, C.m31, 0.0);
  CHECK_NEAR(C.m23, C.m32, 0.0);
}

UNIT_TEST(fit_obb_axis_aligned_cuboid) {
  std::vector<OBBbundle<int>> bundles(1);
  bundles[0].points = cuboid_corners(Vec3d{1, 2, 3}, Vec3d{0.5, 1.0, 2.0});
  const OBB obb = OBBtree<int>::fitOBB(bundles);
  check_orthonormal(obb);
  CHECK_VEC3_NEAR(obb.center, (Vec3d{1, 2, 3}), tol);
  // axes are sorted by increasing variance
  CHECK_NEAR(std::fabs(obb.e1.x), 1.0, tol);
  CHECK_NEAR(std::fabs(obb.e2.y), 1.0, tol);
  CHECK_NEAR(std::fabs(obb.e3.z), 1.0, tol);
  CHECK_VEC3_NEAR(obb.extent, (Vec3d{0.5, 1.0, 2.0}), tol);
}

UNIT_TEST(fit_obb_with_radius) {
  std::vector<OBBbundle<int>> bundles(1);
  bundles[0].points = cuboid_corners(Vec3d{0, 0, 0}, Vec3d{0.5, 1.0, 2.0});
  const OBB obb = OBBtree<int>::fitOBB(bundles, 0.25);
  CHECK_VEC3_NEAR(obb.extent, (Vec3d{0.75, 1.25, 2.25}), tol);
}

UNIT_TEST(fit_obb_rotated_cuboid) {
  // the fitted box of a rotated cuboid recovers its orientation and its extents
  const double s = std::sin(M_PI / 12), c = std::cos(M_PI / 12);
  const Quaternion Q{c, s * 0.6, 0, s * 0.8};  // 30 degrees around (0.6, 0, 0.8)
  const Vec3d center = {-1, 0.5, 2};
  std::vector<OBBbundle<int>> bundles(1);
  for (const auto& p : cuboid_corners(Vec3d{0, 0, 0}, Vec3d{0.5, 1.0, 2.0})) {
    bundles[0].points.push_back(center + Q * p);
  }
  const OBB obb = OBBtree<int>::fitOBB(bundles);
  check_orthonormal(obb);
  CHECK_VEC3_NEAR(obb.center, center, 1e-9);
  CHECK_VEC3_NEAR(obb.extent, (Vec3d{0.5, 1.0, 2.0}), 1e-9);
  CHECK_NEAR(std::fabs(exanb::dot(obb.e3, Q * Vec3d{0, 0, 1})), 1.0, 1e-9);
  for (const auto& p : bundles[0].points) {
    CHECK(contains(obb, p, 1e-9));
  }
}

// ============================================================
// === Tree construction
// ============================================================

UNIT_TEST(recursive_build_structure) {
  const auto bundles = make_bundles();
  OBBtree<int> tree = make_tree();
  CHECK(tree.root != nullptr);
  CHECK(!tree.root->isLeaf());

  std::vector<int> leaves;
  collect_leaves(tree.root, leaves);
  std::sort(leaves.begin(), leaves.end());
  CHECK(int(leaves.size()) == n_leaves);
  for (int i = 0; i < int(leaves.size()); i++) {
    CHECK(leaves[i] == i);  // each leaf appears once
  }
  CHECK(depth(tree.root) == 4);  // 8 leaves split by halves: balanced tree
}

UNIT_TEST(recursive_build_boundaries_contain_points) {
  const auto bundles = make_bundles();
  OBBtree<int> tree = make_tree();
  check_boundaries_contain_points(tree.root, bundles);
  check_orthonormal(tree.root->boundary);
}

UNIT_TEST(recursive_build_single_leaf) {
  std::vector<OBBbundle<int>> bundles(1);
  bundles[0].data = 42;
  bundles[0].points = cuboid_corners(Vec3d{0, 0, 0}, leaf_half_size);
  OBBtree<int> tree;
  tree.root = OBBtree<int>::recursiveBuild(tree.root, bundles);
  CHECK(tree.root != nullptr);
  CHECK(tree.root->isLeaf());
  CHECK(tree.root->data == 42);
}

UNIT_TEST(recursive_build_empty) {
  std::vector<OBBbundle<int>> bundles;
  OBBtree<int> tree;
  tree.root = OBBtree<int>::recursiveBuild(tree.root, bundles);
  CHECK(tree.root == nullptr);
}

// ============================================================
// === Memory management (copy / move / reset)
// ============================================================

UNIT_TEST(copy_is_deep) {
  OBBtree<int> tree = make_tree();
  OBBtree<int> copy(tree);
  CHECK(copy.root != nullptr);
  CHECK(copy.root != tree.root);
  CHECK(copy.root->first != tree.root->first);
  std::vector<int> leaves_tree, leaves_copy;
  collect_leaves(tree.root, leaves_tree);
  collect_leaves(copy.root, leaves_copy);
  CHECK(leaves_tree == leaves_copy);

  // the copy survives the destruction of the original
  tree.reset(tree.root);
  CHECK(tree.root == nullptr);
  leaves_copy.clear();
  collect_leaves(copy.root, leaves_copy);
  CHECK(int(leaves_copy.size()) == n_leaves);
}

UNIT_TEST(copy_assignment) {
  OBBtree<int> tree = make_tree();
  OBBtree<int> other = make_tree(0.5);
  other = tree;
  CHECK(other.root != tree.root);
  CHECK_VEC3_NEAR(other.root->boundary.extent, tree.root->boundary.extent, 0.0);
  OBBtree<int>& self = other;
  other = self;  // self assignment must not free the tree
  CHECK(other.root != nullptr);
  CHECK_VEC3_NEAR(other.root->boundary.extent, tree.root->boundary.extent, 0.0);
}

UNIT_TEST(move) {
  OBBtree<int> tree = make_tree();
  OBBnode<int>* root = tree.root;
  OBBtree<int> moved(std::move(tree));
  CHECK(moved.root == root);
  CHECK(tree.root == nullptr);

  OBBtree<int> assigned;
  assigned = std::move(moved);
  CHECK(assigned.root == root);
  CHECK(moved.root == nullptr);
}

// ============================================================
// === Queries
// ============================================================

UNIT_TEST(obb_intersection_ids) {
  const auto bundles = make_bundles();
  OBBtree<int> tree = make_tree();

  // query around leaf 3 only
  OBB query;
  query.center = Vec3d{3, 0, 0};
  query.extent = Vec3d{0.2, 0.2, 0.2};
  std::vector<int> ids;
  OBBtree<int>::OBBIntersectionIds(query, tree.root, ids);
  CHECK(ids == std::vector<int>{3});

  // query covering leaves 2, 3 and 4
  query.extent = Vec3d{1.2, 0.2, 0.2};
  ids.clear();
  OBBtree<int>::OBBIntersectionIds(query, tree.root, ids);
  std::sort(ids.begin(), ids.end());
  CHECK((ids == std::vector<int>{2, 3, 4}));

  // query covering everything
  query.center = Vec3d{3.5, 0, 0};
  query.extent = Vec3d{10, 10, 10};
  ids.clear();
  OBBtree<int>::OBBIntersectionIds(query, tree.root, ids);
  CHECK(int(ids.size()) == n_leaves);

  // query far away
  query.center = Vec3d{3.5, 5, 0};
  query.extent = Vec3d{1, 1, 1};
  ids.clear();
  OBBtree<int>::OBBIntersectionIds(query, tree.root, ids);
  CHECK(ids.empty());

  // null tree
  ids.clear();
  OBBtree<int>::OBBIntersectionIds(query, nullptr, ids);
  CHECK(ids.empty());
}

UNIT_TEST(obb_intersection_ids_subset_of_brute_force) {
  // every reported leaf must really intersect the query
  const auto bundles = make_bundles();
  OBBtree<int> tree = make_tree();
  for (double x = -1.0; x <= 8.0; x += 0.25) {
    OBB query;
    query.center = Vec3d{x, 0.1, -0.1};
    query.extent = Vec3d{0.3, 0.3, 0.3};
    std::vector<int> ids;
    OBBtree<int>::OBBIntersectionIds(query, tree.root, ids);
    for (int id : ids) {
      CHECK(query.intersect(bundles[id].obb));
    }
  }
}

UNIT_TEST(tree_intersection_ids) {
  OBBtree<int> treeA = make_tree();
  OBBtree<int> treeB = make_tree();
  const Quaternion identity{1, 0, 0, 0};

  // same position: each leaf intersects at least itself and its neighbors, never a far leaf
  std::vector<std::pair<int, int>> pairs;
  OBBtree<int>::TreeIntersectionIds(treeA.root, treeB.root, pairs, 1.0, 1.0, 0.0, Vec3d{0, 0, 0}, identity);
  std::set<std::pair<int, int>> pair_set(pairs.begin(), pairs.end());
  for (int i = 0; i < n_leaves; i++) {
    CHECK(pair_set.count({i, i}) == 1);
  }
  for (const auto& [a, b] : pairs) {
    CHECK(std::abs(a - b) <= 1);
  }

  // B shifted by +1 along x: leaf i of A overlaps leaf i-1 of B
  pairs.clear();
  OBBtree<int>::TreeIntersectionIds(treeA.root, treeB.root, pairs, 1.0, 1.0, 0.0, Vec3d{1, 0, 0}, identity);
  pair_set = std::set<std::pair<int, int>>(pairs.begin(), pairs.end());
  for (int i = 1; i < n_leaves; i++) {
    CHECK(pair_set.count({i, i - 1}) == 1);
  }
  CHECK(pair_set.count({0, n_leaves - 1}) == 0);

  // B far away
  pairs.clear();
  OBBtree<int>::TreeIntersectionIds(treeA.root, treeB.root, pairs, 1.0, 1.0, 0.0, Vec3d{0, 10, 0}, identity);
  CHECK(pairs.empty());

  // enlarge value creates contacts
  OBBtree<int>::TreeIntersectionIds(treeA.root, treeB.root, pairs, 1.0, 1.0, 0.0, Vec3d{0, 0.5, 0}, identity);
  CHECK(pairs.empty());  // gap of 0.5 - 2 * 0.2 = 0.1 along y
  OBBtree<int>::TreeIntersectionIds(treeA.root, treeB.root, pairs, 1.0, 1.0, 0.1, Vec3d{0, 0.5, 0}, identity);
  CHECK(!pairs.empty());
}

UNIT_TEST(tree_intersection_ids_rotation) {
  // B rotated by 90 degrees around z: B becomes a column of leaves along y, centered at (0, i, 0).
  // Only leaf 0 of A (centered at the origin) can touch it, and only leaf 0 of B.
  OBBtree<int> treeA = make_tree();
  OBBtree<int> treeB = make_tree();
  const double s = std::sqrt(0.5);
  const Quaternion rotz90{s, 0, 0, s};
  std::vector<std::pair<int, int>> pairs;
  OBBtree<int>::TreeIntersectionIds(treeA.root, treeB.root, pairs, 1.0, 1.0, 0.0, Vec3d{0, 0, 0}, rotz90);
  CHECK(pairs.size() == 1);
  if (pairs.size() == 1) {
    CHECK(pairs[0].first == 0 && pairs[0].second == 0);
  }
}

UNIT_TEST(tree_intersection_ids_scale) {
  // B scaled by 2: leaf i of B is centered at (2i, 0, 0), so it matches leaf 2i of A
  OBBtree<int> treeA = make_tree();
  OBBtree<int> treeB = make_tree();
  const Quaternion identity{1, 0, 0, 0};
  std::vector<std::pair<int, int>> pairs;
  OBBtree<int>::TreeIntersectionIds(treeA.root, treeB.root, pairs, 1.0, 2.0, 0.0, Vec3d{0, 0, 0}, identity);
  std::set<std::pair<int, int>> pair_set(pairs.begin(), pairs.end());
  for (int i = 0; 2 * i < n_leaves; i++) {
    CHECK(pair_set.count({2 * i, i}) == 1);
  }
  CHECK(pair_set.count({1, 0}) == 0);  // leaf 1 of A is between leaves 0 and 1 of B (gap = 1 - 0.1 - 0.2 = 0.7)
}

UNIT_TEST(tree_intersection_ids_null) {
  OBBtree<int> tree = make_tree();
  std::vector<std::pair<int, int>> pairs;
  OBBtree<int>::TreeIntersectionIds(tree.root, nullptr, pairs, 1.0, 1.0, 0.0, Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
  OBBtree<int>::TreeIntersectionIds(nullptr, tree.root, pairs, 1.0, 1.0, 0.0, Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
  CHECK(pairs.empty());
}

UNIT_TEST_MAIN()
