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
#pragma once

#include <exaDEM/shape.hpp>
#include <exaDEM/shape_detection.hpp>

namespace exaDEM {
//  template<typename Vec>
inline OBB build_OBB(const std::span<exanb::Vec3d> vec, double radius) {
  OBB obb;
  exanb::Vec3d mu{0.0, 0.0, 0.0};
  for (size_t i = 0; i < vec.size(); i++) {
    mu += vec[i];
  }
  mu /= (double)vec.size();

  // loop over the points again to build the
  // covariance matrix.  Note that we only have
  // to build terms for the upper trianglular
  // portion since the matrix is symmetric
  double cxx = 0.0, cxy = 0.0, cxz = 0.0, cyy = 0.0, cyz = 0.0, czz = 0.0;
  for (size_t i = 0; i < vec.size(); i++) {
    exanb::Vec3d p = vec[i];
    cxx += p.x * p.x - mu.x * mu.x;
    cxy += p.x * p.y - mu.x * mu.y;
    cxz += p.x * p.z - mu.x * mu.z;
    cyy += p.y * p.y - mu.y * mu.y;
    cyz += p.y * p.z - mu.y * mu.z;
    czz += p.z * p.z - mu.z * mu.z;
  }

  // now build the covariance matrix
  exanb::Mat3d C;
  C.m11 = cxx;
  C.m12 = cxy;
  C.m13 = cxz;
  C.m21 = cxy;
  C.m22 = cyy;
  C.m23 = cyz;
  C.m31 = cxz;
  C.m32 = cyz;
  C.m33 = czz;

  // ==== set the OBB parameters from the covariance matrix
  // extract the eigenvalues and eigenvectors from C
  exanb::Vec3d eigvec[3];
  double eigval[3];
  exanb::symmetric_matrix_eigensystem(C, eigvec, eigval);

  // find the right, up and forward vectors from the eigenvectors
  // (order does not matter here, only orthonormality does)
  exanb::Vec3d r = eigvec[0];
  exanb::Vec3d u = eigvec[1];
  exanb::Vec3d f = eigvec[2];
  r = r / exanb::norm(r);
  u = u / exanb::norm(u);
  f = f / exanb::norm(f);

  // now build the bounding box extents in the rotated frame
  exanb::Vec3d minim{1e20, 1e20, 1e20}, maxim{-1e20, -1e20, -1e20};
  for (size_t i = 0; i < vec.size(); i++) {
    exanb::Vec3d p_prime{exanb::dot(r, vec[i]), exanb::dot(u, vec[i]), exanb::dot(f, vec[i])};
    if (minim.x > p_prime.x) minim.x = p_prime.x;
    if (minim.y > p_prime.y) minim.y = p_prime.y;
    if (minim.z > p_prime.z) minim.z = p_prime.z;
    if (maxim.x < p_prime.x) maxim.x = p_prime.x;
    if (maxim.y < p_prime.y) maxim.y = p_prime.y;
    if (maxim.z < p_prime.z) maxim.z = p_prime.z;
  }

  // set the center of the OBB to be the average of the
  // minimum and maximum, and the extents be half of the
  // difference between the minimum and maximum
  const exanb::Vec3d half_sum = 0.5 * (maxim + minim);
  obb.center = r * half_sum.x + u * half_sum.y + f * half_sum.z;
  obb.e1 = r;
  obb.e2 = u;
  obb.e3 = f;
  obb.extent = 0.5 * (maxim - minim);

  obb.enlarge(radius);  // Add the Minkowski radius
  return obb;
}

inline OBB build_obb_from_shape(const shape& shp) {
  size_t nv = shp.get_number_of_vertices();

  const double ext = shp.minkowski(1.0);
  std::vector<exanb::Vec3d> vbuf;
  vbuf.resize(nv);
  for (size_t i = 0; i < nv; i++) {
    vbuf[i] = shp.get_vertex(i);
  }
  OBB res = build_OBB(vbuf, ext);
  return res;
}
// #define OLD_VERSION;

// general functon;
inline OBB build_obb_vertex(const int index, const shape* shp, const exanb::Vec3d* v) {
  const double ext = shp->minkowski();
  std::array<exanb::Vec3d, 1> vbuf = {v[index]};
  OBB res = build_OBB(vbuf, ext);
  return res;
}

inline OBB build_obb_edge(const exanb::Vec3d& position, const int index, const shape* shp,
                          const exanb::Quaternion& orientation) {
  const double ext = shp->minkowski();
  auto [first, second] = shp->get_edge(index);
  const exanb::Vec3d vf = shp->get_vertex(first, position, 1.0, orientation);
  const exanb::Vec3d vs = shp->get_vertex(second, position, 1.0, orientation);
  std::array<exanb::Vec3d, 2> v = {vf, vs};
  OBB res = build_OBB(v, ext);
  return res;
}

inline OBB build_obb_edge(const int index, const shape* shp, const exanb::Vec3d* v) {
  const double ext = shp->minkowski();
  auto [first, second] = shp->get_edge(index);
  std::array<exanb::Vec3d, 2> vbuf = {v[first], v[second]};
  OBB res = build_OBB(vbuf, ext);
  return res;
}

inline OBB build_obb_face(const exanb::Vec3d& position, const int index, const shape* shp,
                          const exanb::Quaternion& orientation) {
  const double ext = shp->minkowski();
  const auto [data, nf] = shp->get_face(index);
  std::vector<exanb::Vec3d> v(nf);
  for (int i = 0; i < nf; i++) {
    v[i] = shp->get_vertex(data[i], position, 1.0, orientation);
  }
  OBB res = build_OBB(v, ext);
  return res;
}

inline OBB build_obb_face(const int index, const shape* shp, const exanb::Vec3d* const v,
                          std::vector<exanb::Vec3d>& vbuf) {
  const double ext = shp->minkowski();
  const auto [data, nf] = shp->get_face(index);
  vbuf.resize(nf);
  for (int i = 0; i < nf; i++) {
    vbuf[i] = v[data[i]];
  }
  OBB res = build_OBB(vbuf, ext);
  return res;
}

inline void shape::pre_compute_obb_vertices(const exanb::Vec3d* const v) {
  // This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_face`
  const size_t size = this->get_number_of_vertices();
  obb_vertices_.resize(size);
  exanb::ldbg << "obb [vertices] = " << size << std::endl;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size; i++) {
    obb_vertices_[i] = build_obb_vertex(i, this, v);
  }
}

// Do not include in a OMP parallel region
inline void shape::pre_compute_obb_edges(const exanb::Vec3d& particle_center, const exanb::Quaternion& particle_quat) {
  // This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_edge`
  const size_t size = this->get_number_of_edges();
  obb_edges_.resize(size);
  const exanb::Vec3d center = particle_center;  // conv_to_Vec3d(this->obb_.center) + particle_center;
  exanb::ldbg << "obb [edges]    = " << size << std::endl;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size; i++) {
    obb_edges_[i] = build_obb_edge(center, i, this, particle_quat);
  }
}

// Do not include in a OMP parallel region
inline void shape::pre_compute_obb_edges(const exanb::Vec3d* v) {
  // This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_edge`
  const size_t size = this->get_number_of_edges();
  obb_edges_.resize(size);
  exanb::ldbg << "obb [edges]    = " << size << std::endl;
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size; i++) {
    obb_edges_[i] = build_obb_edge(i, this, v);
  }
}

// Do not include in a OMP parallel region
inline void shape::pre_compute_obb_faces(const exanb::Vec3d& particle_center, const exanb::Quaternion& particle_quat) {
  // This function could be optimized by avoiding to use `position` and `orientation` in `build_obb_face`
  const size_t size = this->get_number_of_faces();
  obb_faces_.resize(size);
  const exanb::Vec3d center = particle_center;  // conv_to_Vec3d(this->obb_.center) + particle_center;
  exanb::ldbg << "obb [faces]    = " << size << std::endl;

#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < size; i++) {
    obb_faces_[i] = build_obb_face(center, i, this, particle_quat);
  }
}

// Do not include in a OMP parallel region
inline void shape::pre_compute_obb_faces(const exanb::Vec3d* v) {
  const size_t size = this->get_number_of_faces();
  obb_faces_.resize(size);
  exanb::ldbg << "obb [faces]    = " << size << std::endl;

#pragma omp parallel
  {
    std::vector<exanb::Vec3d> vbuf;  // buffer that will contain tmp vertex positions
#pragma omp for schedule(static)
    for (size_t i = 0; i < size; i++) {
      obb_faces_[i] = build_obb_face(i, this, v, vbuf);
    }
  }
}

// Do not include in a OMP parallel region
inline void shape::increase_obb(const double value) {
#pragma omp parallel
  {
#pragma omp for schedule(static) nowait
    for (size_t i = 0; i < obb_vertices_.size(); i++) {
      obb_vertices_[i].enlarge(value);
    }
#pragma omp for schedule(static) nowait
    for (size_t i = 0; i < obb_edges_.size(); i++) {
      obb_edges_[i].enlarge(value);
    }
#pragma omp for schedule(static)
    for (size_t i = 0; i < obb_faces_.size(); i++) {
      obb_faces_[i].enlarge(value);
    }
  }
}

// Do not include in a OMP parallel region
inline void shape::compute_prepro_obb(exanb::Vec3d* scratch, const exanb::Vec3d& particle_center,
                                      const exanb::Quaternion& particle_quat) {
  const size_t nv = this->get_number_of_vertices();
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < nv; i++) {
    scratch[i] = this->get_vertex(i, particle_center, 1.0, particle_quat);
  }

  this->pre_compute_obb_vertices(scratch);
  this->pre_compute_obb_edges(scratch);
  this->pre_compute_obb_faces(scratch);
  this->increase_obb(this->minkowski());
}
}  // namespace exaDEM
