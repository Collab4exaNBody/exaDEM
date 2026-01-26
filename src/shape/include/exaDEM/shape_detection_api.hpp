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

#include <math.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>

#include <exaDEM/shape.hpp>
#include <exaDEM/shape_prepro.hpp>
#include <exaDEM/type/contact.hpp>

namespace exaDEM {
/*********/
/* Utils */
/*********/
ONIKA_HOST_DEVICE_FUNC inline const Vec3d* get_ptr(const Vec3d* ptr) {
  return ptr;
}

/********************/
/* Vertex - Vertex  */
/********************/
ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex(const double rVerlet, const Vec3d& pi, const double radius,
                                                        const Vec3d& pj, const int j, const double hj,
                                                        const shape* shpj, const exanb::Quaternion& oj) {
  // === compute vertex position
  Vec3d vj = shpj->get_vertex(j, pj, hj, oj);
  return filter_vertex_vertex(rVerlet, pi, radius, vj, shpj->minskowski(hj));
}

ONIKA_HOST_DEVICE_FUNC
inline bool filter_vertex_vertex(const double rVerlet, const Vec3d& pi, const double hi, const int i, const shape* shpi,
                                 const exanb::Quaternion& oi, const Vec3d& pj, const double hj, const int j,
                                 const shape* shpj, const exanb::Quaternion& oj) {
  // === compute vertex position
  Vec3d vi = shpi->get_vertex(i, pi, hi, oi);
  Vec3d vj = shpj->get_vertex(j, pj, hj, oj);
  return filter_vertex_vertex(rVerlet, vi, shpi->minskowski(hi), vj, shpj->minskowski(hj));
}

template <typename VecI, typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex(const double rVerlet, VecI& vai, const double hi, const int i,
                                                        const shape* shpi, VecJ& vaj, const double hj, const int j,
                                                        const shape* shpj) {
  assert(i < shpi->get_number_of_vertices());
  assert(j < shpj->get_number_of_vertices());

  // === compute vertex position
  return filter_vertex_vertex(rVerlet, vai[i], shpi->minskowski(hi), vaj[j], shpj->minskowski(hj));
}

template <typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex_v2(const double rVerlet, const Vec3d& pi, const double ri,
                                                           const VecJ& vaj, const double rj, const int j) {
  // === compute vertex position
  return filter_vertex_vertex(rVerlet, pi, ri, vaj[j], rj);
}

template <typename VecI, typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex_v2(const double rVerlet, const VecI& vai, const double hi,
                                                           const int i, const shape* shpi, const VecJ& vaj,
                                                           const double hj, const int j, const shape* shpj) {
  assert(i < shpi->get_number_of_vertices());
  assert(j < shpj->get_number_of_vertices());

  // === compute vertex position
  return filter_vertex_vertex(rVerlet, vai[i], shpi->minskowski(hi), vaj[j], shpj->minskowski(hj));
}

ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_vertex(const Vec3d& pi, const double hi, const int i,
                                                              const shape* shpi, const exanb::Quaternion& oi,
                                                              const Vec3d& pj, const double hj, const int j,
                                                              const shape* shpj, const exanb::Quaternion& oj) {
  // === compute vertex position
  Vec3d vi = shpi->get_vertex(i, pi, hi, oi);
  Vec3d vj = shpj->get_vertex(j, pj, hj, oj);
  return detection_vertex_vertex_core(vi, shpi->minskowski(hi), vj, shpj->minskowski(hj));
}

ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_vertex(const Vec3d& pi, const double radius, const Vec3d& pj,
                                                              const double hj, const int j, const shape* shpj,
                                                              const exanb::Quaternion& oj) {
  // === compute vertex position
  Vec3d vj = shpj->get_vertex(j, pj, hj, oj);
  return detection_vertex_vertex_core(pi, radius, vj, shpj->minskowski(hj));
}

template <typename VecI, typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_vertex(const VecI& vai, const double hi, const int i,
                                                              const shape* shpi, const VecJ& vaj, const double hj,
                                                              const int j, const shape* shpj) {
  assert(i < shpi->get_number_of_vertices());
  assert(j < shpj->get_number_of_vertices());

  // === get vertex position
  const Vec3d& vi = vai[i];
  const Vec3d& vj = vaj[j];
  return detection_vertex_vertex_core(vi, shpi->minskowski(hi), vj, shpj->minskowski(hj));
}

/*****************/
/* Vertex - Edge */
/*****************/

// API filter_vertex_edge
ONIKA_HOST_DEVICE_FUNC
inline bool filter_vertex_edge(const double rVerlet, const Vec3d& pi, const double hi, const int i, const shape* shpi,
                               const exanb::Quaternion& oi, const Vec3d& pj, const double hj, const int j,
                               const shape* shpj, const exanb::Quaternion& oj) {
  // === compute vertice positions
  auto [first, second] = shpj->get_edge(j);
  const Vec3d vi = shpi->get_vertex(i, pi, hi, oi);
  const Vec3d vf = shpj->get_vertex(first, pj, hj, oj);
  const Vec3d vs = shpj->get_vertex(second, pj, hj, oj);
  double ri = shpi->minskowski(hi);
  double rj = shpj->minskowski(hj);
  return filter_vertex_edge_core(rVerlet, vi, ri, vf, vs, rj);
}

ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_edge(const double rVerlet, const Vec3d& pi, const double radius,
                                                      const Vec3d& pj, const double hj, const int j, const shape* shpj,
                                                      const exanb::Quaternion& oj) {
  // === compute vertice positions
  auto [first, second] = shpj->get_edge(j);
  const Vec3d vf = shpj->get_vertex(first, pj, hj, oj);
  const Vec3d vs = shpj->get_vertex(second, pj, hj, oj);
  double rj = shpj->minskowski(hj);
  return filter_vertex_edge_core(rVerlet, pi, radius, vf, vs, rj);
}

template <typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_edge(const double rVerlet, const Vec3d& vi, const double ri, VecJ& vaj,
                                                      const double hj, const int j, const shape* shpj) {
  assert(j < shpj->get_number_of_edges());
  auto [first, second] = shpj->get_edge(j);
  double rj = shpj->minskowski(hj);
  return filter_vertex_edge_core(rVerlet, vi, ri, vaj[first], vaj[second], rj);
}

template <typename VecI, typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_edge(const double rVerlet, const VecI& vai, const double hi,
                                                      const int i, const shape* shpi, const VecJ& vaj, const double hj,
                                                      const int j, const shape* shpj) {
  assert(i < shpi->get_number_of_vertices());
  assert(j < shpj->get_number_of_edges());

  auto [first, second] = shpj->get_edge(j);
  double ri = shpi->minskowski(hi);
  double rj = shpj->minskowski(hj);
  return filter_vertex_edge_core(rVerlet, vai[i], ri, vaj[first], vaj[second], rj);
}

// API detection_vertex_edge
ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_edge(const Vec3d& pi, const double hi, const int i,
                                                            const shape* shpi, const exanb::Quaternion& oi,
                                                            const Vec3d& pj, const double hj, const int j,
                                                            const shape* shpj, const exanb::Quaternion& oj) {
  // === compute vertice positions
  auto [first, second] = shpj->get_edge(j);
  const Vec3d vi = shpi->get_vertex(i, pi, hi, oi);
  const Vec3d vf = shpj->get_vertex(first, pj, hj, oj);
  const Vec3d vs = shpj->get_vertex(second, pj, hj, oj);
  double ri = shpi->minskowski(hi);
  double rj = shpj->minskowski(hj);
  return detection_vertex_edge_core(vi, ri, vf, vs, rj);
}

ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_edge(const Vec3d& pi, const double radius, const Vec3d& pj,
                                                            const double hj, const int j, const shape* shpj,
                                                            const exanb::Quaternion& oj) {
  assert(j < shpj->get_number_of_edges());
  // === compute vertice positions
  auto [first, second] = shpj->get_edge(j);
  const Vec3d vf = shpj->get_vertex(first, pj, hj, oj);
  const Vec3d vs = shpj->get_vertex(second, pj, hj, oj);
  double rj = shpj->minskowski(hj);
  return detection_vertex_edge_core(pi, radius, vf, vs, rj);
}

// API detection_vertex_edge
template <typename VecI, typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_edge(VecI& vai, const double hi, const int i, const shape* shpi,
                                                            VecJ& vaj, const double hj, const int j,
                                                            const shape* shpj) {
  assert(i < shpi->get_number_of_vertices());
  assert(j < shpj->get_number_of_edges());
  auto [first, second] = shpj->get_edge(j);
  double ri = shpi->minskowski(hi);
  double rj = shpj->minskowski(hj);
  return detection_vertex_edge_core(vai[i], ri, vaj[first], vaj[second], rj);
}

/*****************/
/* Vertex - Face */
/*****************/

template <typename VecI, typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_face(const double rVerlet, const VecI& vai, const double hi,
                                                      const int i, const shape* shpi, const VecJ& vaj, const double hj,
                                                      const int j, const shape* shpj) {
  assert(i < shpi->get_number_of_vertices());
  assert(j < shpj->get_number_of_faces());
  return filter_vertex_face(rVerlet, vai[i], shpi->minskowski(hi), vaj, hj, j, shpj);
}

template <typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_face(const Vec3d& vi, const double ri, VecJ& vaj,
                                                            const double hj, const int j, const shape* shpj) {
  assert(j < shpj->get_number_of_faces());
  return detection_vertex_face_core(vi, ri, vaj, hj, j, shpj);
}

template <typename VecI, typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_face(VecI& vai, const double hi, const int i, const shape* shpi,
                                                            VecJ& vaj, const double hj, const int j,
                                                            const shape* shpj) {
  assert(i < shpi->get_number_of_vertices());
  assert(j < shpj->get_number_of_faces());
  return detection_vertex_face_core(vai[i], shpi->minskowski(hi), vaj, hj, j, shpj);
}

/***************/
/* Edge - Edge */
/***************/

// API edge - edge
template <typename VecI, typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline bool filter_edge_edge(const double rVerlet, const VecI& vai, const double hi, const int i,
                                                    const shape* shpi, const VecJ& vaj, const double hj, const int j,
                                                    const shape* shpj) {
  assert(i < shpi->get_number_of_edges());
  assert(j < shpj->get_number_of_edges());
  double ri = shpi->minskowski(hi);
  double rj = shpj->minskowski(hj);
  // === compute vertices from shapes
  auto [fi, si] = shpi->get_edge(i);
  auto [fj, sj] = shpj->get_edge(j);
  return filter_edge_edge_core(rVerlet, vai[fi], vai[si], ri, vaj[fj], vaj[sj], rj);
}

ONIKA_HOST_DEVICE_FUNC
inline bool filter_edge_edge(const double rVerlet, const Vec3d& pi, const double hi, const int i, const shape* shpi,
                             const exanb::Quaternion& oi, const Vec3d& pj, const double hj, const int j,
                             const shape* shpj, const exanb::Quaternion& oj) {
  double ri = shpi->minskowski(hi);
  double rj = shpj->minskowski(hj);
  // === compute vertices from shapes
  auto [fi, si] = shpi->get_edge(i);
  const Vec3d vfi = shpi->get_vertex(fi, pi, hi, oi);
  const Vec3d vsi = shpi->get_vertex(si, pi, hi, oi);

  auto [fj, sj] = shpj->get_edge(j);
  const Vec3d vfj = shpj->get_vertex(fj, pj, hj, oj);
  const Vec3d vsj = shpj->get_vertex(sj, pj, hj, oj);
  return filter_edge_edge_core(rVerlet, vfi, vsi, ri, vfj, vsj, rj);
}

// API edge - edge
ONIKA_HOST_DEVICE_FUNC
inline contact detection_edge_edge(const Vec3d& pi, const double hi, const int i, const shape* shpi,
                                   const exanb::Quaternion& oi, const Vec3d& pj, const double hj, const int j,
                                   const shape* shpj, const exanb::Quaternion& oj) {
  double ri = shpi->minskowski(hi);
  double rj = shpj->minskowski(hj);
  // === compute vertices from shapes
  auto [fi, si] = shpi->get_edge(i);
  const Vec3d vfi = shpi->get_vertex(fi, pi, hi, oi);
  const Vec3d vsi = shpi->get_vertex(si, pi, hj, oi);

  auto [fj, sj] = shpj->get_edge(j);
  const Vec3d vfj = shpj->get_vertex(fj, pj, hj, oj);
  const Vec3d vsj = shpj->get_vertex(sj, pj, hj, oj);
  return detection_edge_edge_core(vfi, vsi, ri, vfj, vsj, rj);
}

// API edge - edge
template <typename VecI, typename VecJ>
ONIKA_HOST_DEVICE_FUNC inline contact detection_edge_edge(VecI& vai, const double hi, const int i, const shape* shpi,
                                                          VecJ& vaj, const double hj, const int j, const shape* shpj) {
  assert(i < shpi->get_number_of_edges());
  assert(j < shpj->get_number_of_edges());
  double ri = shpi->minskowski(hi);
  double rj = shpj->minskowski(hj);
  // === compute vertices from shapes
  auto [fi, si] = shpi->get_edge(i);
  auto [fj, sj] = shpj->get_edge(j);
  return detection_edge_edge_core(vai[fi], vai[si], ri, vaj[fj], vaj[sj], rj);
}

// NEW

#define __params__                                                                                                \
  const VertexType &vai, const double hi, const int i, const shape *shpi, const VertexType &vaj, const double hj, \
      const int j, const shape *shpj
#define __params__use__ vai, hi, i, shpi, vaj, hj, j, shpj
template <int INTERACTION_TYPE>
struct detect {};
template <>
struct detect<0> {
  template <typename VertexType>
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const {
    return detection_vertex_vertex(__params__use__);
  }
};
template <>
struct detect<1> {
  template <typename VertexType>
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const {
    return detection_vertex_edge(__params__use__);
  }
};
template <>
struct detect<2> {
  template <typename VertexType>
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const {
    return detection_vertex_face(__params__use__);
  }
};
template <>
struct detect<3> {
  template <typename VertexType>
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const {
    return detection_edge_edge(__params__use__);
  }
};
#undef __params__
#undef __params__use__

/** This part concerns detection between particles and “stl mesh”.  **/
#define __params__                                                                                         \
  const VertexType &pi, const double hi, const int i, const shape *shpi, const Vec3d *pj, const double hj, \
      const int j, const shape *shpj
#define __params__sph__                                                                               \
  const Vec3d &pi, const double ri, const Vec3d &pj, const double hj, const int j, const shape *shpj, \
      const exanb::Quaternion &oj
#define __params__use__ pi, hi, i, shpi, pj, hj, j, shpj
#define __params__use__inv__ pj, hj, j, shpj, pi, hi, i, shpi
#define __params__use__sph__ pi, ri, pj, hj, j, shpj, oj
template <>
struct detect<7> {
  template <typename VertexType>
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const {
    return detection_vertex_vertex(__params__use__);
  }  // polyhedron
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__sph__) const {
    return detection_vertex_vertex(__params__use__sph__);
  }  // sphere
};
template <>
struct detect<8> {
  template <typename VertexType>
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const {
    return detection_vertex_edge(__params__use__);
  }  // polyhedron
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__sph__) const {
    return detection_vertex_edge(__params__use__sph__);
  }  // sphere
};
template <>
struct detect<9> {
  template <typename VertexType>
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const {
    return detection_vertex_face(__params__use__);
  }  // polyhedron
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__sph__) const {
    return detection_vertex_face(__params__use__sph__);
  }  // sphere
};
template <>
struct detect<10> {
  template <typename VertexType>
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const {
    return detection_edge_edge(__params__use__);
  }
};
template <>
struct detect<11> {
  template <typename VertexType>
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const {
    return detection_vertex_edge(__params__use__inv__);
  }
};
template <>
struct detect<12> {
  template <typename VertexType>
  ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const {
    return detection_vertex_face(__params__use__inv__);
  }
};
#undef __params__
#undef __params__sph__
#undef __params__use__
#undef __params__use__inv__
#undef __params__use__sph__

// OLD
/*
# define __params__ const VertexArray &vai, const int i, const shape *shpi,
const VertexArray &vaj, const int j, const shape *shpj # define __params__use__
vai, i, shpi, vaj, j, shpj template<int INTERACTION_TYPE> struct detect{};
template<> struct detect<0>{ ONIKA_HOST_DEVICE_FUNC inline contact
operator()(__params__) const { return detection_vertex_vertex(__params__use__);
} }; template<> struct detect<1>{ ONIKA_HOST_DEVICE_FUNC inline contact
operator()(__params__) const { return detection_vertex_edge(__params__use__); }
}; template<> struct detect<2>{ ONIKA_HOST_DEVICE_FUNC inline contact
operator()(__params__) const { return detection_vertex_face(__params__use__); }
}; template<> struct detect<3>{ ONIKA_HOST_DEVICE_FUNC inline contact
operator()(__params__) const { return detection_edge_edge(__params__use__); } };
# undef __params__
# undef __params__use__
*/
/** This part concerns detection between particles and “stl mesh”.  **/
/*
# define __params__ const VertexArray &pi, const int i, const shape *shpi, const
Vec3d *pj, const int j, const shape *shpj # define __params__sph__ const Vec3d
&pi, const double ri, const Vec3d &pj, const int j, const shape *shpj, const
exanb::Quaternion &oj # define __params__use__      pi, i, shpi, pj, j, shpj #
define __params__use__inv__ pj, j, shpj, pi, i, shpi # define
__params__use__sph__     pi, ri, pj, j, shpj, oj template<> struct detect<7>{
ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return
detection_vertex_vertex(__params__use__); } // polyhedron ONIKA_HOST_DEVICE_FUNC
inline contact operator()(__params__sph__) const { return
detection_vertex_vertex(__params__use__sph__); } // sphere
};
template<> struct detect<8>{
ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return
detection_vertex_edge(__params__use__); } // polyhedron ONIKA_HOST_DEVICE_FUNC
inline contact operator()(__params__sph__) const { return
detection_vertex_edge(__params__use__sph__); } // sphere
};
template<> struct detect<9>{
ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return
detection_vertex_face(__params__use__); } // polyhedron ONIKA_HOST_DEVICE_FUNC
inline contact operator()(__params__sph__) const { return
detection_vertex_face(__params__use__sph__); }  // sphere
};
template<> struct detect<10>{ ONIKA_HOST_DEVICE_FUNC inline contact
operator()(__params__) const { return detection_edge_edge(__params__use__); } };
template<> struct detect<11>{ ONIKA_HOST_DEVICE_FUNC inline contact
operator()(__params__) const { return
detection_vertex_edge(__params__use__inv__); } }; template<> struct detect<12>{
ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return
detection_vertex_face(__params__use__inv__); } }; # undef __params__ # undef
__params__sph__ # undef __params__use__ # undef __params__use__inv__ # undef
__params__use__sph__
*/

}  // namespace exaDEM
