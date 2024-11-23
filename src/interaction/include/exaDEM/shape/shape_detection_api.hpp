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

#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exaDEM/shape/shape.hpp>
#include <math.h>
#include <exaDEM/shape/shape_prepro.hpp>
#include <exaDEM/type/contact.hpp>

namespace exaDEM
{
  using namespace exanb;
  using VertexArray = ::onika::oarray_t<::exanb::Vec3d, EXADEM_MAX_VERTICES>;

  ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_vertex(
      const Vec3d &pi, 
      const int i, 
      const shape *shpi, 
      const exanb::Quaternion &oi, 
      const Vec3d &pj, const int j, 
      const shape *shpj, 
      const exanb::Quaternion &oj)
  {

    // === compute vertex position
    Vec3d vi = shpi->get_vertex(i, pi, oi);
    Vec3d vj = shpj->get_vertex(j, pj, oj);
    return detection_vertex_vertex_core(vi, shpi->m_radius, vj, shpj->m_radius);
  }

  ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_vertex(
      const Vec3d &pi, 
      const double radius, 
      const Vec3d &pj, 
      const int j, 
      const shape *shpj, 
      const exanb::Quaternion &oj)
  {

    // === compute vertex position
    Vec3d vj = shpj->get_vertex(j, pj, oj);
    return detection_vertex_vertex_core(pi, radius, vj, shpj->m_radius);
  }

  ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_vertex_precompute(
      const VertexArray &vai, 
      const int i, 
      const shape *shpi, 
      const VertexArray &vaj, 
      const int j, 
      const shape *shpj)
  {
    // === get vertex position
    const Vec3d &vi = vai[i];
    const Vec3d &vj = vaj[j];
    return detection_vertex_vertex_core(vi, shpi->m_radius, vj, shpj->m_radius);
  }

  /**
   * @brief Filters vertex-vertex interactions based on a specified Verlet radius.
   * @param rVerlet The Verlet radius used for filtering interactions.
   * @param pi The position vector of the first vertex.
   * @param i The index of the first vertex.
   * @param shpi The shape associated with the first vertex.
   * @param oi The orientation of the first vertex.
   * @param pj The position vector of the second vertex.
   * @param j The index of the second vertex.
   * @param shpj The shape associated with the second vertex.
   * @param oj The orientation of the second vertex.
   * @return True if the distance between the vertices is less than or equal to the Verlet radius + shape radii, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex(
      const double rVerlet, 
      const Vec3d &pi, 
      const int i, 
      const shape *shpi, 
      const exanb::Quaternion &oi, 
      const Vec3d &pj, 
      const int j, 
      const shape *shpj, 
      const exanb::Quaternion &oj)
  {
    // === compute vertex position
    Vec3d vi = shpi->get_vertex(i, pi, oi);
    Vec3d vj = shpj->get_vertex(j, pj, oj);
    return filter_vertex_vertex(rVerlet, vi, shpi->m_radius, vj, shpj->m_radius);
  }

  ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex(
      const double rVerlet, 
      const Vec3d &pi, 
      const double radius, 
      const Vec3d &pj, 
      const int j, 
      const shape *shpj, 
      const exanb::Quaternion &oj)
  {
    // === compute vertex position
    Vec3d vj = shpj->get_vertex(j, pj, oj);
    return filter_vertex_vertex(rVerlet, pi, radius, vj, shpj->m_radius);
  }

  ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex(
      const double rVerlet, 
      const VertexArray &vai, 
      const int i, 
      const shape *shpi, 
      const VertexArray &vaj, 
      const int j, 
      const shape *shpj)
  {
    // === compute vertex position
    return filter_vertex_vertex(rVerlet, vai[i], shpi->m_radius, vaj[j], shpj->m_radius);
  }

  // API filter_vertex_edge
  ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_edge(
      const double rVerlet, 
      const Vec3d &pi, 
      const int i, 
      const shape *shpi, 
      const exanb::Quaternion &oi, 
      const Vec3d &pj, 
      const int j, 
      const shape *shpj, 
      const exanb::Quaternion &oj)
  {
    // === compute vertice positions
    auto [first, second] = shpj->get_edge(j);
    const Vec3d vi = shpi->get_vertex(i, pi, oi);
    const Vec3d vf = shpj->get_vertex(first, pj, oj);
    const Vec3d vs = shpj->get_vertex(second, pj, oj);
    double ri = shpi->m_radius;
    double rj = shpj->m_radius;
    return filter_vertex_edge_core(rVerlet, vi, ri, vf, vs, rj);
  }

  ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_edge(
      const double rVerlet, 
      const Vec3d &pi, 
      const double radius, 
      const Vec3d &pj, 
      const int j, 
      const shape *shpj, 
      const exanb::Quaternion &oj)
  {
    // === compute vertice positions
    auto [first, second] = shpj->get_edge(j);
    const Vec3d vf = shpj->get_vertex(first, pj, oj);
    const Vec3d vs = shpj->get_vertex(second, pj, oj);
    double rj = shpj->m_radius;
    return filter_vertex_edge_core(rVerlet, pi, radius, vf, vs, rj);
  }

  // API detection_vertex_edge
  ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_edge(
      const double rVerlet, 
      const VertexArray &vai, 
      const int i, 
      const shape *shpi, 
      const VertexArray &vaj, 
      const int j, 
      const shape *shpj)
  {
    const Vec3d &vi = vai[i];
    auto [first, second] = shpj->get_edge(j);
    const Vec3d &vf = vaj[first];
    const Vec3d &vs = vaj[second];
    double ri = shpi->m_radius;
    double rj = shpj->m_radius;
    return filter_vertex_edge_core(rVerlet, vi, ri, vf, vs, rj);
  }


  template <bool SKIP> ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_edge(
      const OBB &obb_vertex, 
      const Vec3d &position, 
      const int index, 
      const shape *shp, 
      const exanb::Quaternion &orientation)
  {
    if constexpr (SKIP)
    {
      return true;
    }
    else
    {
      // obb_i as already been enlarged with rVerlet
      OBB obb_edge = shp->get_obb_edge(position, index, orientation);
      return obb_vertex.intersect(obb_edge);
    }
  }


  // API detection_vertex_edge
  ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_edge(
      const Vec3d &pi, 
      const int i, 
      const shape *shpi, 
      const exanb::Quaternion &oi, 
      const Vec3d &pj, 
      const int j, 
      const shape *shpj, 
      const exanb::Quaternion &oj)
  {
    // === compute vertice positions
    auto [first, second] = shpj->get_edge(j);
    const Vec3d vi = shpi->get_vertex(i, pi, oi);
    const Vec3d vf = shpj->get_vertex(first, pj, oj);
    const Vec3d vs = shpj->get_vertex(second, pj, oj);
    double ri = shpi->m_radius;
    double rj = shpj->m_radius;
    return detection_vertex_edge_core(vi, ri, vf, vs, rj);
  }

  ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_edge(
      const Vec3d &pi, 
      const double radius, 
      const Vec3d &pj, 
      const int j, 
      const shape *shpj, 
      const exanb::Quaternion &oj)
  {
    // === compute vertice positions
    auto [first, second] = shpj->get_edge(j);
    const Vec3d vf = shpj->get_vertex(first, pj, oj);
    const Vec3d vs = shpj->get_vertex(second, pj, oj);
    double rj = shpj->m_radius;
    return detection_vertex_edge_core(pi, radius, vf, vs, rj);
  }

  // API detection_vertex_edge
  ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_edge_precompute(
      const VertexArray &vai, 
      const int i, 
      const shape *shpi, 
      const VertexArray &vaj, 
      const int j, 
      const shape *shpj)
  {
    const Vec3d &vi = vai[i];
    auto [first, second] = shpj->get_edge(j);
    const Vec3d &vf = vaj[first];
    const Vec3d &vs = vaj[second];
    double ri = shpi->m_radius;
    double rj = shpj->m_radius;
    return detection_vertex_edge_core(vi, ri, vf, vs, rj);
  }

  /**
   * @brief Filters vertex-face interactions based on a specified condition.
   * @tparam SKIP Flag indicating whether to skip the filtering process.
   * @param obb_vertex The oriented bounding box representing the vertex.
   * @param position The position of the vertex.
   * @param index The index of the face.
   * @param shp The shape associated with polyhedron.
   * @param orientation The orientation of polyhedron.
   *
   * @return True if the interaction passes the filtering condition, false otherwise.
   */
  template <bool SKIP> ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_face(
      const OBB &obb_vertex, 
      const Vec3d &position, 
      const int index, 
      const shape *shp, 
      const exanb::Quaternion &orientation)
  {
    if constexpr (SKIP)
    {
      return true;
    }
    else
    {
      // obb_i as already been enlarged with rVerlet
      OBB obb_face = shp->get_obb_face(position, index, orientation);
      return obb_vertex.intersect(obb_face);
    }
  }

  // API edge - edge 
  ONIKA_HOST_DEVICE_FUNC inline bool filter_edge_edge(
      const double rVerlet,
      const VertexArray &vai, 
      const int i,
      const shape *shpi,
      const VertexArray &vaj,
      const int j,
      const shape *shpj)
  {
    double ri = shpi->m_radius;
    double rj = shpj->m_radius;
    // === compute vertices from shapes
    auto [fi, si] = shpi->get_edge(i);
    const Vec3d &vfi = vai[fi];
    const Vec3d &vsi = vai[si];

    auto [fj, sj] = shpj->get_edge(j);
    const Vec3d &vfj = vaj[fj];
    const Vec3d &vsj = vaj[sj];
    return filter_edge_edge_core(rVerlet, vfi, vsi, ri, vfj, vsj, rj);
  }

  // API edge - edge
  ONIKA_HOST_DEVICE_FUNC inline contact detection_edge_edge(
      const Vec3d &pi, 
      const int i, 
      const shape *shpi, 
      const exanb::Quaternion &oi, 
      const Vec3d &pj, 
      const int j, 
      const shape *shpj, 
      const exanb::Quaternion &oj)
  {
    double ri = shpi->m_radius;
    double rj = shpj->m_radius;
    // === compute vertices from shapes
    auto [fi, si] = shpi->get_edge(i);
    const Vec3d vfi = shpi->get_vertex(fi, pi, oi);
    const Vec3d vsi = shpi->get_vertex(si, pi, oi);

    auto [fj, sj] = shpj->get_edge(j);
    const Vec3d vfj = shpj->get_vertex(fj, pj, oj);
    const Vec3d vsj = shpj->get_vertex(sj, pj, oj);
    return detection_edge_edge_core(vfi, vsi, ri, vfj, vsj, rj);
  }

  // API edge - edge
  ONIKA_HOST_DEVICE_FUNC inline contact detection_edge_edge_precompute(
      const VertexArray &vai, 
      const int i, 
      const shape *shpi, 
      const VertexArray &vaj, 
      const int j, 
      const shape *shpj)
  {
    double ri = shpi->m_radius;
    double rj = shpj->m_radius;
    // === compute vertices from shapes
    auto [fi, si] = shpi->get_edge(i);
    const Vec3d &vfi = vai[fi];
    const Vec3d &vsi = vai[si];

    auto [fj, sj] = shpj->get_edge(j);
    const Vec3d &vfj = vaj[fj];
    const Vec3d &vsj = vaj[sj];
    return detection_edge_edge_core(vfi, vsi, ri, vfj, vsj, rj);
  }

# define __params__ const VertexArray &vai, const int i, const shape *shpi, const VertexArray &vaj, const int j, const shape *shpj
# define __params__use__ vai, i, shpi, vaj, j, shpj
  template<int INTERACTION_TYPE> struct detect{};
  template<> struct detect<0>{ ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return detection_vertex_vertex_precompute(__params__use__); } };
  template<> struct detect<1>{ ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return detection_vertex_edge_precompute(__params__use__); } };
  template<> struct detect<2>{ ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return detection_vertex_face_precompute(__params__use__); } };
  template<> struct detect<3>{ ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return detection_edge_edge_precompute(__params__use__); } };
# undef __params__
# undef __params__use__

  /** This part concerns detection between particles and “stl mesh”.  **/
# define __params__ const Vec3d &pi, const int i, const shape *shpi, const exanb::Quaternion &oi, const Vec3d &pj, const int j, const shape *shpj, const exanb::Quaternion &oj 
# define __params__sph__ const Vec3d &pi, const double ri, const Vec3d &pj, const int j, const shape *shpj, const exanb::Quaternion &oj 
# define __params__use__      pi, i, shpi, oi, pj, j, shpj, oj
# define __params__use__sph__     pi, ri, pj, j, shpj, oj
# define __params__use__inv__ pj, j, shpj, oj, pi, i, shpi, oi
  template<> struct detect<7>{ 
    ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return detection_vertex_vertex(__params__use__); } // polyhedron
    ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__sph__) const { return detection_vertex_vertex(__params__use__sph__); } // sphere
  };
  template<> struct detect<8>{ 
    ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return detection_vertex_edge(__params__use__); } // polyhedron
    ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__sph__) const { return detection_vertex_edge(__params__use__sph__); } // sphere
  };
  template<> struct detect<9>{ 
    ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return detection_vertex_face(__params__use__); } // polyhedron 
    ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__sph__) const { return detection_vertex_face(__params__use__sph__); }  // sphere
  };
  template<> struct detect<10>{ ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return detection_edge_edge(__params__use__); } };
  template<> struct detect<11>{ ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return detection_vertex_vertex(__params__use__inv__); } };
  template<> struct detect<12>{ ONIKA_HOST_DEVICE_FUNC inline contact operator()(__params__) const { return detection_vertex_edge(__params__use__inv__); } };
# undef __params__
# undef __params__sph__
# undef __params__use__
# undef __params__use__sph__
# undef __params__use__inv__


} // namespace exaDEM
