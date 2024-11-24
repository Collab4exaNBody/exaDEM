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

  /**
   * @brief Normalizes a 3D vector in-place.
   *
   * @param in The 3D vector to be normalized.
   *
   * @note If the input vector has a length of zero, the behavior is undefined.
   * @note The input vector is modified in-place, and the normalized vector is also returned.
   * @note It is recommended to ensure that the input vector is non-zero before calling this function.
   */
  ONIKA_HOST_DEVICE_FUNC inline void normalize(Vec3d &in)
  {
    const double norm = exanb::norm(in);
    in = in / norm;
  }

  // This function returns : if there is a contact, interpenetration value, normal vector, and the contact position
  ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_vertex_core(const Vec3d &pi, double ri, const Vec3d &pj, double rj)
  {
    // sphero-polyhedron
    double R = ri + rj;

    // === compute distance
    const Vec3d dist = pi - pj;

    // === compute norm
    const double dist_norm = exanb::norm(dist);

    // === inv norm
    const double inv_dist_norm = 1.0 / dist_norm;

    // === compute overlap in dn
    const double dn = dist_norm - R;

    if (dn > 0)
    {
      return contact();
    }
    else
    {
      // === normal vector
      const Vec3d n = dist * inv_dist_norm;

      // === compute contact position
      const Vec3d contact_position = pi - n * (ri + 0.5 * dn);

      return {true, dn, n, contact_position};
    }
  }

  /**
   * @brief Filters vertex-vertex interactions based on a specified Verlet radius.
   * @param rVerlet The Verlet radius used for filtering interactions.
   * @param vi The position of the first vertex.
   * @param ri The radius of the first vertex.
   * @param vj The position of the second vertex.
   * @param rj The radius of the second vertex.
   * @return True if the distance between the vertices is less than or equal to the Verlet radius + shape radii, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex(const double rVerlet, const Vec3d &vi, double ri, const Vec3d &vj, double rj)
  {
    // sphero-polyhedron
    double R = ri + rj + rVerlet;

    // === compute distance
    const Vec3d dist = vi - vj;

    const double d2 = exanb::dot(dist, dist);
    return d2 <= R * R;
  }

  
  ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_edge_core(const double rVerlet, const Vec3d &vi, const double ri, const Vec3d &vf, const Vec3d &vs, const double rj)
  {
    // === compute distances
    const Vec3d distfs = vs - vf;
    const Vec3d distfi = vi - vf;
    double r = (exanb::dot(distfs, distfi)) / (exanb::dot(distfs, distfs));
    if (r < -rVerlet || r > 1.0 + rVerlet)
      return false;

    // === compute minimal distance between the vertex and the edge
    Vec3d dist = distfi - distfs * r;

    // === r max
    const double Rmax = ri + rj + rVerlet;
  
    return (exanb::dot(dist, dist) < (Rmax * Rmax) );
  }

  /**
   * @brief Detects vertex-edge interactions and computes contact information.
   *
   * This function detects vertex-edge interactions and computes contact information.
   * It takes the position and radius of a vertex 'vi' and the endpoints 'vf' and 'vs' of an edge.
   *
   * @param vi The position of the vertex (belong to polyhedron i).
   * @param ri The radius of the polyhedron i.
   * @param vf The position of the first endpoint of the edge (polyhedron j).
   * @param vs The position of the second endpoint of the edge (polyhedron j).
   * @param rj The radius of the polyhedron j.
   *
   * @return A tuple containing:
   *         - A boolean indicating if there is a contact.
   *         - The interpenetration value if there is a contact.
   *         - The normal vector of the contact.
   *         - The contact position.
   */
  ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_edge_core(const Vec3d &vi, const double ri, const Vec3d &vf, const Vec3d &vs, const double rj)
  {
    // === compute distances
    const Vec3d distfs = vs - vf;
    const Vec3d distfi = vi - vf;
    double r = (exanb::dot(distfs, distfi)) / (exanb::dot(distfs, distfs));
    if (r <= 0 || r >= 1.0)
      return  contact();

    // === compute normal direction
    Vec3d n = distfi - distfs * r;

    // === compute overlap in dn
    const double dn = exanb::norm(n) - (ri + rj);

    if (dn > 0)
    {
      return contact();
    }
    else
    {
      // compute normal vector
      normalize(n);

      // === compute contact position
      const Vec3d contact_position = vi - n * (ri + 0.5 * dn);

      return {true, dn, n, contact_position};
    }
  }

  /**
   * @brief Detects vertex-face interactions and computes contact information.
   * @param pi The position vector of polyhedron i.
   * @param i The index of the vertex.
   * @param shpi The shape associated with the polyhedron i.
   * @param oi The orientation of the polyhedron i.
   * @param pj The position vector of polyhedron j.
   * @param j The index of the face.
   * @param shpj The shape associated with polyhedron j.
   * @param oj The orientation of polyhedron j.
   *
   * @return A tuple containing:
   *         - A boolean indicating if there is a contact.
   *         - The penetration depth if there is a contact.
   *         - The normal vector of the contact.
   *         - The contact position.
   */
  ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_face(const Vec3d &pi, const int i, const shape *shpi, const exanb::Quaternion &oi, const Vec3d &pj, const int j, const shape *shpj, const exanb::Quaternion &oj)
  {
    double ri = shpi->m_radius;
    double rj = shpj->m_radius;

    // === compute vertices
    const Vec3d vi = shpi->get_vertex(i, pi, oi);
    auto [data, nf] = shpj->get_face(j);
    assert(nf >= 3);
    Vec3d va = shpj->get_vertex(data[0], pj, oj);
    Vec3d vb = shpj->get_vertex(data[1], pj, oj);
    const Vec3d vc = shpj->get_vertex(data[nf - 1], pj, oj);

    const Vec3d v = vi - va;
    Vec3d v1 = vb - va;
    Vec3d v2 = vc - va;
    normalize(v1);
    //      v2 = normalize(v2);

    // === compute normal vector
    Vec3d n = cross(v1, v2);
    normalize(n);

    // === eliminate possibility
    double dist = exanb::dot(n, v);

    if (dist < 0)
    {
      n = n * (-1);
      dist = -dist;
    }

    if (dist > (ri + rj))
      return contact();

    const Vec3d P = vi - n * dist;

    int ODD = 0;
    v2 = cross(n, v1);
    double ori1 = exanb::dot(P, v1);
    double ori2 = exanb::dot(P, v2);
    double pa1, pa2;
    double pb1, pb2;
    int iva, ivb;
    for (iva = 0; iva < nf; ++iva)
    {
      ivb = iva + 1;
      if (ivb == nf)
        ivb = 0;
      va = shpj->get_vertex(data[iva], pj, oj);
      vb = shpj->get_vertex(data[ivb], pj, oj);
      pa1 = exanb::dot(va, v1);
      pb1 = exanb::dot(vb, v1);
      pa2 = exanb::dot(va, v2);
      pb2 = exanb::dot(vb, v2);

      // @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
      // @see http://alienryderflex.com/polygon/
      if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2))
      {
        if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1)
        {
          ODD = 1 - ODD;
        }
      }
    }

    // === compute overlap in dn
    const double dn = dist - (ri + rj);

    // === compute contact position
    const Vec3d contact_position = vi - n * (ri + 0.5 * dn);

    return {ODD == 1, dn, n, contact_position};
  }

  ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_face(const Vec3d &pi, const double radius, const Vec3d &pj, const int j, const shape *shpj, const exanb::Quaternion &oj)
  {
    double ri = radius;
    double rj = shpj->m_radius;

    // === compute vertices
    const Vec3d vi = pi;
    auto [data, nf] = shpj->get_face(j);
    assert(nf >= 3);
    Vec3d va = shpj->get_vertex(data[0], pj, oj);
    Vec3d vb = shpj->get_vertex(data[1], pj, oj);
    const Vec3d vc = shpj->get_vertex(data[nf - 1], pj, oj);

    const Vec3d v = vi - va;
    Vec3d v1 = vb - va;
    Vec3d v2 = vc - va;
    normalize(v1);
    //      v2 = normalize(v2);

    // === compute normal vector
    Vec3d n = cross(v1, v2);
    normalize(n);

    // === eliminate possibility
    double dist = exanb::dot(n, v);

    if (dist < 0)
    {
      n = n * (-1);
      dist = -dist;
    }

    if (dist > (ri + rj))
      return contact();

    const Vec3d P = vi - n * dist;

    int ODD = 0;
    v2 = cross(n, v1);
    double ori1 = exanb::dot(P, v1);
    double ori2 = exanb::dot(P, v2);
    double pa1, pa2;
    double pb1, pb2;
    int iva, ivb;
    for (iva = 0; iva < nf; ++iva)
    {
      ivb = iva + 1;
      if (ivb == nf)
        ivb = 0;
      va = shpj->get_vertex(data[iva], pj, oj);
      vb = shpj->get_vertex(data[ivb], pj, oj);
      pa1 = exanb::dot(va, v1);
      pb1 = exanb::dot(vb, v1);
      pa2 = exanb::dot(va, v2);
      pb2 = exanb::dot(vb, v2);

      // @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
      // @see http://alienryderflex.com/polygon/
      if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2))
      {
        if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1)
        {
          ODD = 1 - ODD;
        }
      }
    }

    // === compute overlap in dn
    const double dn = dist - (ri + rj);

    // === compute contact position
    const Vec3d contact_position = vi - n * (ri + 0.5 * dn);

    return {ODD == 1, dn, n, contact_position};
  }

  /**
   * @brief Detects vertex-face interactions and computes contact information.
   * @param vai The array of vertices of polyhedron i.
   * @param i The index of the vertex.
   * @param shpi The shape associated with polyhedron i.
   * @param vaj The array of vertices of polyhedron j.
   * @param j The index of the face.
   * @param shpj The shape associated with polyhedron j.
   *
   * @return A tuple containing:
   *         - A boolean indicating if there is a contact.
   *         - The penetration depth if there is a contact.
   *         - The normal vector of the contact.
   *         - The contact position.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_face(
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

    const Vec3d &vi = vai[i];

    // === compute vertices
    auto [data, nf] = shpj->get_face(j);
    assert(nf >= 3);
    const Vec3d &va = vaj[data[0]];
    const Vec3d &vb = vaj[data[1]];
    const Vec3d &vc = vaj[data[nf - 1]];
    const Vec3d v = vi - va;
    Vec3d v1 = vb - va;
    Vec3d v2 = vc - va;
    normalize(v1);

    // === compute normal vector
    Vec3d n = cross(v1, v2);
    normalize(n);

    // === eliminate possibility
    double dist = exanb::dot(n, v);

    if (dist < 0)
    {
      n = n * (-1);
      dist = -dist;
    }

    if (dist > (ri + rj + rVerlet))
      return false;

    const Vec3d P = vi - n * dist;

    int ODD = 0;
    v2 = cross(n, v1);
    double ori1 = exanb::dot(P, v1);
    double ori2 = exanb::dot(P, v2);
    double pa1, pa2;
    double pb1, pb2;
    int iva, ivb;
    for (iva = 0; iva < nf; ++iva)
    {
      ivb = iva + 1;
      if (ivb == nf)
        ivb = 0;
      const Vec3d &_va = vaj[data[iva]];
      const Vec3d &_vb = vaj[data[ivb]];
      pa1 = exanb::dot(_va, v1);
      pb1 = exanb::dot(_vb, v1);
      pa2 = exanb::dot(_va, v2);
      pb2 = exanb::dot(_vb, v2);

      // @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
      // @see http://alienryderflex.com/polygon/
      if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2))
      {
        if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1)
        {
          ODD = 1 - ODD;
        }
      }
    }

    if(ODD == 1) 
    {
      return true;
    }
    return false;
  }

  /**
   * @brief Detects vertex-face interactions and computes contact information.
   * @param vai The array of vertices of polyhedron i.
   * @param i The index of the vertex.
   * @param shpi The shape associated with polyhedron i.
   * @param vaj The array of vertices of polyhedron j.
   * @param j The index of the face.
   * @param shpj The shape associated with polyhedron j.
   *
   * @return A tuple containing:
   *         - A boolean indicating if there is a contact.
   *         - The penetration depth if there is a contact.
   *         - The normal vector of the contact.
   *         - The contact position.
   */
  ONIKA_HOST_DEVICE_FUNC inline contact detection_vertex_face_precompute(const VertexArray &vai, const int i, const shape *shpi, const VertexArray &vaj, const int j, const shape *shpj)
  {
    double ri = shpi->m_radius;
    double rj = shpj->m_radius;

    const Vec3d &vi = vai[i];

    // === compute vertices
    auto [data, nf] = shpj->get_face(j);
    assert(nf >= 3);
    const Vec3d &va = vaj[data[0]];
    const Vec3d &vb = vaj[data[1]];
    const Vec3d &vc = vaj[data[nf - 1]];
    const Vec3d v = vi - va;
    Vec3d v1 = vb - va;
    Vec3d v2 = vc - va;
    normalize(v1);
    //      v2 = normalize(v2);

    // === compute normal vector
    Vec3d n = cross(v1, v2);
    normalize(n);

    // === eliminate possibility
    double dist = exanb::dot(n, v);

    if (dist < 0)
    {
      n = n * (-1);
      dist = -dist;
    }


    if (dist > (ri + rj))
      return contact();


    const Vec3d P = vi - n * dist;

    int ODD = 0;
    v2 = cross(n, v1);
    double ori1 = exanb::dot(P, v1);
    double ori2 = exanb::dot(P, v2);
    double pa1, pa2;
    double pb1, pb2;
    int iva, ivb;
    for (iva = 0; iva < nf; ++iva)
    {
      ivb = iva + 1;
      if (ivb == nf)
        ivb = 0;
      const Vec3d &_va = vaj[data[iva]];
      const Vec3d &_vb = vaj[data[ivb]];
      pa1 = exanb::dot(_va, v1);
      pb1 = exanb::dot(_vb, v1);
      pa2 = exanb::dot(_va, v2);
      pb2 = exanb::dot(_vb, v2);

      // @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
      // @see http://alienryderflex.com/polygon/
      if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2))
      {
        if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1)
        {
          ODD = 1 - ODD;
        }
      }
    }

    // === compute overlap in dn
    const double dn = dist - (ri + rj);

    // === compute contact position
    const Vec3d contact_position = vi - n * (ri + 0.5 * dn);

    return {ODD == 1, dn, n, contact_position};
  }

  ONIKA_HOST_DEVICE_FUNC inline bool filter_edge_edge_core(
      const double rVerlet,
      const Vec3d &vfi, 
      const Vec3d &vsi, 
      const double ri, 
      const Vec3d &vfj, 
      const Vec3d &vsj, 
      const double rj)
  {
#   define _EPSILON_VALUE_ 1.0e-12
    const double R = ri + rj + rVerlet;

    const Vec3d Ei = vsi - vfi;
    const Vec3d Ej = vsj - vfj;
    const Vec3d v = vfi - vfj;

    const double c = exanb::dot(Ei, Ei);
    const double d = exanb::dot(Ej, Ej);
    const double e = exanb::dot(Ei, Ej);
    double f = (c * d) - (e * e);
    double s, t;
    if (fabs(f) > _EPSILON_VALUE_)
    {
      f = 1.0 / f;
      const double a = exanb::dot(Ei, v);
      const double b = exanb::dot(Ej, v);
      s = (e * b - a * d) * f; // for edge i
      t = (c * b - e * a) * f; // for edge j
      if (s <= 0.0 || s >= 1.0 || t <= 0.0 || t >= 1.0)
      {
        return false;
      }

      Vec3d pi = vfi + Ei * s;
      Vec3d pj = vfj + Ej * t;

      Vec3d n = pi - pj; // from j to i

      // === compute overlap in dn
      return (exanb::dot(n,n) < (R*R));
    }
    return false;
#undef _EPSILON_VALUE_
  }

  /**
   * @brief Detects edge-edge interactions and computes contact information.
   * @param vfi The position vector of the first endpoint of edge i.
   * @param vsi The position vector of the second endpoint of edge i.
   * @param ri The radius of polyhedron i.
   * @param vfj The position vector of the first endpoint of edge j.
   * @param vsj The position vector of the second endpoint of edge j.
   * @param rj The radius of polyhedron j.
   * @return A tuple containing:
   *         - A boolean indicating if there is a contact.
   *         - The penetration depth if there is a contact.
   *         - The normal vector of the contact.
   *         - The contact position.
   */
  ONIKA_HOST_DEVICE_FUNC inline contact detection_edge_edge_core(const Vec3d &vfi, const Vec3d &vsi, const double ri, const Vec3d &vfj, const Vec3d &vsj, const double rj)
  {
#   define _EPSILON_VALUE_ 1.0e-12
    const double R = ri + rj;

    const Vec3d Ei = vsi - vfi;
    const Vec3d Ej = vsj - vfj;
    const Vec3d v = vfi - vfj;

    const double c = exanb::dot(Ei, Ei);
    const double d = exanb::dot(Ej, Ej);
    const double e = exanb::dot(Ei, Ej);
    double f = (c * d) - (e * e);
    double s, t;
    if (fabs(f) > _EPSILON_VALUE_)
    {
      f = 1.0 / f;
      const double a = exanb::dot(Ei, v);
      const double b = exanb::dot(Ej, v);
      s = (e * b - a * d) * f; // for edge i
      t = (c * b - e * a) * f; // for edge j
      if (s <= 0.0 || s >= 1.0 || t <= 0.0 || t >= 1.0)
      {
        return contact();
      }

      Vec3d pi = vfi + Ei * s;
      Vec3d pj = vfj + Ej * t;

      Vec3d n = pi - pj; // from j to i

      // === compute overlap in dn
      const double dn = exanb::norm(n) - R;

      if (dn > 0)
      {
        return contact();
      }
      else
      {
        // === compute normal vector
        normalize(n);

        // === compute contact position
        const Vec3d contact_position = pi - n * (ri + 0.5 * dn);
        return {dn <= 0, dn, n, contact_position};
      }
    }

    return contact();
#undef _EPSILON_VALUE_
  }
} // namespace exaDEM
