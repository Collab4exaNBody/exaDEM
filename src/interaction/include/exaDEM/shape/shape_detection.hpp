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
	ONIKA_HOST_DEVICE_FUNC inline void normalize (Vec3d& in)
	{
		const double norm = exanb::norm (in);
		in = in / norm ;
	}

	// This function returns : if there is a contact, interpenetration value, normal vector, and the contact position
	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_vertex_core( const Vec3d& pi, double ri,  const Vec3d& pj, double rj)
	{
		// sphero-polyhedron
		double R = ri + rj;

		// === compute distance
		const Vec3d dist = pi - pj;

		// === compute norm
		const double dist_norm = sqrt(exanb::dot(dist, dist));

		// === inv norm
		const double inv_dist_norm = 1.0 / dist_norm;

		// === compute overlap in dn
		const double dn = dist_norm - R;

		if (dn > 0)
		{
			return {false, 0.0, Vec3d(), Vec3d()};
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

	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_vertex(
			const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
			const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
	{

		// === compute vertex position
		Vec3d vi = shpi->get_vertex(i, pi, oi);
		Vec3d vj = shpj->get_vertex(j, pj, oj);
		return detection_vertex_vertex_core(vi, shpi->m_radius, vj, shpj->m_radius);
	}

	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_vertex(
			const Vec3d& pi, const double radius,
			const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
	{

		// === compute vertex position
		Vec3d vj = shpj->get_vertex(j, pj, oj);
		return detection_vertex_vertex_core(pi, radius, vj, shpj->m_radius);
	}

	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_vertex_precompute(
			const VertexArray& vai, const int i, const shape* shpi,
			const VertexArray& vaj, const int j, const shape* shpj)
	{
		// === get vertex position
		const Vec3d& vi = vai[i];
		const Vec3d& vj = vaj[j];
		return detection_vertex_vertex_core(vi, shpi->m_radius, vj, shpj->m_radius);
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
	ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex( const double rVerlet, const Vec3d& vi, double ri,  const Vec3d& vj, double rj)
	{
		// sphero-polyhedron
		double R = ri + rj + rVerlet;

		// === compute distance
		const Vec3d dist = vi - vj;

		const double d2 = exanb::dot(dist, dist);
		return d2 <= R * R;
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
	ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex( const double rVerlet,
			const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
			const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
	{
		// === compute vertex position
		Vec3d vi = shpi->get_vertex(i, pi, oi);
		Vec3d vj = shpj->get_vertex(j, pj, oj);
		return filter_vertex_vertex( rVerlet, vi, shpi->m_radius, vj, shpj->m_radius);
	}

	ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex( const double rVerlet,
			const Vec3d& pi, const double radius,
			const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
	{
		// === compute vertex position
		Vec3d vj = shpj->get_vertex(j, pj, oj);
		return filter_vertex_vertex( rVerlet, pi, radius, vj, shpj->m_radius);
	}

	ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_vertex( const double rVerlet,
			const VertexArray& vai, const int i, const shape* shpi, 
			const VertexArray& vaj, const int j, const shape* shpj)
	{
		// === compute vertex position
		return filter_vertex_vertex( rVerlet, vai[i], shpi->m_radius, vaj[j], shpj->m_radius);
	}
	/**
	 * @brief Filters vertex-edge interactions based on a specified condition.
	 * @tparam SKIP Flag indicating whether to skip the filtering process.
	 * @param obb_vertex The oriented bounding box representing the edge.
	 * @param position The position of the polyhedron.
	 * @param index The index of the edge.
	 * @param shp The shape associated with the polyhedron.
	 * @param orientation The orientation of the polyhedron.
	 * @return True if the interaction passes the filtering condition, false otherwise.
	 */
	template<bool SKIP>
		ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_edge(const OBB& obb_vertex, const Vec3d& position, const int index, const shape* shp, const exanb::Quaternion& orientation)
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
	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_edge_core(
			const Vec3d& vi, 
			const double ri, 
			const Vec3d& vf, 
			const Vec3d& vs,
			const double rj)
	{
		// === compute distances
		const Vec3d distfs = vs - vf;
		const Vec3d distfi = vi - vf;
		double r = (exanb::dot(distfs,distfi)) / (exanb::dot(distfs, distfs));
		if(r <= 0 || r >= 1.0) return {false, 0.0, Vec3d(), Vec3d()};

		// === compute normal direction
		Vec3d n = distfi - distfs * r;

		// === compute overlap in dn
		const double dn = exanb::norm(n)  - (ri + rj);

		if ( dn > 0 )
		{
			return {false, 0.0, Vec3d(), Vec3d()};
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

	// API detection_vertex_edge
	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_edge(
			const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
			const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
	{
		// === compute vertice positions
		auto [first, second] = shpj->get_edge(j);
		const Vec3d vi = shpi->get_vertex(i, pi, oi);
		const Vec3d vf = shpj->get_vertex (first, pj, oj); 
		const Vec3d vs = shpj->get_vertex (second, pj, oj); 
		double ri = shpi->m_radius;
		double rj = shpj->m_radius;
		return detection_vertex_edge_core( vi, ri, vf, vs, rj);
	}

	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_edge(
			const Vec3d& pi, const double radius,
			const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
	{
		// === compute vertice positions
		auto [first, second] = shpj->get_edge(j);
		const Vec3d vf = shpj->get_vertex (first, pj, oj); 
		const Vec3d vs = shpj->get_vertex (second, pj, oj); 
		double rj = shpj->m_radius;
		return detection_vertex_edge_core( pi, radius, vf, vs, rj);
	}

	// API detection_vertex_edge
	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_edge_precompute( 
			const VertexArray& vai, const int i, const shape* shpi,
			const VertexArray& vaj, const int j, const shape* shpj)
	{
		const Vec3d& vi = vai[i]; 
		auto [first, second] = shpj->get_edge(j);
		const Vec3d& vf = vaj[first];
		const Vec3d& vs = vaj[second];
		double ri = shpi->m_radius;
		double rj = shpj->m_radius;
		return detection_vertex_edge_core( vi, ri, vf, vs, rj);
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
	template<bool SKIP>
		ONIKA_HOST_DEVICE_FUNC inline bool filter_vertex_face(const OBB& obb_vertex, const Vec3d& position, const int index, const shape* shp, const exanb::Quaternion& orientation)
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
	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_face(
			const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
			const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
	{
		double ri = shpi->m_radius;
		double rj = shpj->m_radius;

		// === compute vertices
		const Vec3d vi = shpi->get_vertex(i, pi, oi);
		auto [data, nf] = shpj->get_face(j);
		assert(nf >= 3);
		Vec3d va = shpj->get_vertex(data[0], pj, oj);
		Vec3d vb = shpj->get_vertex(data[1], pj, oj);
		const Vec3d vc = shpj->get_vertex(data[nf-1], pj, oj);

		const Vec3d v  = vi - va; 
		Vec3d v1 = vb - va; 
		Vec3d v2 = vc - va; 
		normalize(v1);
		//			v2 = normalize(v2);

		// === compute normal vector
		Vec3d n = cross(v1,v2);
		normalize(n);

		// === eliminate possibility
		double dist = exanb::dot(n, v);

		if(dist < 0) 
		{
			n = n * (-1);
			dist = -dist;
		}

		if( dist > (ri + rj)) return {false, 0.0, Vec3d(), Vec3d()}; 

		const Vec3d P = vi - n * dist;

		int ODD = 0;
		v2 = cross(n, v1);
		double ori1 = exanb::dot(P, v1);
		double ori2 = exanb::dot(P, v2);
		double pa1, pa2;
		double pb1, pb2;
		int iva, ivb;
		for (iva = 0; iva < nf; ++iva) {
			ivb = iva + 1;
			if (ivb == nf) ivb = 0;
			va = shpj->get_vertex(data[iva], pj, oj);
			vb = shpj->get_vertex(data[ivb], pj, oj);
			pa1 = exanb::dot(va, v1);
			pb1 = exanb::dot(vb, v1);
			pa2 = exanb::dot(va, v2);
			pb2 = exanb::dot(vb, v2);

			// @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
			// @see http://alienryderflex.com/polygon/
			if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2)) {
				if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1) {
					ODD = 1 - ODD;
				}
			}
		}

		// === compute overlap in dn
		const double dn = dist  - (ri + rj);

		// === compute contact position
		const Vec3d contact_position = vi - n * (ri + 0.5 * dn);

		return {ODD == 1, dn, n, contact_position};
	}

	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_face(
			const Vec3d& pi, const double radius,
			const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
	{
		double ri = radius;
		double rj = shpj->m_radius;

		// === compute vertices
		const Vec3d vi = pi;
		auto [data, nf] = shpj->get_face(j);
		assert(nf >= 3);
		Vec3d va = shpj->get_vertex(data[0], pj, oj);
		Vec3d vb = shpj->get_vertex(data[1], pj, oj);
		const Vec3d vc = shpj->get_vertex(data[nf-1], pj, oj);

		const Vec3d v  = vi - va; 
		Vec3d v1 = vb - va; 
		Vec3d v2 = vc - va; 
		normalize(v1);
		//			v2 = normalize(v2);

		// === compute normal vector
		Vec3d n = cross(v1,v2);
		normalize(n);

		// === eliminate possibility
		double dist = exanb::dot(n, v);

		if(dist < 0) 
		{
			n = n * (-1);
			dist = -dist;
		}

		if( dist > (ri + rj)) return {false, 0.0, Vec3d(), Vec3d()}; 

		const Vec3d P = vi - n * dist;

		int ODD = 0;
		v2 = cross(n, v1);
		double ori1 = exanb::dot(P, v1);
		double ori2 = exanb::dot(P, v2);
		double pa1, pa2;
		double pb1, pb2;
		int iva, ivb;
		for (iva = 0; iva < nf; ++iva) {
			ivb = iva + 1;
			if (ivb == nf) ivb = 0;
			va = shpj->get_vertex(data[iva], pj, oj);
			vb = shpj->get_vertex(data[ivb], pj, oj);
			pa1 = exanb::dot(va, v1);
			pb1 = exanb::dot(vb, v1);
			pa2 = exanb::dot(va, v2);
			pb2 = exanb::dot(vb, v2);

			// @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
			// @see http://alienryderflex.com/polygon/
			if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2)) {
				if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1) {
					ODD = 1 - ODD;
				}
			}
		}

		// === compute overlap in dn
		const double dn = dist  - (ri + rj);

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
	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_face_precompute(
			const VertexArray& vai, const int i, const shape* shpi,
			const VertexArray& vaj, const int j, const shape* shpj)
	{
		double ri = shpi->m_radius;
		double rj = shpj->m_radius;

		const Vec3d& vi = vai[i];

		// === compute vertices
		auto [data, nf] = shpj->get_face(j);
		assert(nf >= 3);
		const Vec3d& va = vaj[data[0]];
		const Vec3d& vb = vaj[data[1]];
		const Vec3d& vc = vaj[data[nf-1]];
		const Vec3d v  = vi - va; 
		Vec3d v1 = vb - va; 
		Vec3d v2 = vc - va; 
		normalize(v1);
		//			v2 = normalize(v2);

		// === compute normal vector
		Vec3d n = cross(v1,v2);
		normalize(n);

		// === eliminate possibility
		double dist = exanb::dot(n, v);

		if(dist < 0) 
		{
			n = n * (-1);
			dist = -dist;
		}

		if( dist > (ri + rj)) return {false, 0.0, Vec3d(), Vec3d()}; 

		const Vec3d P = vi - n * dist;

		int ODD = 0;
		v2 = cross(n, v1);
		double ori1 = exanb::dot(P, v1);
		double ori2 = exanb::dot(P, v2);
		double pa1, pa2;
		double pb1, pb2;
		int iva, ivb;
		for (iva = 0; iva < nf; ++iva) {
			ivb = iva + 1;
			if (ivb == nf) ivb = 0;
			const Vec3d& _va = vaj[data[iva]];
			const Vec3d& _vb = vaj[data[ivb]];
			pa1 = exanb::dot(_va, v1);
			pb1 = exanb::dot(_vb, v1);
			pa2 = exanb::dot(_va, v2);
			pb2 = exanb::dot(_vb, v2);

			// @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
			// @see http://alienryderflex.com/polygon/
			if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2)) {
				if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1) {
					ODD = 1 - ODD;
				}
			}
		}

		// === compute overlap in dn
		const double dn = dist  - (ri + rj);

		// === compute contact position
		const Vec3d contact_position = vi - n * (ri + 0.5 * dn);

		return {ODD == 1, dn, n, contact_position};
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
	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_edge_edge_core(
			const Vec3d& vfi, const Vec3d& vsi, const double ri,
			const Vec3d& vfj, const Vec3d& vsj, const double rj)
	{
#define _EPSILON_VALUE_ 1.0e-12
		const double R = ri + rj;

		const Vec3d Ei = vsi - vfi;
		const Vec3d Ej = vsj - vfj;
		const Vec3d v = vfi - vfj;

		const double c = exanb::dot(Ei, Ei);
		const double d = exanb::dot(Ej, Ej);
		const double e = exanb::dot(Ei, Ej);
		double f = (c * d) - (e * e);
		double s, t;
		if (fabs(f) >  _EPSILON_VALUE_) 
		{
			f = 1.0 / f;
			const double a = exanb::dot(Ei, v);
			const double b = exanb::dot(Ej, v);
			s = (e * b - a * d) * f;  // for edge i
			t = (c * b - e * a) * f;  // for edge j
			if (s <= 0.0 || s >= 1.0 || t <= 0.0 || t >= 1.0) 
			{
				return {false, 0.0, Vec3d(), Vec3d()};
			}

			Vec3d pi = vfi + Ei * s;
			Vec3d pj = vfj + Ej * t; 

			Vec3d n = pi - pj;  // from j to i

			// === compute overlap in dn
			const double dn = exanb::norm (n) - R;

			if (dn > 0 )
			{
				return {false, 0.0, Vec3d(), Vec3d()};
			}
			else
			{ 
				// === compute normal vector
				normalize(n);

				// === compute contact position
				const Vec3d contact_position = pj + n * (rj + 0.5 * dn);
				return {dn <= 0 , dn, n, contact_position};
			}
		}

		return {false, 0.0, Vec3d(), Vec3d()};
#undef _EPSILON_VALUE_
	}

	// API edge - edge
	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_edge_edge(
			const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
			const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
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
	ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detection_edge_edge_precompute(
			const VertexArray& vai, const int i, const shape* shpi,
			const VertexArray& vaj, const int j, const shape* shpj)
	{
		double ri = shpi->m_radius;
		double rj = shpj->m_radius;
		// === compute vertices from shapes
		auto [fi, si] = shpi->get_edge(i);
		const Vec3d& vfi = vai[fi];
		const Vec3d& vsi = vai[si];

		auto [fj, sj] = shpj->get_edge(j);
		const Vec3d& vfj = vaj[fj];
		const Vec3d& vsj = vaj[sj];
		return detection_edge_edge_core(vfi, vsi, ri, vfj, vsj, rj);
	}
}
