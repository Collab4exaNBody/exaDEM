#pragma once

#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exaDEM/shape.hpp>
#include <math.h>

namespace exaDEM
{
	namespace shape_polyhedron
	{
		using namespace exanb;

		/**
		 * @brief Normalizes a 3D vector in-place.
		 *
		 * @param in The 3D vector to be normalized. 
		 *
		 * @note If the input vector has a length of zero, the behavior is undefined.
		 * @note The input vector is modified in-place, and the normalized vector is also returned.
		 * @note It is recommended to ensure that the input vector is non-zero before calling this function.
		 */
		inline
			void normalize (Vec3d& in)
			{
				const double norm = exanb::norm (in);
				in = in / norm ;
			}

		// This function returns : if there is a contact, interpenetration value, normal vector, and the contact position
		inline
			std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_vertex(
					const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
					const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
			{
				// sphero-polyhedron
				double ri = shpi->m_radius;
				double rj = shpj->m_radius;
				double R = ri + rj;

				// === compute vertex position
				Vec3d vi = shpi->get_vertex(i, pi, oi);
				Vec3d vj = shpj->get_vertex(j, pj, oj);

				// === compute distance
				const Vec3d dist = vi - vj;
				//const Vec3d dist = vj - vi;

				// === compute norm
				const double dist_norm = sqrt(exanb::dot(dist, dist));

				// === inv norm
				const double inv_dist_norm = 1.0 / dist_norm;

				// === normal vector
				const Vec3d n = dist * inv_dist_norm;

				// === compute overlap in dn
				const double dn = dist_norm - R;

				// === compute contact position
				const Vec3d contact_position = vi - n * (ri + 0.5 * dn);

				return {dn <= 0, dn, n, contact_position};
			}

		// This function returns : if there is a contact, interpenetration value, normal vector, and the contact position
		inline
			std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_edge(
					const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
					const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
			{
				double ri = shpi->m_radius;
				double rj = shpj->m_radius;

				// === compute vertice positions
				auto [first, second] = shpj->get_edge(j);
				const Vec3d vi = shpi->get_vertex(i, pi, oi);
				const Vec3d vf = shpj->get_vertex (first, pj, oj); 
				const Vec3d vs = shpj->get_vertex (second, pj, oj); 

				// === compute distances
				const Vec3d distfs = vs - vf;
				const Vec3d distfi = vi - vf;
				double r = (exanb::dot(distfs,distfi)) / (exanb::dot(distfs, distfs));
				if(r <= 0 || r >= 1.0) return {false, 0.0, Vec3d(), Vec3d()};

				// === compute normal direction
				Vec3d n = distfi - distfs * r;

				// === compute overlap in dn
				const double dn = exanb::norm(n)  - (ri + rj);

				// compute normal vector
				normalize(n);

				// === compute contact position
				const Vec3d contact_position = vi - n * (ri + 0.5 * dn);

				return {dn <= 0, dn, n, contact_position};
			}


		inline
			std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_face(
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


		inline
			std::tuple<bool, double, Vec3d, Vec3d> detection_edge_edge(
					const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
					const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
			{
#define _EPSILON_VALUE_ 1.0e-12
				double ri = shpi->m_radius;
				double rj = shpj->m_radius;
				const double R = ri + rj;

				// === compute vertices from shapes
				auto [fi, si] = shpi->get_edge(i);
				const Vec3d vfi = shpi->get_vertex(fi, pi, oi);
				const Vec3d vsi = shpi->get_vertex(si, pi, oi);

				auto [fj, sj] = shpj->get_edge(j);
				const Vec3d vfj = shpj->get_vertex(fj, pj, oj);
				const Vec3d vsj = shpj->get_vertex(sj, pj, oj);

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

					// === compute normal vector
					normalize(n);
			
					// === compute contact position
					const Vec3d contact_position = pj + n * (rj + 0.5 * dn);
					return {dn <= 0 , dn, n, contact_position};

				}

				return {false, 0.0, Vec3d(), Vec3d()};
#undef _EPSILON_VALUE_
			}

		inline
			bool filter_vertex_cylinder(
					const double rcut,
					const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
					const Vec3d& center_proj, const Vec3d& axis, const double radius)
			{
				const double ri = shpi->m_radius;
				const Vec3d vi = shpi->get_vertex(i, pi, oi);
				const Vec3d proj = vi * axis;

				// === direction
				const auto dir = proj - center_proj;

				// === interpenetration
				const double d = norm(dir);
				const double dn = radius - (ri + d + rcut);
				return dn <= 0;
			}

		/**
		 * @brief Detects the intersection between a vertex of a polyhedron and a cylinder.
		 *
		 * This function checks if a vertex, represented by its position 'pi' and orientation 'oi',
		 * intersects with a cylindrical shape defined by its center projection 'center_proj', axis 'axis',
		 * and radius 'radius'.
		 *
		 * @param pi The position of the vertex.
		 * @param i Index of the vertex.
		 * @param shpi Pointer to the shape of the vertex.
		 * @param oi Orientation of the polyhedron.
		 * @param center_proj The center projection of the cylinder center on the axis.
		 * @param axis The axis of the cylindrical shape.
		 * @param radius The radius of the cylinder.
		 *
		 * @return A tuple containing:
		 *   - A boolean indicating whether there is an intersection (true) or not (false).
		 *   - The penetration depth in case of intersection.
		 *   - The contact normal at the intersection point.
		 *   - The contact point between the vertex and the cylinder.
		 */
		inline
			std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_cylinder(
					const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
					const Vec3d& center_proj, const Vec3d& axis, const double radius)
			{
				// === get vertex radius and vertex position	
				const double ri = shpi->m_radius;
				const Vec3d vi = shpi->get_vertex(i, pi, oi);

				// === project the vertex in the plan as the cylinder center
				const Vec3d proj = vi * axis;

				// === direction
				const Vec3d dir = center_proj - proj;

				// === interpenetration
				const double d = exanb::norm(dir);

				// === compute interpenetration
				const double dn = radius - (ri + d);

				// === compute contact normal 
				const Vec3d n = dir / d;

				// === compute contact point
				const Vec3d contact_position = vi - n * (ri + 0.5 * dn);

				return {dn <= 0, dn, n, contact_position};
			}
	}
}
