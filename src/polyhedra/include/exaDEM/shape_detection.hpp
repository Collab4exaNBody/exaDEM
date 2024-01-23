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

		inline
			Vec3d normalize (Vec3d& in)
			{
				Vec3d ret = in * (1 / std::sqrt(exanb::dot(in,in)));
				return ret;
			}

		inline
		bool filter_vertex_vertex(
				const double rcut,
				const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
				const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
		{
			// sphero-polyhedron
			double ri = shpi->m_radius;
			double rj = shpj->m_radius;
			double R = ri + rj + rcut;

			// === compute vertex position
			Vec3d vi = shpi->get_vertex(i, pi, oi);
			Vec3d vj = shpj->get_vertex(j, pj, oj);

			// === compute distance
			const Vec3d dist = vi - vj;

			// === compute norm
			const double dist_norm = sqrt(exanb::dot(dist, dist));

			// === compute overlap in dn
			const double dn = dist_norm - R;

			return dn <= 0;
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

		inline
		bool filter_vertex_edge(
				const double rcut,
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
			if(r <= 0 || r >= 1.0) return false;

			// === compute normal vector
			const Vec3d n = vi - (vf + distfs * r);

			// === compute overlap in dn
			const double dn = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z)  - (ri + rj + rcut);

			return dn <= 0;
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

			// === compute normal vector
			Vec3d n = vi - (vf + distfs * r);

			// === compute overlap in dn
			const double dn = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z)  - (ri + rj);

			n = normalize(n);

			// === compute contact position
			const Vec3d contact_position = vi - n * (ri + 0.5 * dn);

			return {dn <= 0, dn, n, contact_position};
		}

		inline
		bool filter_vertex_face(
				const double rcut,
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
			v1 = normalize(v1);

			// === compute normal vector
			Vec3d n = cross(v1,v2);
			n = normalize(n);

			// === eliminate possibility
			double dist = exanb::dot(n, v);

			if(dist < 0) 
			{
				n = n * (-1);
				dist = -dist;
			}

			if( dist > (ri + rj + rcut)) return false; 
			const Vec3d P = vi - n * dist;

			size_t ODD = 0;
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
			const double dn = dist  - (ri + rj + rcut);

			return ODD && dn <=0;
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
			v1 = normalize(v1);

			// === compute normal vector
			Vec3d n = cross(v1,v2);
			n = normalize(n);

			// === eliminate possibility
			double dist = exanb::dot(n, v);

			if(dist < 0) 
			{
				n = n * (-1);
				dist = -dist;
			}

			if( dist > (ri + rj)) return {false, 0.0, Vec3d(), Vec3d()}; 
			const Vec3d P = vi - n * dist;

			size_t ODD = 0;
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

			return {ODD && dn <=0, dn, n, contact_position};
		}

		inline
		bool filter_edge_edge(
				const double rcut,
				const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
				const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
		{
#define _EPSILON_VALUE_ 1.0e-12
			double ri = shpi->m_radius;
			double rj = shpj->m_radius;

			// === compute vertices from shapes
			auto [fi, si] = shpi->get_edge(i);
			auto [fj, sj] = shpj->get_edge(j);

			const Vec3d vfi = shpi->get_vertex(fi, pi, oi);
			const Vec3d vsi = shpi->get_vertex(si, pi, oi);
			const Vec3d vfj = shpj->get_vertex(fj, pj, oj);
			const Vec3d vsj = shpj->get_vertex(sj, pj, oj);

			const Vec3d Ei = vsi - vfi;
			const Vec3d Ej = vsj - vfj;
			const Vec3d v = vfi - vfj;
			double c = exanb::dot(Ei, Ei);
			double d = exanb::dot(Ej, Ej);
			double e = exanb::dot(Ei, Ej);
			double f = (c * d) - (e * e);
			double s, t;
			if (std::abs(f) >  _EPSILON_VALUE_) 
			{
				f = 1.0 / f;
				double a = exanb::dot(Ei, v);
				double b = exanb::dot(Ej, v);
				s = (e * b - a * d) * f;  // for edge i
				t = (c * b - e * a) * f;  // for edge j
				if (s <= 0.0 || s >= 1.0 || t <= 0.0 || t >= 1.0) 
				{
					return false;
				}
				const Vec3d ji = (vfi + Ei * s ) - (vfj + Ej * t);  // from j to i

				// === compute overlap in dn
				const double dn =  std::sqrt(exanb::dot(ji,ji))  - (ri + rj + rcut);
				return dn <=0;
			}

			return false;
#undef _EPSILON_VALUE_
		}

		inline
		std::tuple<bool, double, Vec3d, Vec3d> detection_edge_edge(
				const Vec3d& pi, const int i, const shape* shpi, const exanb::Quaternion& oi, 
				const Vec3d& pj, const int j, const shape* shpj, const exanb::Quaternion& oj)
		{
#define _EPSILON_VALUE_ 1.0e-12
			double ri = shpi->m_radius;
			double rj = shpj->m_radius;

			// === compute vertices from shapes
			auto [fi, si] = shpi->get_edge(i);
			auto [fj, sj] = shpj->get_edge(j);

			const Vec3d vfi = shpi->get_vertex(fi, pi, oi);
			const Vec3d vsi = shpi->get_vertex(si, pi, oi);
			const Vec3d vfj = shpj->get_vertex(fj, pj, oj);
			const Vec3d vsj = shpj->get_vertex(sj, pj, oj);

			const Vec3d Ei = vsi - vfi;
			const Vec3d Ej = vsj - vfj;
			const Vec3d v = vfi - vfj;
			double c = exanb::dot(Ei, Ei);
			double d = exanb::dot(Ej, Ej);
			double e = exanb::dot(Ei, Ej);
			double f = (c * d) - (e * e);
			double s, t;
			if (std::abs(f) >  _EPSILON_VALUE_) 
			{
				f = 1.0 / f;
				double a = exanb::dot(Ei, v);
				double b = exanb::dot(Ej, v);
				s = (e * b - a * d) * f;  // for edge i
				t = (c * b - e * a) * f;  // for edge j
				if (s <= 0.0 || s >= 1.0 || t <= 0.0 || t >= 1.0) 
				{
					return {false, 0.0, Vec3d(), Vec3d()};
				}
				Vec3d ji = (vfi + Ei * s ) - (vfj + Ej * t);  // from j to i

				// === compute overlap in dn
				const double dn =  std::sqrt(exanb::dot(ji,ji))  - (ri + rj);

				// === compute normal vector
				const Vec3d n = normalize(ji);

				// === compute contact position
				const Vec3d contact_position = vfi + Ei * s - n * (ri + 0.5 * dn);

				return {dn <=0, dn, n, contact_position};

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


		inline
		std::tuple<bool, double, Vec3d, Vec3d> detection_vertex_cylinder(
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
			const double dn = radius - (ri + d);
			const Vec3d n = -dir / d;
			//const Vec3d contact_position = (vi - n * (ri + 0.5 * dn))*axis;
			const Vec3d contact_position = vi - n * (ri + 0.5 * dn);
			return {dn <= 0, dn, n, contact_position};
		}
	}
}
