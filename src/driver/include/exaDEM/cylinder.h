#pragma once

#include <exaDEM/driver_base.h>

namespace exaDEM
{
	using namespace exanb;

	struct Cylinder
	{
		double radius;
		exanb::Vec3d axis;
		exanb::Vec3d center;
		exanb::Vec3d vel;
		exanb::Vec3d vrot;

		constexpr DRIVER_TYPE get_type() {return DRIVER_TYPE::CYLINDER;}
		void print()
		{
			std::cout << "Driver Type: Cylinder" << std::endl;
			std::cout << "Radius: " << radius << std::endl;
			std::cout << "Axis  : " << axis << std::endl;
			std::cout << "Center: " << center << std::endl;
			std::cout << "Vel   : " << vel << std::endl;
			std::cout << "AngVel: " << vrot << std::endl;
		}

		//rcut = rverlet + r shape
		inline bool filter( const double rcut, const Vec3d& vi)
		{
			const Vec3d proj = vi * axis;

			// === direction
			const auto dir = proj - center;

			// === interpenetration
			const double d = norm(dir);
			const double dn = radius - (d + rcut);
			return dn <= 0;
		}

		/**
		 * @brief Detects the intersection between a vertex of a polyhedron and a cylinder.
		 *
		 * This function checks if a vertex, represented by its position 'pi' and orientation 'oi',
		 * intersects with a cylindrical shape defined by its center projection 'center_proj', axis 'axis',
		 * and radius 'radius'.
		 *
		 * @param rcut The shape radius. 
		 * @param pi The position of the vertex.
		 *
		 * @return A tuple containing:
		 *   - A boolean indicating whether there is an intersection (true) or not (false).
		 *   - The penetration depth in case of intersection.
		 *   - The contact normal at the intersection point.
		 *   - The contact point between the vertex and the cylinder.
		 */
		// rcut = r shape
		inline std::tuple<bool, double, Vec3d, Vec3d> detector(const double rcut, const Vec3d& pi)
		{
			// === project the vertex in the plan as the cylinder center
			const Vec3d proj = pi * axis;

			// === direction
			const Vec3d dir = center - proj;

			// === interpenetration
			const double d = exanb::norm(dir);

			// === compute interpenetration
			const double dn = radius - (rcut + d);

			if ( dn > 0 )
			{
				return {false, 0.0, Vec3d(), Vec3d()};
			}
			else
			{
				// === compute contact normal 
				const Vec3d n = dir / d;

				// === compute contact point
				const Vec3d contact_position = pi - n * (rcut + 0.5 * dn);

				return {true, dn, n, contact_position};
			}
		}
	};
}
