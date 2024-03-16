#pragma once
#include <exanb/core/basic_types.h>
#include <exaDEM/driver_base.h>

namespace exaDEM
{
	using namespace exanb;

  struct Ball
	{
		double radius;
		exanb::Vec3d center; // normal * offset
		exanb::Vec3d vel;   // 0,0,0
		exanb::Vec3d vrot;

		constexpr DRIVER_TYPE get_type() {return DRIVER_TYPE::BALL;}

		void print()
		{
			std::cout << "Driver Type: Ball"  << std::endl;
			std::cout << "Radius: " << radius << std::endl;
			std::cout << "Center: " << center << std::endl;
			std::cout << "Vel   : " << vel    << std::endl;
			std::cout << "AngVel: " << vrot   << std::endl;
		}

		inline void initialize ()
		{
			assert (radius > 0 );
		}

		inline void update_radius (const double incr)
		{
			radius += incr;
		}

		inline void update_position ( const double t )
		{
			center = normal * offset + t * vel; 
		}

		inline bool filter( const double rcut , const exanb::Vec3d& p)
		{
			const Vec3d dist = center - p;
			double d = norm ( dist );
			return d <= rcut;
		}

		inline std::tuple<bool, double, Vec3d, Vec3d> detector( const double rcut , const Vec3d& p)
		{
			Vec3d point_to_center = center - proj;
			double d = norm ( point_to_center );
			double dn = d - rcut;
			if( dn > 0 )
			{
				return {false, 0.0, Vec3d(), Vec3d()};
			}
			else
			{
				Vec3d n = point_to_center / d;
				Vec3d contact_position = p - n * ( rcut + 0.5 * dn ); 
				return {true, dn, n, contact_position};
			}
		}
	};
}
