#pragma once
#include <exanb/core/basic_types.h>
#include <exaDEM/driver_base.h>

namespace exaDEM
{
	using namespace exanb;

  struct Surface
	{
		double offset;
		exanb::Vec3d normal;
		exanb::Vec3d center; // normal * offset
		exanb::Vec3d vel;   // 0,0,0
		exanb::Vec3d vrot;

		constexpr DRIVER_TYPE get_type() {return DRIVER_TYPE::SURFACE;}

		void print()
		{
			std::cout << "Driver Type: Surface" << std::endl;
			std::cout << "Offset: " << offset   << std::endl;
			std::cout << "Normal: " << normal   << std::endl;
			std::cout << "Vel   : " << vel << std::endl;
			std::cout << "AngVel: " << vrot << std::endl;
		}

		inline void initialize ()
		{
			center = normal * offset;
			// checks
			
		}

		inline void update_position ( const double t )
		{
			center = normal * offset + t * vel; 
		}

		inline bool filter( const double rcut , const exanb::Vec3d& p)
		{
			//Vec3d proj = exanb::dot(p , normal);
			Vec3d proj = dot(p , normal) * normal;
			double d = norm ( proj - center );
			return d <= rcut;
		}

		inline std::tuple<bool, double, Vec3d, Vec3d> detector( const double rcut , const Vec3d& p)
		{
			Vec3d proj = dot(p , normal) * normal;
			Vec3d surface_to_point = center - proj;
			double d = norm ( surface_to_point );
			double dn = d - rcut;
			if( dn > 0 )
			{
				return {false, 0.0, Vec3d(), Vec3d()};
			}
			else
			{
				Vec3d n = surface_to_point / d;
				Vec3d contact_position = p - n * ( rcut + 0.5 * dn ); 
				return {true, dn, n, contact_position};
			}
		}

	};
}
