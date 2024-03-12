#pragma once

#include <exaDEM/driver_base.h>

namespace exaDEM
{
	using namespace exanb;

  struct Cylinder
	{
		double radius;
		exanb::Vec3d center;
		exanb::Vec3d axis;
		exanb::Vec3d angular_velocity;
    exanb::Vec3d velocity;

		constexpr DRIVER_TYPE get_type() {return DRIVER_TYPE::CYLINDER;}
		void print()
		{
			std::cout << "Driver Type: Cylinder" << std::endl;
			std::cout << "Radius: " << radius << std::endl;
			std::cout << "Center: " << center << std::endl;
			std::cout << "Axis  : " << axis << std::endl;
			std::cout << "AngVel: " << angular_velocity << std::endl;
			std::cout << "Vel   : " << velocity << std::endl;
		}
	};
}
