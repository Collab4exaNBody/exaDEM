#pragma once
#include <exaDEM/driver_base.h>

namespace exaDEM
{
	using namespace exanb;

  struct Surface
	{
		double offset;
		exanb::Vec3d normal;
		exanb::Vec3d velocity = Vec3d{0,0,0};
		constexpr DRIVER_TYPE get_type() {return DRIVER_TYPE::SURFACE;}
		void print()
		{
			std::cout << "Driver Type: Surface" << std::endl;
			std::cout << "Offset: " << offset   << std::endl;
			std::cout << "Normal: " << normal   << std::endl;
			std::cout << "Vel   : " << velocity << std::endl;
		}
	};
}
