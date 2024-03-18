#pragma once
#include <exaDEM/driver_base.h>

namespace exaDEM
{
	using namespace exanb;

  struct UndefinedDriver
	{
		constexpr DRIVER_TYPE get_type() {return DRIVER_TYPE::UNDEFINED;}
		void print()
		{
			std::cout << "Driver Type: UNDEFINED" << std::endl;
		}
	};
}
