#pragma once

#include <string.h>

namespace exaDEM
{
	enum DRIVER_TYPE
	{
		CYLINDER,
		SURFACE,
		UNDEFINED
	};

	constexpr int DRIVER_TYPE_SIZE = 3;


	inline std::string print(DRIVER_TYPE type)
	{
    switch(type)
		{
			case DRIVER_TYPE::CYLINDER: return "Cylinder";
			case DRIVER_TYPE::SURFACE: return "Surface";
			case DRIVER_TYPE::UNDEFINED: return "Undefined Driver";
			default: return "Undefined Driver";
		}
	}

	constexpr unsigned int str2int(const char* str, int h = 0)
	{
		return !str[h] ? 5381 : (str2int(str, h+1) * 33) ^ str[h];
	}

	inline DRIVER_TYPE get_type(std::string driver_name)
	{
		switch(str2int(driver_name.c_str()))
		{
			case str2int("CYLINDER"): return DRIVER_TYPE::CYLINDER;
			case str2int("SURFACE"): return DRIVER_TYPE::SURFACE;
			default: std::cout << "error, no driver " << driver_name << " found" << std::endl;
							 std::cout << "Use: CYLINDER or SURFACE" << std::endl;
							 std::abort();
		}
	}

	struct Cylinder;
	struct Surface;
	struct UndefinedDriver;

	template<typename T>
		constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::UNDEFINED;}
	template<> constexpr DRIVER_TYPE get_type<exaDEM::Cylinder> () { return DRIVER_TYPE::CYLINDER; }
	template<> constexpr DRIVER_TYPE get_type<exaDEM::Surface> () { return DRIVER_TYPE::SURFACE;	}
	template<> constexpr DRIVER_TYPE get_type<exaDEM::UndefinedDriver> () { return DRIVER_TYPE::UNDEFINED;	}

	struct Driver
	{
		virtual constexpr DRIVER_TYPE get_type();
		virtual void print();
	};
}
