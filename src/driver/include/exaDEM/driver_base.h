#pragma once
#include <exanb/core/basic_types.h>
#include <string.h>
#include <tuple>

namespace exaDEM
{
	using namespace exanb;
	enum DRIVER_TYPE
	{
		CYLINDER,
		SURFACE,
    BALL,
    STL_MESH,
		UNDEFINED
	};

	constexpr int DRIVER_TYPE_SIZE = 3;

	inline std::string print(DRIVER_TYPE type)
	{
    switch(type)
		{
			case DRIVER_TYPE::CYLINDER: return "Cylinder";
			case DRIVER_TYPE::SURFACE: return "Surface";
			case DRIVER_TYPE::BALL: return "Ball";
			case DRIVER_TYPE::STL_MESH: return "Stl_mesh";
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
			case str2int("BALL"): return DRIVER_TYPE::BALL;
			case str2int("STL_MESH"): return DRIVER_TYPE::STL_MESH;
			default: std::cout << "error, no driver " << driver_name << " found" << std::endl;
							 std::cout << "Use: CYLINDER, SURFACE, or BALL" << std::endl;
							 std::abort();
		}
	}

	struct Cylinder;
	struct Surface;
	struct Ball;
	struct Stl_mesh;
	struct UndefinedDriver;

	template<typename T>
		constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::UNDEFINED;}
	template<> constexpr DRIVER_TYPE get_type<exaDEM::Cylinder> () { return DRIVER_TYPE::CYLINDER; }
	template<> constexpr DRIVER_TYPE get_type<exaDEM::Surface> () { return DRIVER_TYPE::SURFACE;	}
	template<> constexpr DRIVER_TYPE get_type<exaDEM::Ball> () { return DRIVER_TYPE::BALL;	}
	template<> constexpr DRIVER_TYPE get_type<exaDEM::Stl_mesh> () { return DRIVER_TYPE::STL_MESH;	}
	template<> constexpr DRIVER_TYPE get_type<exaDEM::UndefinedDriver> () { return DRIVER_TYPE::UNDEFINED;	}

	struct Driver
	{
		constexpr DRIVER_TYPE get_type();
		virtual void print();
		virtual bool filter( const double, const Vec3d&);
		virtual std::tuple<bool, double, Vec3d, Vec3d> dectector( const double, const Vec3d&);
	};
}
