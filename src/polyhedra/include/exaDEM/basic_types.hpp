#pragma once
#include <exanb/core/quaternion_operators.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exaDEM/type/OBB.hpp>

namespace exaDEM
{
	using namespace exanb;

	inline vec3r conv_to_vec3r (const exanb::Vec3d& v)
	{
		return vec3r {v.x,v.y,v.z};
	}

	inline Vec3d conv_to_Vec3d (vec3r& v)
	{
		return Vec3d {v[0],v[1],v[2]};
	}

	inline quat conv_to_quat( const exanb::Quaternion& Q)
	{
		return quat{vec3r{Q.x, Q.y, Q.z}, Q.w};
	}
}
