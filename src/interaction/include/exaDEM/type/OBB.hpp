// Copyright (C) OBB <vincent.richefeu@3sr-grenoble.fr>
// 
// This file is part of mbox.
// 
// OBB can not be copied and/or distributed without the express
// permission of the authors.
// It is coded for academic purposes.
//
// Note 
// Without a license, the code is copyrighted by default. 
// People can read the code, but they have no legal right to use it. 
// To use the code, you must contact the author directly and ask permission.

#ifndef OBB_HPP_D3958314
#define OBB_HPP_D3958314
/// @file
/// @brief Oriented Bounding Box
/// @author Vincent Richefeu <Vincent.Richefeu@3sr-grenoble.fr>,
/// Lab 3SR, Grenoble University

#include <cfloat>

#include "vec3.hpp"
#include "quat.hpp"
#include "mat9.hpp"

/// @ingroup Bounding_Volumes
/// @brief Oriented Bounding Box
class OBB
{
public:
	vec3r center;  //< Center
	vec3r e[3];    //< 3 directions (normalized vectors)
	vec3r extent;  //< 3 extents (in the the 3 directions)
	
	// Constructors
	OBB(): center(), extent(0.0, 0.0, 0.0) {
		e[0].set(1.0, 0.0, 0.0);
		e[1].set(0.0, 1.0, 0.0);
		e[2].set(0.0, 0.0, 1.0);
	}

	OBB(const OBB &obb): center(obb.center), extent(obb.extent) {
		e[0] = obb.e[0];
		e[1] = obb.e[1];
		e[2] = obb.e[2];
	}
  
	OBB & operator = (const OBB &obb) {
		center = obb.center;
		e[0] = obb.e[0];
		e[1] = obb.e[1];
		e[2] = obb.e[2];
    extent = obb.extent;
		return (*this);
	}

	void enlarge(double more) {
		extent.x += more;
		extent.y += more;
		extent.z += more;
	}

	void translate(const vec3r & v) {
		center += v;
	}

	void rotate(const quat & Q) {
		e[0] = Q * e[0];
		e[1] = Q * e[1];
		e[2] = Q * e[2];
		center = Q * center;
	}

	// see page 101 of the book 'Real-Time Collision Detection' (Christer Ericson)
	bool intersect(const OBB& obb, double tol = FLT_EPSILON) const {
		double  ra, rb;
		mat9r R, AbsR;

		// Compute first terms of rotation matrix expressing obb frame in this OBB coordinate frame
		// (other terms will be computed later)
		R.xx = e[0] * obb.e[0];
		R.xy = e[0] * obb.e[1];
		R.xz = e[0] * obb.e[2];

		// Same thing for absolut values. Add in an epsilon term to
		// counteract arithmetic errors when two edges are parallel and
		// their cross product is (near) null
		AbsR.xx = fabs(R.xx) + tol;
		AbsR.xy = fabs(R.xy) + tol;
		AbsR.xz = fabs(R.xz) + tol;

		// Compute translation vector t into this OBB coordinate frame
		vec3r tt = center - obb.center;
		vec3r t(tt * e[0], tt * e[1], tt * e[2]);

		// Test axes eA0
		ra = extent.x;
		rb = obb.extent.x * AbsR.xx + obb.extent.y * AbsR.xy + obb.extent.z * AbsR.xz;
		if (fabs(t.x) > ra + rb) return false;

		R.yx = e[1] * obb.e[0];
		AbsR.yx = fabs(R.yx) + tol;
		R.yy = e[1] * obb.e[1];
		AbsR.yy = fabs(R.yy) + tol;
		R.yz = e[1] * obb.e[2];
		AbsR.yz = fabs(R.yz) + tol;

		// Test axes eA1
		ra = extent.y;
		rb = obb.extent.x * AbsR.yx + obb.extent.y * AbsR.yy + obb.extent.z * AbsR.yz;
		if (fabs(t.y) > ra + rb) return false;

		R.zx = e[2] * obb.e[0];
		AbsR.zx = fabs(R.zx) + tol;
		R.zy = e[2] * obb.e[1];
		AbsR.zy = fabs(R.zy) + tol;
		R.zz = e[2] * obb.e[2];
		AbsR.zz = fabs(R.zz) + tol;

		// Test axes eA2
		ra = extent.z;
		rb = obb.extent.x * AbsR.zx + obb.extent.y * AbsR.zy + obb.extent.z * AbsR.zz;
		if (fabs(t.z) > ra + rb) return false;

		// Test axes L = eB0, L = eB1, L = eB2
		ra = extent.x * AbsR.xx + extent.y * AbsR.yx + extent.z * AbsR.zx;
		rb = obb.extent.x;
		if (fabs(t.x * R.xx + t.y * R.yx + t.z * R.zx) > ra + rb) return false;

		ra = extent.x * AbsR.xy + extent.y * AbsR.yy + extent.z * AbsR.zy;
		rb = obb.extent.y;
		if (fabs(t.x * R.xy + t.y * R.yy + t.z * R.zy) > ra + rb) return false;

		ra = extent.x * AbsR.xz + extent.y * AbsR.yz + extent.z * AbsR.zz;
		rb = obb.extent.z;
		if (fabs(t.x * R.xz + t.y * R.yz + t.z * R.zz) > ra + rb) return false;

		// Test axis L = eA0 x eB0
		ra = extent.y * AbsR.zx + extent.z * AbsR.yx;
		rb = obb.extent.y * AbsR.xz + obb.extent.z * AbsR.xy;
		if (fabs(t.z * R.yx - t.y * R.zx) > ra + rb) return false;
		// Test axis L = eA0 x eB1
		ra = extent.y * AbsR.zy + extent.z * AbsR.yy;
		rb = obb.extent.x * AbsR.xz + obb.extent.z * AbsR.xx;
		if (fabs(t.z * R.yy - t.y * R.zy) > ra + rb) return false;
		// Test axis L = eA0 x eB2
		ra = extent.y * AbsR.zz + extent.z * AbsR.yz;
		rb = obb.extent.x * AbsR.xy + obb.extent.y * AbsR.xx;
		if (fabs(t.z * R.yz - t.y * R.zz) > ra + rb) return false;
		// Test axis L = eA1 x eB0
		ra = extent.x * AbsR.zx + extent.z * AbsR.xx;
		rb = obb.extent.y * AbsR.yz + obb.extent.z * AbsR.yy;
		if (fabs(t.x * R.zx - t.z * R.xx) > ra + rb) return false;
		// Test axis L = eA1 x eB1
		ra = extent.x * AbsR.zy + extent.z * AbsR.xy;
		rb = obb.extent.x * AbsR.yz + obb.extent.z * AbsR.yx;
		if (fabs(t.x * R.zy - t.z * R.xy) > ra + rb) return false;
		// Test axis L = eA1 x eB2
		ra = extent.x * AbsR.zz + extent.z * AbsR.xz;
		rb = obb.extent.x * AbsR.yy + obb.extent.y * AbsR.yx;
		if (fabs(t.x * R.zz - t.z * R.xz) > ra + rb) return false;
		// Test axis L = eA2 x eB0
		ra = extent.x * AbsR.yx + extent.y * AbsR.xx;
		rb = obb.extent.y * AbsR.zz + obb.extent.z * AbsR.zy;
		if (fabs(t.y * R.xx - t.x * R.yx) > ra + rb) return false;
		// Test axis L = eA2 x eB1
		ra = extent.x * AbsR.yy + extent.y * AbsR.xy;
		rb = obb.extent.x * AbsR.zz + obb.extent.z * AbsR.zx;
		if (fabs(t.y * R.xy - t.x * R.yy) > ra + rb) return false;
		// Test axis L = eA2 x eB2
		ra = extent.x * AbsR.yz + extent.y * AbsR.xz;
		rb = obb.extent.x * AbsR.zy + obb.extent.y * AbsR.zx;
		if (fabs(t.y * R.xz - t.x * R.yz) > ra + rb) return false;

		// Since no separating axis is found, the OBBs must be intersecting
		return true;
	}


	// see page 101 of the book 'Real-Time Collision Detection' (Christer Ericson)
	bool intersectPeriodicity(const OBB& obb, const vec3r& periodicity, double tol = FLT_EPSILON) const {
		double  ra, rb;
		mat9r R, AbsR;

		// Compute first terms of rotation matrix expressing obb frame in this OBB coordinate frame
		// (other terms will be computed later)
		R.xx = e[0] * obb.e[0];
		R.xy = e[0] * obb.e[1];
		R.xz = e[0] * obb.e[2];

		// Same thing for absolut values. Add in an epsilon term to
		// counteract arithmetic errors when two edges are parallel and
		// their cross product is (near) null
		AbsR.xx = fabs(R.xx) + tol;
		AbsR.xy = fabs(R.xy) + tol;
		AbsR.xz = fabs(R.xz) + tol;

		// Compute translation vector t into this OBB coordinate frame
		vec3r tt = center - obb.center;
		tt.round(periodicity);

		vec3r t(tt * e[0], tt * e[1], tt * e[2]);

		// Test axes eA0
		ra = extent.x;
		rb = obb.extent.x * AbsR.xx + obb.extent.y * AbsR.xy + obb.extent.z * AbsR.xz;
		if (fabs(t.x) > ra + rb) return false;

		R.yx = e[1] * obb.e[0];
		AbsR.yx = fabs(R.yx) + tol;
		R.yy = e[1] * obb.e[1];
		AbsR.yy = fabs(R.yy) + tol;
		R.yz = e[1] * obb.e[2];
		AbsR.yz = fabs(R.yz) + tol;

		// Test axes eA1
		ra = extent.y;
		rb = obb.extent.x * AbsR.yx + obb.extent.y * AbsR.yy + obb.extent.z * AbsR.yz;
		if (fabs(t.y) > ra + rb) return false;

		R.zx = e[2] * obb.e[0];
		AbsR.zx = fabs(R.zx) + tol;
		R.zy = e[2] * obb.e[1];
		AbsR.zy = fabs(R.zy) + tol;
		R.zz = e[2] * obb.e[2];
		AbsR.zz = fabs(R.zz) + tol;

		// Test axes eA2
		ra = extent.z;
		rb = obb.extent.x * AbsR.zx + obb.extent.y * AbsR.zy + obb.extent.z * AbsR.zz;
		if (fabs(t.z) > ra + rb) return false;

		// Test axes L = eB0, L = eB1, L = eB2
		ra = extent.x * AbsR.xx + extent.y * AbsR.yx + extent.z * AbsR.zx;
		rb = obb.extent.x;
		if (fabs(t.x * R.xx + t.y * R.yx + t.z * R.zx) > ra + rb) return false;

		ra = extent.x * AbsR.xy + extent.y * AbsR.yy + extent.z * AbsR.zy;
		rb = obb.extent.y;
		if (fabs(t.x * R.xy + t.y * R.yy + t.z * R.zy) > ra + rb) return false;

		ra = extent.x * AbsR.xz + extent.y * AbsR.yz + extent.z * AbsR.zz;
		rb = obb.extent.z;
		if (fabs(t.x * R.xz + t.y * R.yz + t.z * R.zz) > ra + rb) return false;

		// Test axis L = eA0 x eB0
		ra = extent.y * AbsR.zx + extent.z * AbsR.yx;
		rb = obb.extent.y * AbsR.xz + obb.extent.z * AbsR.xy;
		if (fabs(t.z * R.yx - t.y * R.zx) > ra + rb) return false;
		// Test axis L = eA0 x eB1
		ra = extent.y * AbsR.zy + extent.z * AbsR.yy;
		rb = obb.extent.x * AbsR.xz + obb.extent.z * AbsR.xx;
		if (fabs(t.z * R.yy - t.y * R.zy) > ra + rb) return false;
		// Test axis L = eA0 x eB2
		ra = extent.y * AbsR.zz + extent.z * AbsR.yz;
		rb = obb.extent.x * AbsR.xy + obb.extent.y * AbsR.xx;
		if (fabs(t.z * R.yz - t.y * R.zz) > ra + rb) return false;
		// Test axis L = eA1 x eB0
		ra = extent.x * AbsR.zx + extent.z * AbsR.xx;
		rb = obb.extent.y * AbsR.yz + obb.extent.z * AbsR.yy;
		if (fabs(t.x * R.zx - t.z * R.xx) > ra + rb) return false;
		// Test axis L = eA1 x eB1
		ra = extent.x * AbsR.zy + extent.z * AbsR.xy;
		rb = obb.extent.x * AbsR.yz + obb.extent.z * AbsR.yx;
		if (fabs(t.x * R.zy - t.z * R.xy) > ra + rb) return false;
		// Test axis L = eA1 x eB2
		ra = extent.x * AbsR.zz + extent.z * AbsR.xz;
		rb = obb.extent.x * AbsR.yy + obb.extent.y * AbsR.yx;
		if (fabs(t.x * R.zz - t.z * R.xz) > ra + rb) return false;
		// Test axis L = eA2 x eB0
		ra = extent.x * AbsR.yx + extent.y * AbsR.xx;
		rb = obb.extent.y * AbsR.zz + obb.extent.z * AbsR.zy;
		if (fabs(t.y * R.xx - t.x * R.yx) > ra + rb) return false;
		// Test axis L = eA2 x eB1
		ra = extent.x * AbsR.yy + extent.y * AbsR.xy;
		rb = obb.extent.x * AbsR.zz + obb.extent.z * AbsR.zx;
		if (fabs(t.y * R.xy - t.x * R.yy) > ra + rb) return false;
		// Test axis L = eA2 x eB2
		ra = extent.x * AbsR.yz + extent.y * AbsR.xz;
		rb = obb.extent.x * AbsR.zy + obb.extent.y * AbsR.zx;
		if (fabs(t.y * R.xz - t.x * R.yz) > ra + rb) return false;

		// Since no separating axis is found, the OBBs must be intersecting
		return true;
	}

	// To know wether a point inside the OBB
	bool intersect(const vec3r& a) const {
		vec3r v = a - center;
		return !( (fabs(v * e[0]) > extent.x) || (fabs(v * e[1]) > extent.y) || (fabs(v * e[2]) > extent.z) );
	}

	// Input/Output
	friend std::ostream& operator<< (std::ostream& pStr, const OBB& pOBB) {
		return (pStr <<  pOBB.center << ' ' << pOBB.e[0] << ' ' << pOBB.e[1] << ' ' << pOBB.e[2] << ' ' << pOBB.extent );
	}

	friend std::istream& operator>> (std::istream& pStr, OBB& pOBB) {
		return (pStr >> pOBB.center >> pOBB.e[0] >> pOBB.e[1] >> pOBB.e[2] >> pOBB.extent);
	}	
};

#endif /* end of include guard: OBB_HPP_D3958314 */
