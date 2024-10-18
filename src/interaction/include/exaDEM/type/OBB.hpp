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

#pragma once
/// @file
/// @brief Oriented Bounding Box
/// @author Vincent Richefeu <Vincent.Richefeu@3sr-grenoble.fr>,
/// Lab 3SR, Grenoble University

#include <cfloat>

#include "vec3.hpp"
#include "quat.hpp"
#include "mat9.hpp"
#include <exanb/core/basic_types.h>

/// @ingroup Bounding_Volumes
/// @brief Oriented Bounding Box
class OBB
{
public:
  vec3r center; //< Center
  vec3r e1;     //< 3 directions (normalized vectors)
  vec3r e2;     //< 3 directions (normalized vectors)
  vec3r e3;     //< 3 directions (normalized vectors)
  vec3r extent; //< 3 extents (in the the 3 directions)

  // Constructors
  ONIKA_HOST_DEVICE_FUNC
  OBB() : center(), extent(0.0, 0.0, 0.0)
  {
    e1.set(1.0, 0.0, 0.0);
    e2.set(0.0, 1.0, 0.0);
    e3.set(0.0, 0.0, 1.0);
  }

  ONIKA_HOST_DEVICE_FUNC
  OBB(const OBB &obb) : center(obb.center), extent(obb.extent)
  {
    e1 = obb.e1;
    e2 = obb.e2;
    e3 = obb.e3;
  }

  ONIKA_HOST_DEVICE_FUNC
  OBB &operator=(const OBB &obb)
  {
    center = obb.center;
    e1 = obb.e1;
    e2 = obb.e2;
    e3 = obb.e3;
    extent = obb.extent;
    return (*this);
  }

  ONIKA_HOST_DEVICE_FUNC
  void enlarge(double more)
  {
    extent.x += more;
    extent.y += more;
    extent.z += more;
  }

  ONIKA_HOST_DEVICE_FUNC
  void translate(const vec3r &v) { center += v; }

  ONIKA_HOST_DEVICE_FUNC
  void rotate(const quat &Q)
  {
    e1 = Q * e1;
    e2 = Q * e2;
    e3 = Q * e3;
    center = Q * center;
  }

  // see page 101 of the book 'Real-Time Collision Detection' (Christer Ericson)
  ONIKA_HOST_DEVICE_FUNC
  bool intersect(const OBB &obb, double tol = FLT_EPSILON) const
  {
    double ra, rb;
    mat9r R, AbsR;

    // Compute first terms of rotation matrix expressing obb frame in this OBB coordinate frame
    // (other terms will be computed later)
    R.xx = e1 * obb.e1;
    R.xy = e1 * obb.e2;
    R.xz = e1 * obb.e3;

    // Same thing for absolut values. Add in an epsilon term to
    // counteract arithmetic errors when two edges are parallel and
    // their cross product is (near) null
    AbsR.xx = fabs(R.xx) + tol;
    AbsR.xy = fabs(R.xy) + tol;
    AbsR.xz = fabs(R.xz) + tol;

    // Compute translation vector t into this OBB coordinate frame
    vec3r tt = center - obb.center;
    vec3r t(tt * e1, tt * e2, tt * e3);

    // Test axes eA0
    ra = extent.x;
    rb = obb.extent.x * AbsR.xx + obb.extent.y * AbsR.xy + obb.extent.z * AbsR.xz;
    if (fabs(t.x) > ra + rb)
      return false;

    R.yx = e2 * obb.e1;
    AbsR.yx = fabs(R.yx) + tol;
    R.yy = e2 * obb.e2;
    AbsR.yy = fabs(R.yy) + tol;
    R.yz = e2 * obb.e3;
    AbsR.yz = fabs(R.yz) + tol;

    // Test axes eA1
    ra = extent.y;
    rb = obb.extent.x * AbsR.yx + obb.extent.y * AbsR.yy + obb.extent.z * AbsR.yz;
    if (fabs(t.y) > ra + rb)
      return false;

    R.zx = e3 * obb.e1;
    AbsR.zx = fabs(R.zx) + tol;
    R.zy = e3 * obb.e2;
    AbsR.zy = fabs(R.zy) + tol;
    R.zz = e3 * obb.e3;
    AbsR.zz = fabs(R.zz) + tol;

    // Test axes eA2
    ra = extent.z;
    rb = obb.extent.x * AbsR.zx + obb.extent.y * AbsR.zy + obb.extent.z * AbsR.zz;
    if (fabs(t.z) > ra + rb)
      return false;

    // Test axes L = eB0, L = eB1, L = eB2
    ra = extent.x * AbsR.xx + extent.y * AbsR.yx + extent.z * AbsR.zx;
    rb = obb.extent.x;
    if (fabs(t.x * R.xx + t.y * R.yx + t.z * R.zx) > ra + rb)
      return false;

    ra = extent.x * AbsR.xy + extent.y * AbsR.yy + extent.z * AbsR.zy;
    rb = obb.extent.y;
    if (fabs(t.x * R.xy + t.y * R.yy + t.z * R.zy) > ra + rb)
      return false;

    ra = extent.x * AbsR.xz + extent.y * AbsR.yz + extent.z * AbsR.zz;
    rb = obb.extent.z;
    if (fabs(t.x * R.xz + t.y * R.yz + t.z * R.zz) > ra + rb)
      return false;

    // Test axis L = eA0 x eB0
    ra = extent.y * AbsR.zx + extent.z * AbsR.yx;
    rb = obb.extent.y * AbsR.xz + obb.extent.z * AbsR.xy;
    if (fabs(t.z * R.yx - t.y * R.zx) > ra + rb)
      return false;
    // Test axis L = eA0 x eB1
    ra = extent.y * AbsR.zy + extent.z * AbsR.yy;
    rb = obb.extent.x * AbsR.xz + obb.extent.z * AbsR.xx;
    if (fabs(t.z * R.yy - t.y * R.zy) > ra + rb)
      return false;
    // Test axis L = eA0 x eB2
    ra = extent.y * AbsR.zz + extent.z * AbsR.yz;
    rb = obb.extent.x * AbsR.xy + obb.extent.y * AbsR.xx;
    if (fabs(t.z * R.yz - t.y * R.zz) > ra + rb)
      return false;
    // Test axis L = eA1 x eB0
    ra = extent.x * AbsR.zx + extent.z * AbsR.xx;
    rb = obb.extent.y * AbsR.yz + obb.extent.z * AbsR.yy;
    if (fabs(t.x * R.zx - t.z * R.xx) > ra + rb)
      return false;
    // Test axis L = eA1 x eB1
    ra = extent.x * AbsR.zy + extent.z * AbsR.xy;
    rb = obb.extent.x * AbsR.yz + obb.extent.z * AbsR.yx;
    if (fabs(t.x * R.zy - t.z * R.xy) > ra + rb)
      return false;
    // Test axis L = eA1 x eB2
    ra = extent.x * AbsR.zz + extent.z * AbsR.xz;
    rb = obb.extent.x * AbsR.yy + obb.extent.y * AbsR.yx;
    if (fabs(t.x * R.zz - t.z * R.xz) > ra + rb)
      return false;
    // Test axis L = eA2 x eB0
    ra = extent.x * AbsR.yx + extent.y * AbsR.xx;
    rb = obb.extent.y * AbsR.zz + obb.extent.z * AbsR.zy;
    if (fabs(t.y * R.xx - t.x * R.yx) > ra + rb)
      return false;
    // Test axis L = eA2 x eB1
    ra = extent.x * AbsR.yy + extent.y * AbsR.xy;
    rb = obb.extent.x * AbsR.zz + obb.extent.z * AbsR.zx;
    if (fabs(t.y * R.xy - t.x * R.yy) > ra + rb)
      return false;
    // Test axis L = eA2 x eB2
    ra = extent.x * AbsR.yz + extent.y * AbsR.xz;
    rb = obb.extent.x * AbsR.zy + obb.extent.y * AbsR.zx;
    if (fabs(t.y * R.xz - t.x * R.yz) > ra + rb)
      return false;

    // Since no separating axis is found, the OBBs must be intersecting
    return true;
  }

  // To know wether a point inside the OBB
  ONIKA_HOST_DEVICE_FUNC
  bool intersect(const vec3r &a) const
  {
    vec3r v = a - center;
    return !((fabs(v * e1) > extent.x) || (fabs(v * e2) > extent.y) || (fabs(v * e3) > extent.z));
  }

  // Input/Output
  friend std::ostream &operator<<(std::ostream &pStr, const OBB &pOBB) { return (pStr << pOBB.center << ' ' << pOBB.e1 << ' ' << pOBB.e2 << ' ' << pOBB.e3 << ' ' << pOBB.extent); }

  friend std::istream &operator>>(std::istream &pStr, OBB &pOBB) { return (pStr >> pOBB.center >> pOBB.e1 >> pOBB.e2 >> pOBB.e3 >> pOBB.extent); }
};

inline OBB conv_to_obb(const exanb::AABB &aabb)
{
  OBB res; // e1 e2 e3 are set corretly
  auto &_min = aabb.bmin;
  auto &_max = aabb.bmax;
  res.center.x = (_max.x + _min.x) / 2;
  res.center.y = (_max.y + _min.y) / 2;
  res.center.z = (_max.z + _min.z) / 2;
  res.extent.x = (_max.x - _min.x) / 2;
  res.extent.y = (_max.y - _min.y) / 2;
  res.extent.z = (_max.z - _min.z) / 2;
  return res;
}

// optimize it later
inline exanb::AABB conv_to_aabb(const OBB &obb)
{
  auto my_abs = [](const vec3r &in) -> vec3r
  {
    vec3r res = {std::abs(in.x), std::abs(in.y), std::abs(in.z)};
    return res;
  };

  vec3r abs = my_abs(obb.e1) * obb.extent.x + my_abs(obb.e2) * obb.extent.y + my_abs(obb.e3) * obb.extent.z;
  exanb::AABB res = {exanb::Vec3d{obb.center.x - abs.x, obb.center.y - abs.y, obb.center.z - abs.z}, exanb::Vec3d{obb.center.x + abs.x, obb.center.y + abs.y, obb.center.z + abs.z}};

  return res;
}
