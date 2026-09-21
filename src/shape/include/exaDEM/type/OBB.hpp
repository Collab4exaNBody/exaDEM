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
#include <cmath>

#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/math/quaternion_operators.h>

/// @ingroup Bounding_Volumes
/// @brief Oriented Bounding Box
class OBB {
 public:
  exanb::Vec3d center;  //< Center
  exanb::Vec3d e1;      //< 3 directions (normalized vectors)
  exanb::Vec3d e2;      //< 3 directions (normalized vectors)
  exanb::Vec3d e3;      //< 3 directions (normalized vectors)
  exanb::Vec3d extent;  //< 3 extents (in the the 3 directions)

  // Constructors
  ONIKA_HOST_DEVICE_FUNC
  OBB() : center{0.0, 0.0, 0.0}, e1{1.0, 0.0, 0.0}, e2{0.0, 1.0, 0.0}, e3{0.0, 0.0, 1.0}, extent{0.0, 0.0, 0.0} {}

  ONIKA_HOST_DEVICE_FUNC
  void enlarge(double more) {
    extent.x += more;
    extent.y += more;
    extent.z += more;
  }

  void rescale(const double scale) {
    e1 = scale * e1;
    e2 = scale * e2;
    e3 = scale * e3;
    extent = scale * extent;
    center = scale * center;
  }

  ONIKA_HOST_DEVICE_FUNC
  void translate(const exanb::Vec3d& v) { center += v; }

  ONIKA_HOST_DEVICE_FUNC
  void rotate(const exanb::Quaternion& Q) {
    e1 = Q * e1;
    e2 = Q * e2;
    e3 = Q * e3;
    center = Q * center;
  }

  // see page 101 of the book 'Real-Time Collision Detection' (Christer Ericson)
  ONIKA_HOST_DEVICE_FUNC
  bool intersect(const OBB& obb, double tol = FLT_EPSILON) const {
    double ra, rb;
    exanb::Mat3d R, AbsR;

    // Compute first terms of rotation matrix expressing obb frame in this OBB coordinate frame
    // (other terms will be computed later)
    R.m11 = exanb::dot(e1, obb.e1);
    R.m12 = exanb::dot(e1, obb.e2);
    R.m13 = exanb::dot(e1, obb.e3);

    // Same thing for absolut values. Add in an epsilon term to
    // counteract arithmetic errors when two edges are parallel and
    // their cross product is (near) null
    AbsR.m11 = fabs(R.m11) + tol;
    AbsR.m12 = fabs(R.m12) + tol;
    AbsR.m13 = fabs(R.m13) + tol;

    // Compute translation vector t into this OBB coordinate frame
    exanb::Vec3d tt = center - obb.center;
    exanb::Vec3d t{exanb::dot(tt, e1), exanb::dot(tt, e2), exanb::dot(tt, e3)};

    // Test axes eA0
    ra = extent.x;
    rb = obb.extent.x * AbsR.m11 + obb.extent.y * AbsR.m12 + obb.extent.z * AbsR.m13;
    if (fabs(t.x) > ra + rb) {
      return false;
    }

    R.m21 = exanb::dot(e2, obb.e1);
    AbsR.m21 = fabs(R.m21) + tol;
    R.m22 = exanb::dot(e2, obb.e2);
    AbsR.m22 = fabs(R.m22) + tol;
    R.m23 = exanb::dot(e2, obb.e3);
    AbsR.m23 = fabs(R.m23) + tol;

    // Test axes eA1
    ra = extent.y;
    rb = obb.extent.x * AbsR.m21 + obb.extent.y * AbsR.m22 + obb.extent.z * AbsR.m23;
    if (fabs(t.y) > ra + rb) {
      return false;
    }

    R.m31 = exanb::dot(e3, obb.e1);
    AbsR.m31 = fabs(R.m31) + tol;
    R.m32 = exanb::dot(e3, obb.e2);
    AbsR.m32 = fabs(R.m32) + tol;
    R.m33 = exanb::dot(e3, obb.e3);
    AbsR.m33 = fabs(R.m33) + tol;

    // Test axes eA2
    ra = extent.z;
    rb = obb.extent.x * AbsR.m31 + obb.extent.y * AbsR.m32 + obb.extent.z * AbsR.m33;
    if (fabs(t.z) > ra + rb) {
      return false;
    }

    // Test axes L = eB0, L = eB1, L = eB2
    ra = extent.x * AbsR.m11 + extent.y * AbsR.m21 + extent.z * AbsR.m31;
    rb = obb.extent.x;
    if (fabs(t.x * R.m11 + t.y * R.m21 + t.z * R.m31) > ra + rb) {
      return false;
    }

    ra = extent.x * AbsR.m12 + extent.y * AbsR.m22 + extent.z * AbsR.m32;
    rb = obb.extent.y;
    if (fabs(t.x * R.m12 + t.y * R.m22 + t.z * R.m32) > ra + rb) {
      return false;
    }

    ra = extent.x * AbsR.m13 + extent.y * AbsR.m23 + extent.z * AbsR.m33;
    rb = obb.extent.z;
    if (fabs(t.x * R.m13 + t.y * R.m23 + t.z * R.m33) > ra + rb) {
      return false;
    }

    // Test axis L = eA0 x eB0
    ra = extent.y * AbsR.m31 + extent.z * AbsR.m21;
    rb = obb.extent.y * AbsR.m13 + obb.extent.z * AbsR.m12;
    if (fabs(t.z * R.m21 - t.y * R.m31) > ra + rb) {
      return false;
    }
    // Test axis L = eA0 x eB1
    ra = extent.y * AbsR.m32 + extent.z * AbsR.m22;
    rb = obb.extent.x * AbsR.m13 + obb.extent.z * AbsR.m11;
    if (fabs(t.z * R.m22 - t.y * R.m32) > ra + rb) {
      return false;
    }
    // Test axis L = eA0 x eB2
    ra = extent.y * AbsR.m33 + extent.z * AbsR.m23;
    rb = obb.extent.x * AbsR.m12 + obb.extent.y * AbsR.m11;
    if (fabs(t.z * R.m23 - t.y * R.m33) > ra + rb) {
      return false;
    }
    // Test axis L = eA1 x eB0
    ra = extent.x * AbsR.m31 + extent.z * AbsR.m11;
    rb = obb.extent.y * AbsR.m23 + obb.extent.z * AbsR.m22;
    if (fabs(t.x * R.m31 - t.z * R.m11) > ra + rb) {
      return false;
    }
    // Test axis L = eA1 x eB1
    ra = extent.x * AbsR.m32 + extent.z * AbsR.m12;
    rb = obb.extent.x * AbsR.m23 + obb.extent.z * AbsR.m21;
    if (fabs(t.x * R.m32 - t.z * R.m12) > ra + rb) {
      return false;
    }
    // Test axis L = eA1 x eB2
    ra = extent.x * AbsR.m33 + extent.z * AbsR.m13;
    rb = obb.extent.x * AbsR.m22 + obb.extent.y * AbsR.m21;
    if (fabs(t.x * R.m33 - t.z * R.m13) > ra + rb) {
      return false;
    }
    // Test axis L = eA2 x eB0
    ra = extent.x * AbsR.m21 + extent.y * AbsR.m11;
    rb = obb.extent.y * AbsR.m33 + obb.extent.z * AbsR.m32;
    if (fabs(t.y * R.m11 - t.x * R.m21) > ra + rb) {
      return false;
    }
    // Test axis L = eA2 x eB1
    ra = extent.x * AbsR.m22 + extent.y * AbsR.m12;
    rb = obb.extent.x * AbsR.m33 + obb.extent.z * AbsR.m31;
    if (fabs(t.y * R.m12 - t.x * R.m22) > ra + rb) {
      return false;
    }
    // Test axis L = eA2 x eB2
    ra = extent.x * AbsR.m23 + extent.y * AbsR.m13;
    rb = obb.extent.x * AbsR.m32 + obb.extent.y * AbsR.m31;
    if (fabs(t.y * R.m13 - t.x * R.m23) > ra + rb) {
      return false;
    }

    // Since no separating axis is found, the OBBs must be intersecting
    return true;
  }

  // To know wether a point inside the OBB
  ONIKA_HOST_DEVICE_FUNC
  bool intersect(const exanb::Vec3d& a) const {
    exanb::Vec3d v = a - center;
    return !((fabs(exanb::dot(v, e1)) > extent.x) || (fabs(exanb::dot(v, e2)) > extent.y) ||
             (fabs(exanb::dot(v, e3)) > extent.z));
  }

  // Input/Output
  friend std::ostream& operator<<(std::ostream& pStr, const OBB& pOBB) {
    return (pStr << pOBB.center << ' ' << pOBB.e1 << ' ' << pOBB.e2 << ' ' << pOBB.e3 << ' ' << pOBB.extent);
  }
};

inline OBB conv_to_obb(const exanb::AABB& aabb) {
  OBB res;  // e1 e2 e3 are set corretly
  auto& _min = aabb.bmin;
  auto& _max = aabb.bmax;
  res.center.x = (_max.x + _min.x) / 2;
  res.center.y = (_max.y + _min.y) / 2;
  res.center.z = (_max.z + _min.z) / 2;
  res.extent.x = (_max.x - _min.x) / 2;
  res.extent.y = (_max.y - _min.y) / 2;
  res.extent.z = (_max.z - _min.z) / 2;
  return res;
}

// optimize it later
inline exanb::AABB conv_to_aabb(const OBB& obb) {
  auto my_abs = [](const exanb::Vec3d& in) -> exanb::Vec3d {
    return exanb::Vec3d{std::abs(in.x), std::abs(in.y), std::abs(in.z)};
  };

  exanb::Vec3d abs =
      my_abs(obb.e1) * obb.extent.x + my_abs(obb.e2) * obb.extent.y + my_abs(obb.e3) * obb.extent.z;
  exanb::AABB res = {exanb::Vec3d{obb.center.x - abs.x, obb.center.y - abs.y, obb.center.z - abs.z},
                     exanb::Vec3d{obb.center.x + abs.x, obb.center.y + abs.y, obb.center.z + abs.z}};

  return res;
}
