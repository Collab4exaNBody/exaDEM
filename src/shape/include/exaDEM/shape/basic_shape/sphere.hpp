#pragma once

#include <cmath>
#include <exaDEM/shape.hpp>

namespace exaDEM {
namespace basic_shape {
/**
 * @brief Return a shape of a cuboïde.
 * @param name Shape name.
 * @param minkowski Minkowski radius
 */
shape create_sphere(std::string name, double minkowski) {
  using exanb::Quaternion;
  using exanb::Vec3d;
  // rename
  const double r = minkowski;
  const double pi = M_PI;
  shape shp;
  shp.name_ = name;
  shp.add_radius(r);

  // ---------- Volume ----------
  shp.volume_ = 4 / 3 * pi * r * r * r;

  // ---------- I / m ----------
  shp.inertia_on_mass_ = 0.5 * r * r * Vec3d{1, 1, 1};

  shp.add_vertex({0., 0., 0.});  // v0

  shp.obb_ = build_obb_from_shape(shp);
  shp.pre_compute_obb_edges(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
  shp.pre_compute_obb_faces(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
  return shp;
}
}  // namespace basic_shape
}  // namespace exaDEM
