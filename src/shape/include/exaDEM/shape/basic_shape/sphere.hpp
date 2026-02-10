#pragma once

#include <exaDEM/shape.hpp>

#include <cmath>

namespace exaDEM {
namespace basic_shape {
/**
 * @brief Return a shape of a cuboïde. 
 * @param name Shape name.
 * @param minskowski Minskowski radius
 */
shape create_sphere(std::string name, double minskowski) {
  // rename
  const double r = minskowski;
  const double pi = M_PI;
  shape shp;
  shp.m_name = name;
  shp.m_radius = r;

  // ---------- Volume ----------
  shp.m_volume = 4/3 * pi * r * r * r;

  // ---------- I / m ----------
  shp.m_inertia_on_mass = 0.5 * r * r * Vec3d{1,1,1};

  shp.add_vertex({0., 0., 0.});  // v0

  shp.obb = build_obb_from_shape(shp);
  shp.pre_compute_obb_edges(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
  shp.pre_compute_obb_faces(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
  return shp;
} 
}  // namespace shape
}  // namespace exaDEM
