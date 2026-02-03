#pragma once

#include <exaDEM/shape.hpp>
#include <cmath>

namespace exaDEM {
namespace basic_shape {
/**
 * @brief Return a shape of a cuboïde. 
 * @param name Shape name.
 * @param length Cube length
 * @param minskowski Minskowski radius
 */
shape create_rice(std::string name, double length, double minskowski) {
  // rename
  const double c = length;
  const double r = minskowski;
  const double pi = M_PI;
  shape shp;
  shp.m_name = name;
  shp.m_radius = r;

  // ---------- Volume ----------
  const double v_cyl = pi * r * r * c;
  const double v_sph = (4.0/3.0) * pi * r * r * r;
  shp.m_volume = v_cyl + v_sph;

  // ---------- Normalized Inertia (I/m) ----------
  // Calculated in the local frame where the tube is aligned along the X-axis.
  // We use the decomposition: total_inertia = inertia_cylinder + inertia_sphere
  const double inv_v_tot = 1.0 / shp.m_volume;

  // X-axis: Rolling (sum of axial inertias)
  double Ix_m = (v_cyl * 0.5 * r * r + v_sph * 0.4 * r * r) * inv_v_tot;

  // Y & Z axes: Tumbling (Parallel Axis Theorem for the two hemispheres)
  double Iy_cyl = v_cyl * ( (c * c / 12.0) + (r * r / 4.0) );
  double Iy_sph = v_sph * ( (0.4 * r * r) + (0.25 * c * c) );
  double Iy_m = (Iy_cyl + Iy_sph) * inv_v_tot;

  // Final normalized inertia vector
  shp.m_inertia_on_mass = { Ix_m, Iy_m, Iy_m };

  shp.add_vertex({-0.5, 0, 0});  // v0
  shp.add_vertex({ 0.5, 0, 0});  // v1

  auto func = [c] (Vec3d& v) -> void {
    v *= c;
  };

  shp.for_all_vertices(func);
  shp.add_edge(0, 1);

  shp.obb = build_obb_from_shape(shp);
  shp.pre_compute_obb_edges(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
  shp.pre_compute_obb_faces(Vec3d{0, 0, 0}, Quaternion{1, 0, 0, 0});
  return shp;
} 
}  // namespace basic_shape
}  // namespace exaDEM
