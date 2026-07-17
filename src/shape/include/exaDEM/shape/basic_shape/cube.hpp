#pragma once

#include <cmath>
#include <exaDEM/shape.hpp>

namespace exaDEM {
namespace basic_shape {
/**
 * @brief Return a shape of a cuboïde.
 * @param name Shape name.
 * @param length Cube length
 * @param minkowski Minkowski radius
 */
shape create_cube(std::string name, double length, double minkowski) {
  // rename
  const double c = length;
  const double r = minkowski;
  const double pi = M_PI;
  shape shp;
  shp.name_ = name;
  shp.add_radius(r);

  // ---------- Volume ----------
  shp.volume_ = c * c * c + 6.0 * c * c * r + 3.0 * pi * r * r * c + (4.0 / 3.0) * pi * r * r * r;

  /// Compute Inertia
  // Inertia Cube
  double I = (1.0 / 6.0) * std::pow(c, 5);

  // Intertia Faces (6 prismes c x c x r)
  const double d = 0.5 * (c + r);
  const double I_face = (1.0 / 12.0) * c * c * r * (c * c + r * r);
  const double V_face = c * c * r;

  I += 6.0 * (I_face + V_face * d * d);

  // Interia Edges
  const double V_edges = 3.0 * pi * r * r * c;
  I += V_edges * ((c * c) / 12.0 + (r * r) / 2.0);

  // Interia Vertices
  I += (8.0 / 15.0) * pi * std::pow(r, 5);

  // ---------- I / m ----------
  shp.inertia_on_mass_ = I / shp.get_volume() * exanb::Vec3d{1, 1, 1};

  shp.add_vertex({-0.5, -0.5, -0.5});  // v0
  shp.add_vertex({0.5, -0.5, -0.5});   // v1
  shp.add_vertex({0.5, 0.5, -0.5});    // v2
  shp.add_vertex({-0.5, 0.5, -0.5});   // v3
  shp.add_vertex({-0.5, -0.5, 0.5});   // v4
  shp.add_vertex({0.5, -0.5, 0.5});    // v5
  shp.add_vertex({0.5, 0.5, 0.5});     // v6
  shp.add_vertex({-0.5, 0.5, 0.5});    // v7

  auto func = [c](exanb::Vec3d& v) -> void { v *= c; };
  shp.for_all_vertices(func);

  shp.add_edge(0, 1);  // below
  shp.add_edge(1, 2);  // below
  shp.add_edge(2, 3);  // below
  shp.add_edge(3, 0);  // below
  shp.add_edge(4, 5);  // above
  shp.add_edge(5, 6);  // above
  shp.add_edge(6, 7);  // above
  shp.add_edge(7, 4);  // above
  shp.add_edge(0, 4);
  shp.add_edge(1, 5);
  shp.add_edge(2, 6);
  shp.add_edge(3, 7);

  shp.add_face({3, 2, 1, 0});
  shp.add_face({4, 5, 6, 7});
  shp.add_face({0, 1, 5, 4});
  shp.add_face({2, 3, 7, 6});
  shp.add_face({4, 7, 3, 0});
  shp.add_face({1, 2, 6, 5});
  shp.compute_offset_faces();

  shp.compute_face_areas();
  shp.obb_ = build_obb_from_shape(shp);
  shp.pre_compute_obb_edges(exanb::Vec3d{0, 0, 0}, exanb::Quaternion{1, 0, 0, 0});
  shp.pre_compute_obb_faces(exanb::Vec3d{0, 0, 0}, exanb::Quaternion{1, 0, 0, 0});
  return shp;
}
}  // namespace basic_shape
}  // namespace exaDEM
