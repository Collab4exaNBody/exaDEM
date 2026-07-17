/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#pragma once

#include <onika/math/basic_types.h>
#include <exaDEM/normalize.hpp>

namespace exaDEM {
/**
 * @brief Struct representing a 3D box.
 */
struct Box {
  exanb::Vec3d inf_; /**< The lower corner of the box. */
  exanb::Vec3d sup_; /**< The upper corner of the box. */

  /**
   * @brief Calculate the center of the box.
   * @return The center of the box as a exanb::Vec3d.
   */
  exanb::Vec3d center() {
    exanb::Vec3d res = (sup_ - inf_) / 2;
    return res;
  };
};

/**
 * @brief Struct representing a 3D face.
 */
struct Face {
  std::vector<exanb::Vec3d> vertices_; /**< The vertices_ of the face. */
  exanb::Vec3d normal_;                /**< The normal_ vector of the face. */
  double offset_;               /**< The offset_ of the face. */

  /**
   * @brief Constructor for the Face struct.
   * @param in A vector of exanb::Vec3d representing the vertices_ of the face.
   */
  Face(std::vector<exanb::Vec3d>& in) {
    vertices_ = in;
    auto [_normal, _offset, _exist] = compute_normal_and_offset();
    normal_ = _normal;
    offset_ = _offset;
    if (!_exist) {
      std::cout << " error when filling this Face " << std::endl;
    }
  }

  /**
   * @brief Determines if a sphere and a face potentially intersect and calculates the contact position.
   *
   * This function checks whether a sphere with the given radius and center intersects with a face defined by its
   * vertices_, normal_ vector, and offset_. It returns information about the potential intersection and the contact
   * position.
   *
   * @param rx The x-coordinate of the sphere's center.
   * @param ry The y-coordinate of the sphere's center.
   * @param rz The z-coordinate of the sphere's center.
   * @param rad The radius of the sphere.
   * @return A tuple containing three values:
   *         - `bool` face_contact: Indicates whether there is an intersection with the face.
   *         - `bool` potential_contact: Indicates potential intersection with the face for further testing.
   *         - `Vec3d` contact_position: The contact position if an intersection occurs (otherwise, it is {0,0,0}).
   */
  std::tuple<bool, bool, exanb::Vec3d> contact_face_sphere(const double rx, const double ry, const double rz,
                                                    const double rad) const {
    // printf("CONTACT FACE SPHERE\n");
    const exanb::Vec3d center = {rx, ry, rz};
    const exanb::Vec3d default_contact_point = {0, 0, 0};  // won't be used
    bool potential_contact = false;
    bool face_contact = false;
    exanb::Vec3d contact_position = default_contact_point;

    double p = exanb::dot(center, normal_) - offset_;
    if (std::abs(p) > rad) {
      return std::make_tuple(face_contact, potential_contact, contact_position);
    }

    potential_contact = true;  // This face will be tested versus edges (second pass)

    const int nb_vertices = vertices_.size();
    const exanb::Vec3d& pa = vertices_[0];
    const exanb::Vec3d& pb = vertices_[1];
    const exanb::Vec3d& pc = vertices_[nb_vertices - 1];
    exanb::Vec3d v1 = pb - pa;
    exanb::Vec3d v2 = pc - pa;
    _normalize(v1);
    exanb::Vec3d n = exanb::cross(v1, v2);
    _normalize(n);
    exanb::Vec3d iv = center;  // - pa;
    double dist = exanb::dot(iv, n);
    if (dist < 0.0) {
      dist = -dist;
      n = -n;
    }

    // test if the sphere intersects the surface
    int intersections = 0;

    // from rockable
    exanb::Vec3d P = iv - dist * n;
    v2 = exanb::cross(n, v1);
    double ori1 = exanb::dot(P, v1);
    double ori2 = exanb::dot(P, v2);

    for (int iva = 0; iva < nb_vertices; ++iva) {
      int ivb = iva + 1;
      if (ivb == nb_vertices) ivb = 0;
      const exanb::Vec3d& posNodeA_jv = vertices_[iva];
      const exanb::Vec3d& posNodeB_jv = vertices_[ivb];
      double pa1 = exanb::dot(posNodeA_jv, v1);
      double pb1 = exanb::dot(posNodeB_jv, v1);
      double pa2 = exanb::dot(posNodeA_jv, v2);
      double pb2 = exanb::dot(posNodeB_jv, v2);

      // @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
      // @see http://alienryderflex.com/polygon/
      if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2)) {
        if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1) {
          intersections = 1 - intersections;
        }
      }
    }

    if (intersections == 1) {  // ODD
      contact_position = normal_ * offset_;  // we need dot(conatct_position, normal_)
      face_contact = true;
    }

    return std::make_tuple(face_contact, potential_contact, contact_position);
  }

  /**
   * @brief Determines if a sphere intersects with an edge and calculates the contact position.
   *
   * This function checks whether a sphere with the given radius and center intersects with any edge of a polygon
   * defined by its vertices_. It returns information about the intersection and the contact position if applicable.
   *
   * @param rx The x-coordinate of the sphere's center.
   * @param ry The y-coordinate of the sphere's center.
   * @param rz The z-coordinate of the sphere's center.
   * @param rad The radius of the sphere.
   * @return A tuple containing two values:
   *         - `bool` intersects: Indicates whether there is an intersection with an edge.
   *         - `Vec3d` contact_position: The contact position if an intersection occurs (otherwise, it is {0,0,0}).
   */
  std::tuple<bool, exanb::Vec3d> contact_edge_sphere(const double rx, const double ry, const double rz,
                                              const double rad) const {
    // already tested if  exanb::dot(center,normal_) - offset_ < rad
    // test if the sphere intersects an edge
    const exanb::Vec3d center = {rx, ry, rz};
    const exanb::Vec3d default_contact_point = {0, 0, 0};  // won't be used
    for (size_t i = 0; i < vertices_.size(); ++i) {
      exanb::Vec3d p1 = vertices_[i];
      exanb::Vec3d p2 = vertices_[(i + 1) % vertices_.size()];
      exanb::Vec3d edge = p2 - p1;
      exanb::Vec3d sphereToEdge = center - p1;

      double distanceToEdge = length(exanb::cross(edge, sphereToEdge)) / length(edge);

      if (distanceToEdge <= rad && exanb::dot(sphereToEdge, edge) > 0 && exanb::dot(sphereToEdge - edge, edge) < 0) {
        auto n_edge = edge / exanb::norm(edge);
        exanb::Vec3d contact_position = p1 + n_edge * dot(sphereToEdge, n_edge);
        return std::make_tuple(true, contact_position);
      }
    }
    return std::make_tuple(false, default_contact_point);
  }

  std::tuple<bool, exanb::Vec3d, int> intersect_sphere(const double rx, const double ry, const double rz,
                                                const double rad) const {
    exanb::Vec3d center = {rx, ry, rz};
    const exanb::Vec3d default_norm = {0, 0, 0};

    double p = exanb::dot(center, normal_) - offset_;
    if (std::abs(p) >= rad) {
      return std::make_tuple(false, default_norm, -1);
    }

    const int nb_vertices = vertices_.size();
    exanb::Vec3d v1 = vertices_[1] - vertices_[0];
    exanb::Vec3d v2 = vertices_[nb_vertices - 1] - vertices_[0];
    v1 = v1 / exanb::norm(v1);
    exanb::Vec3d n = exanb::cross(v1, v2);
    n = n / exanb::norm(n);
    exanb::Vec3d iv = center - vertices_[0];
    double dist = exanb::dot(iv, n);
    if (dist < 0.0) {
      dist = -dist;
      n = -n;
    }

    // test if the sphere intersects the surface
    int intersections = 0;

    // from rockable
    exanb::Vec3d P = iv - dist * n;
    v2 = exanb::cross(n, v1);
    double ori1 = exanb::dot(P, v1);
    double ori2 = exanb::dot(P, v2);

    for (int iva = 0; iva < nb_vertices; ++iva) {
      int ivb = iva + 1;
      if (ivb == nb_vertices) ivb = 0;
      const exanb::Vec3d& posNodeA_jv = vertices_[iva];
      const exanb::Vec3d& posNodeB_jv = vertices_[ivb];
      double pa1 = exanb::dot(posNodeA_jv, v1);
      double pb1 = exanb::dot(posNodeB_jv, v1);
      double pa2 = exanb::dot(posNodeA_jv, v2);
      double pb2 = exanb::dot(posNodeB_jv, v2);

      // @see http://local.wasp.uwa.edu.au/~pbourke/geometry/insidepoly/
      // @see http://alienryderflex.com/polygon/
      if ((pa2 < ori2 && pb2 >= ori2) || (pb2 < ori2 && pa2 >= ori2)) {
        if (pa1 + (ori2 - pa2) / (pb2 - pa2) * (pb1 - pa1) < ori1) {
          intersections = 1 - intersections;
        }
      }
    }

    if (intersections == 1) {  // ODD
      exanb::Vec3d contact_position = normal_ * offset_;  // we need dot(conatct_position, normal_)
      return std::make_tuple(true, contact_position, 0);
    }

    // test if the sphere intersects an edge
    for (size_t i = 0; i < vertices_.size(); ++i) {
      exanb::Vec3d p1 = vertices_[i];
      exanb::Vec3d p2 = vertices_[(i + 1) % vertices_.size()];
      exanb::Vec3d edge = p2 - p1;
      exanb::Vec3d sphereToEdge = center - p1;

      // Calculer la distance entre le centre de la sphère et le bord le plus proche
      double distanceToEdge = length(exanb::cross(edge, sphereToEdge)) / length(edge);

      if (distanceToEdge <= rad && exanb::dot(sphereToEdge, edge) > 0 && exanb::dot(sphereToEdge - edge, edge) < 0) {
        auto n_edge = edge / exanb::norm(edge);
        exanb::Vec3d contact_position = p1 + n_edge * dot(sphereToEdge, n_edge);
        return std::make_tuple(true, contact_position, 1);  // La sphère touche un bord
      }
    }
    return std::make_tuple(false, default_norm, -1);
  }

  /**
   * @brief Computes the normal_ vector and offset_ for a face defined by its vertices_.
   *
   * This function calculates the normal_ vector and offset_ for a face based on its vertices_. It returns the computed
   * normal_ vector, offset_, and a boolean indicating success or failure.
   *
   * @return A tuple containing three values:
   *         - `Vec3d` normal_: The computed normal_ vector.
   *         - `double` offset_: The computed offset_.
   *         - `bool` success: Indicates whether the calculation was successful (true) or not (false).
   */
  std::tuple<exanb::Vec3d, double, bool> compute_normal_and_offset() {
    exanb::Vec3d _normal;
    double dist = 0;
    if (vertices_.size() < 3) {
      // need three vertices_ at least
      return std::make_tuple(_normal, dist, false);
    }

    exanb::Vec3d v1 = vertices_[1] - vertices_[0];
    exanb::Vec3d v2 = vertices_[2] - vertices_[0];
    _normal = cross(v1, v2);
    _normal = _normal / exanb::norm(_normal);
    dist = dot(_normal, vertices_[0]);
    return std::make_tuple(_normal, dist, true);
  }

  /**
   * @brief Creates a bounding box that contains the vertices_ of a polygon.
   *
   * This function computes a bounding box that encloses the vertices_ of a polygon. It returns the created box.
   *
   * @return A Box struct representing the bounding box of the polygon.
   */
  Box create_box() {
    exanb::Vec3d inf_ = vertices_[0];
    exanb::Vec3d sup_ = inf_;
    for (auto vertex : vertices_) {
      inf_.x = std::min(inf_.x, vertex.x);
      inf_.y = std::min(inf_.y, vertex.y);
      inf_.z = std::min(inf_.z, vertex.z);
      sup_.x = std::max(sup_.x, vertex.x);
      sup_.y = std::max(sup_.y, vertex.y);
      sup_.z = std::max(sup_.z, vertex.z);
    }
    Box res = {inf_, sup_};
    return res;
  }
};
}  // namespace exaDEM
