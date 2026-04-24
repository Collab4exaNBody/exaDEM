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

#include <exaDEM/color_log.hpp>
#include <exaDEM/driver_base.hpp>
#include <onika/physics/units.h>

namespace exaDEM {
struct CylinderFields {
  double radius = -1;              /**< Radius of the cylinder. */
  exanb::Vec3d axis = {1, 0, 1};   /**< Axis direction of the cylinder. */
  exanb::Vec3d center = {0, 0, 0}; /**< Center position of the cylinder. */
  exanb::Vec3d vel = {0, 0, 0};    /**< Velocity of the cylinder. */
  exanb::Vec3d vrot = {0, 0, 0};   /**< Angular velocity of the cylinder. */
};
}  // namespace exaDEM

namespace YAML {
template <>
struct convert<exaDEM::CylinderFields> {
  static bool decode(const Node& node, exaDEM::CylinderFields& v) {
    if (!node.IsMap()) {
      return false;
    }
    if (!check_error(node, "radius")) {
      return false;
    }
    if (!check_error(node, "axis")) {
      return false;
    }
    if (!check_error(node, "center")) {
      return false;
    }
    v.radius = node["radius"].as<Quantity>().convert();
    v.axis = node["axis"].as<exanb::Vec3d>();
    v.center = node["center"].as<exanb::Vec3d>();
    if (check(node, "vel")) {
      v.vel = node["vel"].as<exanb::Vec3d>();
    }
    if (check(node, "vrot")) {
      v.vrot = node["vrot"].as<exanb::Vec3d>();
    }
    return true;
  }
};
}  // namespace YAML

namespace exaDEM {

/**
 * @brief Struct representing a cylinder in the exaDEM simulation.
 */
struct Cylinder {
  CylinderFields fields;
  Driver_params motion;

  /**
   * @brief Get the type of the driver (in this case, CYLINDER).
   * @return The type of the driver.
   */
  constexpr DRIVER_TYPE get_type() {
    return DRIVER_TYPE::CYLINDER;
  }

  /**
   * @brief Initialize the cylinder.
   * @details This function asserts that the radius of the cylinder is greater than 0.
   */
  inline void initialize() {
    const std::vector<MotionType> cylinder_valid_motion_types = {
      STATIONARY};

    if (!motion.is_valid_motion_type(cylinder_valid_motion_types)) {
      std::exit(EXIT_FAILURE);
    } else if (!motion.check_motion_coherence()) {
      std::exit(EXIT_FAILURE);
    }
    assert(fields.radius > 0);
    fields.center = fields.axis * fields.center;
  }

  /**
   * @brief Print information about the cylinder.
   */
  inline void print() const {
    exanb::lout << "Driver Type: Cylinder" << std::endl;
    exanb::lout << "Radius: " << fields.radius << std::endl;
    exanb::lout << "Axis  : " << fields.axis << std::endl;
    exanb::lout << "Center: " << fields.center << std::endl;
    exanb::lout << "Vel   : " << fields.vel << std::endl;
    exanb::lout << "AngVel: " << fields.vrot << std::endl;
    motion.print_driver_params();
  }

  /**
   * @brief Write cylinder information into a stream.
   */
  void dump_driver(int id, std::stringstream& stream) {
    stream << "  - register_cylinder:" << std::endl;
    stream << "     id: " << id << std::endl;
    stream << "     state: { radius: " << fields.radius;
    stream << ",axis: [" << fields.axis << "]";
    stream << ",center: [" << fields.center << "]";
    stream << ",vel: [" << fields.vel << "]";
    stream << ",vrot: [" << fields.vrot << "]}" << std::endl;
    motion.dump_driver_params(stream);
  }

  /**
   * @brief return driver velocity
   */
  ONIKA_HOST_DEVICE_FUNC inline void force_to_accel() {
    /** not implemented */
  }
  ONIKA_HOST_DEVICE_FUNC inline void push_f_v(const double dt) {
    /** not implemented */
  }
  ONIKA_HOST_DEVICE_FUNC inline void push_f_v_r(const double time, const double dt) {
    /** not implemented */
  }
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d get_vel() {
    return fields.vel;
  }

  /**
   * @brief Compute a normal vector associated with the given axis.
   *
   * This function constructs two vectors in a plane orthogonal to the
   * provided axis (via a crude projection method), then computes their
   * cross product to obtain a normal vector.
   *
   * @note
   * - This implementation is not optimized.
   * - The method assumes the axis is aligned with Cartesian directions.
   *   Behavior is undefined or incorrect for arbitrary (non-axis-aligned) vectors.
   *
   * @warning
   * The projection used here is not mathematically rigorous for general cases.
   * It may produce incorrect normals if `axis` is not aligned with (1,0,0),
   * (0,1,0), or (0,0,1).
   *
   * @return exanb::Vec3d A normalized vector perpendicular to the constructed plane.
   */
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d get_normal() {
    exanb::Vec3d p1 = {1,1,1};
    exanb::Vec3d p2 = {-1,-1,-1};
    exanb::Vec3d p3 = {-1,1,1};
    p3 = p3 * fields.axis; // projection
    p1 = p1 * fields.axis - p3; // projection
    p2 = p2 * fields.axis - p3; // projection
    exanb::Vec3d normal = exanb::cross(p1, p2);
    return normal / exanb::norm(normal);
  }

  /**
   * @brief Filter function to check if a point is within a certain radius of the cylinder.
   * @param rcut The cut-off radius. Note: rcut = rverlet + r shape
   * @param vi The vector representing the point to check.
   * @return True if the point is within the cut-off radius of the cylinder, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool filter(const double rcut, const exanb::Vec3d& vi) {
    const exanb::Vec3d proj = vi * fields.axis;

    // === direction
    const exanb::Vec3d dir = proj - fields.center;

    // === interpenetration
    const double d = exanb::norm(dir);
    const double dn = fields.radius - (d + rcut);
    return dn <= 0;
  }

  /**
   * @brief Detects the intersection between a vertex of a polyhedron and a cylinder.
   *
   * This function checks if a vertex, represented by its position 'pi' and orientation 'oi',
   * intersects with a cylindrical shape defined by its center projection 'center_proj', axis 'axis',
   * and radius 'radius'.
   *
   * @param rcut The shape radius.
   * @param pi The position of the vertex.
   *
   * @return A tuple containing:
   *   - A boolean indicating whether there is an intersection (true) or not (false).
   *   - The penetration depth in case of intersection.
   *   - The contact normal at the intersection point.
   *   - The contact point between the vertex and the cylinder.
   */
  // rcut = r shape
  ONIKA_HOST_DEVICE_FUNC
      inline std::tuple<bool, double, exanb::Vec3d, exanb::Vec3d> detector(const double rcut, const exanb::Vec3d& pi) {
        // === project the vertex in the plan as the cylinder center
        const exanb::Vec3d proj = pi * fields.axis;

        // === direction
        const exanb::Vec3d dir = fields.center - proj;

        // === interpenetration
        const double d = exanb::norm(dir);

        // === compute interpenetration
        const double dn = fields.radius - (rcut + d);

        if (dn > 0) {
          return {false, dn, exanb::Vec3d(), exanb::Vec3d()};
        } else {
          // === compute contact normal
          const exanb::Vec3d n = dir / d;

          // === compute contact point
          const exanb::Vec3d contact_position = pi - n * (rcut + 0.5 * dn);

          return {true, dn, n, contact_position};
        }
      }
};
}  // namespace exaDEM

namespace onika {
namespace memory {

template <>
struct MemoryUsage<exaDEM::Cylinder> {
  static inline size_t memory_bytes(const exaDEM::Cylinder& obj) {
    return onika::memory::memory_bytes(obj);
  }
};

}  // namespace memory
}  // namespace onika
