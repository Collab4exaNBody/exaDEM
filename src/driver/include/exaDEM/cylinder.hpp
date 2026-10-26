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

#include <onika/physics/units.h>

#include <exaDEM/color_log.hpp>
#include <exaDEM/driver_base.hpp>

namespace exaDEM {
// Data fields for a cylindrical driver
struct CylinderFields {
  double radius_ = -1;              /**< Radius of the cylinder. */
  exanb::Vec3d axis_ = {1, 0, 1};   /**< Component mask (not a direction): the component set to 0 is the cylinder
                                         axis, e.g. {1, 0, 1} for a cylinder along y. */
  exanb::Vec3d center_ = {0, 0, 0}; /**< Center position of the cylinder in 3D space. */
  exanb::Vec3d vel_ = {0, 0, 0};    /**< Linear velocity of the cylinder. */
  exanb::Vec3d vrot_ = {0, 0, 0};   /**< Angular velocity (rotation) of the cylinder. */
  exanb::Vec3d forces_ = {0, 0, 0}; /**< Accumulated forces applied to the cylinder from interactions. */
  exanb::Vec3d mom_ = {0, 0, 0};    /**< Accumulated moments (torques) applied to the cylinder. */
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
    v.radius_ = node["radius"].as<Quantity>().convert();
    v.axis_ = node["axis"].as<exanb::Vec3d>();
    v.center_ = node["center"].as<exanb::Vec3d>();
    if (check(node, "vel")) {
      v.vel_ = node["vel"].as<exanb::Vec3d>();
    }
    if (check(node, "vrot")) {
      v.vrot_ = node["vrot"].as<exanb::Vec3d>();
    }
    return true;
  }
};
}  // namespace YAML

namespace exaDEM {

/**
 * @brief Cylindrical driver for DEM simulations.
 *
 * Represents a cylinder-shaped rigid body driver that can interact with particles.
 * Currently supports only stationary motion type. The cylinder is defined by a radius,
 * an axis direction, and a center position.
 */
struct Cylinder {
  CylinderFields fields_;
  MotionType motion_type_;

  /**
   * @brief Get the type of the driver (in this case, CYLINDER).
   * @return The type of the driver.
   */
  constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::CYLINDER; }

  /**
   * @brief Check that an axis is a component mask: exactly one component equal to 0 and the two others equal to 1.
   * @param axis The axis mask, e.g. {1, 0, 1} for a cylinder along y.
   * @return True if the mask is valid.
   */
  static inline bool is_valid_axis_mask(const exanb::Vec3d& axis) {
    int zeros = 0, ones = 0;
    for (double c : {axis.x, axis.y, axis.z}) {
      zeros += (c == 0.0);
      ones += (c == 1.0);
    }
    return zeros == 1 && ones == 2;
  }

  /**
   * @brief Initialize the cylinder.
   * @details This function asserts that the radius of the cylinder is greater than 0.
   */
  inline void initialize(Driver_params& motion) {
    const std::vector<MotionType> cylinder_valid_motion_types = {STATIONARY, SHAKER};

    if (!is_valid_axis_mask(fields_.axis_)) {
      color_log::error("Cylinder::initialize",
                       "axis [" + std::to_string(fields_.axis_) +
                           "] is not valid: it is a component mask, not a direction. The component set to 0 is the "
                           "cylinder axis, use [0,1,1] (x), [1,0,1] (y) or [1,1,0] (z).");
    }
    if (!is_valid_motion_type(motion_type_, cylinder_valid_motion_types)) {
      std::exit(EXIT_FAILURE);
    } else if (!motion.check_motion_coherence(motion_type_)) {
      std::exit(EXIT_FAILURE);
    }
    assert(fields_.radius_ > 0);
    fields_.center_ = fields_.axis_ * fields_.center_;
  }

  /**
   * @brief Print information about the cylinder.
   */
  inline void print() const {
    exanb::lout << "Driver Type: Cylinder" << std::endl;
    exanb::lout << "Radius: " << fields_.radius_ << std::endl;
    exanb::lout << "Axis  : " << fields_.axis_ << std::endl;
    exanb::lout << "Center: " << fields_.center_ << std::endl;
    exanb::lout << "Vel   : " << fields_.vel_ << std::endl;
    exanb::lout << "AngVel: " << fields_.vrot_ << std::endl;
  }

  /**
   * @brief Write cylinder information into a stream.
   */
  void dump_op(const Driver_params& motion, int id, std::stringstream& stream) {
    stream << "  - register_cylinder:" << std::endl;
    stream << "     id: " << id << std::endl;
    stream << "     state: { radius: " << fields_.radius_;
    stream << ", axis: [" << fields_.axis_ << "]";
    stream << ", center: [" << fields_.center_ << "]";
    stream << ", vel: [" << fields_.vel_ << "]";
    stream << ", vrot: [" << fields_.vrot_ << "]}" << std::endl;
    motion.dump_driver_params(motion_type_, stream);
  }

  /**
   * @brief Write cylinder data into a stream, in the plain drivers: storage format (see
   * dump_drivers/read_drivers), as opposed to dump_op()'s register_cylinder: operator form.
   */
  void dump_data(const Driver_params& motion, int id, std::stringstream& stream) {
    stream << "  - type: CYLINDER" << std::endl;
    stream << "    id: " << id << std::endl;
    stream << "    state: { radius: " << fields_.radius_;
    stream << ", axis: [" << fields_.axis_ << "]";
    stream << ", center: [" << fields_.center_ << "]";
    stream << ", vel: [" << fields_.vel_ << "]";
    stream << ", vrot: [" << fields_.vrot_ << "]}" << std::endl;
    motion.dump_driver_params(motion_type_, stream, 4);
  }

  /**
   * @brief Return driver velocity (not implemented for cylinder).
   * @details Cylinder motion dynamics are not yet implemented.
   *          This placeholder maintains interface compatibility.
   * @param motion Driver motion parameters.
   */
  ONIKA_HOST_DEVICE_FUNC inline void force_to_accel(const Driver_params& motion) { /** not implemented */ }

  /**
   * @brief Update position and velocity (not implemented for cylinder).
   * @details Cylinder kinematics are not yet implemented.
   * @param motion Driver motion parameters.
   * @param dt Time step.
   */
  ONIKA_HOST_DEVICE_FUNC inline void push_f_v(const Driver_params& motion, const double dt) { /** not implemented */ }

  /**
   * @brief Update position, velocity.
   * @details Move center in the shaker direction and update velocity.
   * @param motion Driver motion parameters.
   * @param time Current simulation time.
   * @param dt Time step.
   */
  inline void push_f_v_r(const Driver_params& motion, const double time, const double dt) {
    if (!is_stationary(motion_type_)) {
      if (motion_type_ == MotionType::SHAKER) {
        // Shaker motion: displacement is driven by the waveform signal difference
        const double signal_next = motion.shaker_signal(time + dt);
        const double signal_current = motion.shaker_signal(time);
        const double displ = (signal_next - signal_current);

        fields_.center_ += displ * motion.shaker_direction();
        fields_.vel_ = motion.shaker_velocity(time + dt);
      }
    }
  }

  /**
   * @brief Field accessors for position, velocity, forces, and rotation.
   */
  // Position getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& position() { return fields_.center_; }
  // Linear velocity getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& velocity() { return fields_.vel_; }
  // Accumulated forces getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& forces() { return fields_.forces_; }
  // Accumulated moment (torque) getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& moment() { return fields_.mom_; }
  // Angular velocity getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& angular_velocity() { return fields_.vrot_; }

  /**
   * @brief Compute the direction of the cylinder axis.
   *
   * axis_ is a component mask: the masked component (0) is the cylinder axis,
   * e.g. axis_ = {1, 0, 1} gives a cylinder along y.
   *
   * @return exanb::Vec3d The normalized direction of the cylinder axis.
   */
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d compute_normal() {
    const exanb::Vec3d normal = exanb::Vec3d{1, 1, 1} - fields_.axis_;
    return normal / exanb::norm(normal);
  }

  /**
   * @brief Filter function to check if a point is within a certain radius of the cylinder.
   * @param rcut The cut-off radius. Note: rcut = rverlet + r shape
   * @param vi The vector representing the point to check.
   * @return True if the point is within the cut-off radius of the cylinder, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool filter(const double rcut, const exanb::Vec3d& vi) {
    const exanb::Vec3d proj = vi * fields_.axis_;

    // === direction
    const exanb::Vec3d dir = proj - fields_.center_;

    // === interpenetration
    const double d = exanb::norm(dir);
    const double dn = fields_.radius_ - (d + rcut);
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
    const exanb::Vec3d proj = pi * fields_.axis_;

    // === direction
    const exanb::Vec3d dir = fields_.center_ - proj;

    // === interpenetration
    const double d = exanb::norm(dir);

    // === compute interpenetration
    const double dn = fields_.radius_ - (rcut + d);

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

template <>
struct DriverProperty<Cylinder> {
  static constexpr bool use_moment = false;
  static constexpr bool use_quaternion = false;
};
}  // namespace exaDEM

namespace onika {
namespace memory {

template <>
struct MemoryUsage<exaDEM::Cylinder> {
  static inline size_t memory_bytes(const exaDEM::Cylinder& obj) { return onika::memory::memory_bytes(obj); }
};

}  // namespace memory
}  // namespace onika
