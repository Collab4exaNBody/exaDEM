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
#include <onika/physics/units.h>

#include <exaDEM/color_log.hpp>
#include <exaDEM/driver_base.hpp>

namespace exaDEM {
// Data fields for a planar surface/wall driver
struct SurfaceFields {
  /** Required geometry parameters */
  double offset_ = 0;               /**< Offset distance from origin along the normal vector. */
  exanb::Vec3d normal_ = {0, 0, 1}; /**< Normalized normal vector of the surface plane. */
  exanb::Vec3d center_ = {0, 0, 0}; /**< Reference center position of the surface. */

  /** Optional motion parameters */
  exanb::Vec3d vel_ = {0, 0, 0};              /**< Linear velocity of the surface. */
  exanb::Vec3d vrot_ = exanb::Vec3d{0, 0, 0}; /**< Angular velocity (rotation) of the surface. */
  exanb::Vec3d forces_ = {0, 0, 0};           /**< Accumulated forces applied to the surface from interactions. */
  exanb::Vec3d mom_ = {0, 0, 0};              /**< Accumulated moments (torques) applied to the surface. */
  double mass_ = std::numeric_limits<double>::max() / 4; /**< Mass of the surface. */
  double surface_ = -1;                                  /**< Contact surface area (for compressive motion). */

  /** Derived quantities (not persisted) */
  exanb::Vec3d center_proj_; /**< Center position projected onto the normal (offset * normal). */
  double acc_ = 0;           /**< Scalar acceleration along the normal direction. */
};
}  // namespace exaDEM

namespace YAML {
template <>
struct convert<exaDEM::SurfaceFields> {
  static bool decode(const Node& node, exaDEM::SurfaceFields& v) {
    if (!node.IsMap()) {
      return false;
    }
    if (!check_error(node, "offset")) {
      return false;
    }
    if (!check_error(node, "normal")) {
      return false;
    }
    v.offset_ = node["offset"].as<Quantity>().convert();
    v.normal_ = node["normal"].as<exanb::Vec3d>();
    if (check(node, "vel")) {
      v.vel_ = node["vel"].as<exanb::Vec3d>();
    }
    if (check(node, "vrot")) {
      v.vrot_ = node["vrot"].as<exanb::Vec3d>();
    }
    if (check(node, "mass")) {
      v.mass_ = node["mass"].as<Quantity>().convert();
    }
    if (check(node, "surface")) {
      v.surface_ = node["surface"].as<Quantity>().convert();
    }
    if (v.vrot_ != exanb::Vec3d{0, 0, 0}) {
      if (!check_error(node, "center")) return false;
      v.center_ = node["center"].as<exanb::Vec3d>();
    } else {
      if (check(node, "center")) {
        v.center_ = node["center"].as<exanb::Vec3d>();
      } else {
        v.center_ = v.offset_ * v.normal_;
      }
    }
    return true;
  }
};
}  // namespace YAML

namespace exaDEM {
/**
 * @brief Planar surface/wall driver for DEM simulations.
 *
 * Represents a flat rigid surface (wall) defined by a normal vector and offset.
 * Can move in various ways: stationary, linear motion, compressive force, shaker, or pendulum motion.
 * The surface geometry is always planar (infinite plane).
 */
struct Surface {
  SurfaceFields fields_;
  MotionType motion_type_;

  /**
   * @brief Get the type of the driver (in this case, SURFACE).
   * @return The type of the driver.
   */
  constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::SURFACE; }

  /**
   * @brief Print information about the surface.
   */
  inline void print() const {
    exanb::lout << "Driver Type: Surface" << std::endl;
    exanb::lout << "Offset: " << fields_.offset_ << std::endl;
    exanb::lout << "Normal: " << fields_.normal_ << std::endl;
    exanb::lout << "Center: " << fields_.center_ << std::endl;
    exanb::lout << "Vel   : " << fields_.vel_ << std::endl;
    exanb::lout << "AngVel: " << fields_.vrot_ << std::endl;
    if (is_compressive(motion_type_)) {
      exanb::lout << "Acceleration: " << fields_.acc_ << std::endl;
      exanb::lout << "Surface Value [>0]: " << fields_.surface_ << std::endl;
    }
  }

  /**
   * @brief Write surface data into a stream.
   */
  inline void dump_driver(const Driver_params& motion, int id, std::stringstream& stream) {
    stream << "  - register_surface:" << std::endl;
    stream << "     id: " << id << std::endl;
    stream << "     state: {offset: " << fields_.offset_;
    stream << ", center: [" << fields_.center_ << "]";
    stream << ", normal: [" << fields_.normal_ << "]";
    stream << ", vel: [" << fields_.vel_ << "]";
    stream << ", vrot: [" << fields_.vrot_ << "]";
    stream << ", surface: " << fields_.surface_;
    stream << "}" << std::endl;
    motion.dump_driver_params(motion_type_, stream);
  }

  /**
   * @brief Initialize the surface.
   * @details Calculates the center position based on the normal and offset.
   */
  inline void initialize(Driver_params& motion) {
    const std::vector<MotionType> surface_valid_motion_types = {STATIONARY, LINEAR_MOTION, LINEAR_COMPRESSIVE_MOTION,
                                                                SHAKER, PENDULUM_MOTION};

    fields_.center_proj_ = fields_.offset_ * fields_.normal_;

    if (motion_type_ == MotionType::PENDULUM_MOTION) {
      if (fields_.center_ != motion.pendulum_anchor_point_) {
        color_log::warning("register_surface", "Surface center should be equal to pendulum_anchor_point");
        color_log::warning("register_surface",
                           "Surface center is update to [" + std::to_string(motion.pendulum_anchor_point_) + "]");
        fields_.center_ = motion.pendulum_anchor_point_;
      }
      exanb::Vec3d axis = motion.pendulum_anchor_point_ - motion.pendulum_initial_position_;
      if (exanb::dot(axis, motion.pendulum_swing_dir_) >= 1e-14) {
        color_log::error("register_surface",
                         "The vector from pendulum_anchor_point to pendulum_initial_position must be orthogonal to "
                         "pendulum_swing_dir.");
      }
    }

    // checks
    if (exanb::dot(fields_.center_ - fields_.center_proj_, fields_.normal_) >= 1e-14) {
      color_log::warning("register_surface", "The Center point (surface) is not correctly defined");
      fields_.center_ =
          exanb::dot(fields_.center_proj_ - fields_.center_, fields_.normal_) * fields_.normal_ + fields_.center_proj_;
      color_log::warning("register_surface",
                         "center is re-computed because it doesn't fit with offset, new center is: [" +
                             std::to_string(fields_.center_) + "] and center_proj is: [" +
                             std::to_string(fields_.center_proj_) + "]");
    }

    if (!is_valid_motion_type(motion_type_, surface_valid_motion_types)) {
      color_log::error("register_surface", "Invalid Motion Type.");
    } else if (!motion.check_motion_coherence(motion_type_)) {
      color_log::error("register_surface", "Invalid Coherency [Motion Type].");
    } else if (fields_.mass_ <= 0.0) {
      color_log::error("register_surface", "Please, define a positive mass.");
    }
    if (is_linear(motion_type_)) {
      // We do not accept that motion_vector is not equal to -normal for compression mode
      if (fields_.normal_ != motion.motion_vector_ &&
          (fields_.normal_ != -motion.motion_vector_ && !is_compressive(motion_type_))) {
        color_log::warning("register_surface",
                           "The motion vector of the surface has been adjusted to align with the normal vector, i.e. "
                           "the motion vecor[" +
                               std::to_string(motion.motion_vector_) + "] is now equal to [" +
                               std::to_string(fields_.normal_) + "].");
        motion.motion_vector_ = fields_.normal_;
      }
    }

    if (is_compressive(motion_type_)) {
      if (fields_.surface_ <= 0) {
        color_log::error("register_surface",
                         "The surface value must be positive for LINEAR_COMPRESSIVE_FORCE. You need to specify "
                         "surface: XX in the 'state' slot.");
      }
    }
  }

  /**
   * @brief Convert accumulated forces to acceleration along the normal direction.
   * @param motion Driver motion parameters and constraints.
   */
  void force_to_accel(const Driver_params& motion) {
    if (is_compressive(motion_type_)) {
      constexpr double C = 0.5;

      if (motion.mass_ != 0) {
        const double contact_surface = fields_.surface_;

        // Net force vector: F_contact - σ·S·n
        exanb::Vec3d tmp = (fields_.forces_ - motion.sigma_ * contact_surface * motion.motion_vector_) / (motion.mass_ * C);

        // Project onto motion axis → scalar acceleration (Surface driver, unlike Vec3d in particle driver)
        fields_.acc_ = exanb::dot(tmp, motion.motion_vector_);

      } else {
        fields_.acc_ = 0;  // zero mass: reset to avoid stale values
      }
    }
  }

  /**
   * @brief Update surface velocity based on acceleration and motion type.
   * @param motion Driver motion parameters.
   * @param dt Time step.
   */
  inline void push_f_v(const Driver_params& motion, const double dt) {
    if (is_stationary(motion_type_)) {
      // Stationary particle: enforce zero velocity regardless of any accumulated acceleration.
      fields_.vel_ = {0, 0, 0};

    } else if (is_compressive(motion_type_)) {
      // Compressive (stress-controlled) motion: increment velocity only when a target stress
      // sigma is defined (sigma == 0 means stress control is inactive / free boundary).
      // The acceleration is projected onto the motion axis to ensure uniaxial compression.
      if (motion.sigma_ != 0) {
        fields_.vel_ += (dt * fields_.acc_) * motion.motion_vector_;
      }

    } else if (motion_type_ == MotionType::LINEAR_MOTION) {
      // Linear (kinematic) motion: override velocity with the prescribed constant velocity.
      // Reset instead of increment to avoid drift from previous integration steps.
      fields_.vel_ = motion.const_vel_ * motion.motion_vector_;  // I prefer reset it
    }
  }

  /**
   * @brief Updates the position of the wall for the current time step.
   * @param motion  Driver parameters (motion vector, shaker config, pendulum config, etc.).
   * @param time    Current simulation time.
   * @param dt      Raw time step.
   */
  inline void push_f_v_r(const Driver_params& motion, const double time, const double dt) {
    if (!is_stationary(motion_type_)) {
      if (motion_type_ == MotionType::LINEAR_MOTION) {
        // Sanity check: for linear motion the velocity must match the prescribed
        // 1e-12 is an arbitrary small number to account for numerical precision issues.
        assert(exanb::norm(fields_.vel_ - motion.const_vel_ * motion.motion_vector_) < 1e-12);
      }

      if (motion_type_ == MotionType::PENDULUM_MOTION) {
        // Pendulum motion: fully recompute geometry and kinematics at (time + dt).
        // Early return: standard Verlet displacement does not apply here.
        auto [Offset, Normal] = motion.compute_offset_normal_pendulum_motion(time + dt);
        fields_.normal_ = Normal;
        fields_.offset_ = Offset;
        fields_.vel_ = motion.pendulum_velocity(time + dt);
        fields_.center_proj_ = fields_.normal_ * fields_.offset_;
        return;
      }

      // Default Velocity Verlet displacement along the wall normal:
      // This may be overridden below for shaker motion.
      double displ = dt * exanb::dot(fields_.vel_, fields_.normal_) + 0.5 * dt * dt * fields_.acc_;

      if (motion_type_ == MotionType::SHAKER) {
        // Shaker motion: displacement is driven by the waveform signal difference
        const double signal_next = motion.shaker_signal(time + dt);
        const double signal_current = motion.shaker_signal(time);
        const double angle_factor = exanb::dot(motion.shaker_direction(), fields_.normal_);
        displ = (signal_next - signal_current) * angle_factor;
        fields_.vel_ = motion.shaker_velocity(time + dt);
      }

      // Apply displacement to wall position, offset, and projected center.
      fields_.center_ += displ * fields_.normal_;
      fields_.offset_ += displ;
      fields_.center_proj_ += displ * fields_.normal_;
    }
  }

  /**
   * @brief Field accessors for position, velocity, forces, and moments.
   */
  // Position (center) getter
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
   * @brief Check if a point is close enough to the surface for potential interaction.
   * @details Uses distance perpendicular to the surface (along normal direction).\n   * @param rcut The cut-off radius
   * for interaction detection.
   * @param p The point to check.
   * @return True if the point is within cut-off radius of the surface, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool filter(const double rcut, const exanb::Vec3d& p) {
    exanb::Vec3d proj = dot(p, fields_.normal_) * fields_.normal_;
    double d = norm(proj - fields_.center_proj_);
    return d <= rcut;
  }

  /**
   * @brief Detect collision between a point and the surface plane.
   * @details Computes penetration depth and contact information for particle-surface interaction.
   * @param rcut The cut-off radius for collision detection.
   * @param p The point (particle center) to check for collision.
   * @return A tuple containing:
   *         - bool: true if collision/contact is detected within cut-off radius.
   *         - double: penetration depth (negative = on approaching side, positive = inside surface).
   *         - Vec3d: surface normal vector at contact point.
   *         - Vec3d: contact position on the surface plane.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline std::tuple<bool, double, exanb::Vec3d, exanb::Vec3d> detector(const double rcut, const exanb::Vec3d& p) {
    exanb::Vec3d proj = dot(p, fields_.normal_) * fields_.normal_;
    exanb::Vec3d surface_to_point = -(fields_.center_proj_ - proj);
    double d = exanb::norm(surface_to_point);
    double dn = d - rcut;
    if (dn > 0) {
      return {false, dn, exanb::Vec3d(), exanb::Vec3d()};
    } else {
      exanb::Vec3d n = surface_to_point / d;
      exanb::Vec3d contact_position = p - n * (rcut + 0.5 * dn);
      return {true, dn, n, contact_position};
    }
  }
};

template <>
struct DriverProperty<Surface> {
  /// No moment/torque computation required for surface drivers.
  static constexpr bool use_moment = false;
  /// No quaternion orientation tracking required for surface drivers.
  static constexpr bool use_quaternion = false;
};
}  // namespace exaDEM

namespace onika {
namespace memory {
template <>
struct MemoryUsage<exaDEM::Surface> {
  static inline size_t memory_bytes(const exaDEM::Surface& obj) { return onika::memory::memory_bytes(obj); }
};
}  // namespace memory
}  // namespace onika
