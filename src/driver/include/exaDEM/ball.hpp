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
#include <onika/math/basic_types.h>
#include <exaDEM/color_log.hpp>
#include <exaDEM/driver_base.hpp>

namespace exaDEM {
// Data fields for a spherical driver (ball)
struct BallFields {
  double radius_;                                        /**< Radius of the ball. */
  exanb::Vec3d center_;                                  /**< Center position of the ball in 3D space. */
  exanb::Vec3d vel_ = exanb::Vec3d{0, 0, 0};             /**< Linear velocity of the ball. */
  exanb::Vec3d vrot_ = exanb::Vec3d{0, 0, 0};            /**< Angular velocity (rotation) of the ball. */
  exanb::Vec3d forces_ = {0, 0, 0};                      /**< Accumulated forces applied to the ball from interactions. */
  exanb::Vec3d mom_ = {0, 0, 0};                         /**< Accumulated moments (torques) applied to the ball. */
  double mass_ = std::numeric_limits<double>::max() / 4; /**< Mass of the ball (default: infinite/fixed). */
  double rv_ = 0;                                        /**< Radius velocity for compressive/expanding motion. */

  /** Derived quantities (not persisted) */
  exanb::Vec3d acc_ = {0, 0, 0};                         /**< Linear acceleration of the ball. */
  double ra_ = 0;                                        /**< Radius acceleration for compressive/expanding motion. */
};
}  // namespace exaDEM

namespace YAML {
template <>
struct convert<exaDEM::BallFields> {
  static bool decode(const Node& node, exaDEM::BallFields& v) {
    if (!node.IsMap()) {
      return false;
    }
    if (!check_error(node, "radius")) {
      return false;
    }
    if (!check_error(node, "center")) {
      return false;
    }
    v.radius_ = node["radius"].as<Quantity>().convert();
    v.center_ = node["center"].as<exanb::Vec3d>();
    if (check(node, "vel")) {
      v.vel_ = node["vel"].as<exanb::Vec3d>();
    }
    if (check(node, "vrot")) {
      v.vrot_ = node["vrot"].as<exanb::Vec3d>();
    }
    if (check(node, "rv")) {
      v.rv_ = node["rv"].as<Quantity>().convert();
    }
    if (check(node, "mass")) {
      v.mass_ = node["mass"].as<Quantity>().convert();
    }
    if (check(node, "density") && !check(node, "mass")) {
      const double pi = 4 * atan(1);
      v.mass_ = 4 / 3 * pi * v.radius_ * v.radius_ * v.radius_;
    }
    return true;
  }
};
}  // namespace YAML

namespace exaDEM {

/**
 * @brief Spherical driver (ball) for DEM simulations.
 * 
 * Represents a ball-shaped rigid body driver that can interact with particles.
 * Supports various motion types: stationary, linear, compressive force, and tabulated motion.
 */
struct Ball {
  BallFields fields_;
  MotionType motion_type_;

  Ball(BallFields& bp, MotionType& mt) : fields_(bp), motion_type_(mt) {}

  /**
   * @brief Get the type of the driver (in this case, BALL).
   * @return The type of the driver.
   */
  constexpr DRIVER_TYPE get_type() {
    return DRIVER_TYPE::BALL;
  }

  /**
   * @brief Print information about the ball.
   */
  inline void print() const {
    exanb::lout << "Driver Type: Ball" << std::endl;
    exanb::lout << "Radius: " << fields_.radius_ << std::endl;
    exanb::lout << "Center: " << fields_.center_ << std::endl;
    exanb::lout << "Vel   : " << fields_.vel_ << std::endl;
    exanb::lout << "AngVel: " << fields_.vrot_ << std::endl;
    if (is_compressive(motion_type_)) {
      exanb::lout << "Radius acceleration: " << fields_.ra_ << std::endl;
      exanb::lout << "Radius velocity: " << fields_.rv_ << std::endl;
    }
    if (is_force_motion(motion_type_)) {
      exanb::lout << "Mass: " << fields_.mass_ << std::endl;
    }
  }

  /**
   * @brief Write ball data into a stream.
   */
  void dump_driver(const Driver_params& motion, int id, std::stringstream& stream) {
    stream << "  - register_ball:" << std::endl;
    stream << "     id: " << id << std::endl;
    stream << "     state: { radius:" << fields_.radius_;
    stream << ",center: [" << fields_.center_ << "]";
    stream << ",vel: [" << fields_.vel_ << "]";
    stream << ",vrot: [" << fields_.vrot_ << "]";
    if (is_compressive(motion_type_)) {
      stream << ",rv: " << fields_.rv_;
    }
    if (is_force_motion(motion_type_)) {
      stream << ",mass: " << fields_.mass_;
    }
    stream << "}" << std::endl;
    motion.dump_driver_params(motion_type_, stream);
  }

  /**
   * @brief Initialize the ball.
   * @details This function asserts that the radius of the ball is greater than 0.
   */
  inline void initialize(Driver_params& motion) {
    const std::vector<MotionType> ball_valid_motion_types = {
      STATIONARY,
      LINEAR_MOTION,
      COMPRESSIVE_FORCE,
      TABULATED};

    if (!is_valid_motion_type(motion_type_, ball_valid_motion_types)) {
      color_log::error("Ball::initialize", "Invalid Motion Type.");
    } else if (!motion.check_motion_coherence(motion_type_)) {
      color_log::error("Ball::initialize", "Invalid Motion.");
    } else if (fields_.mass_ <= 0.0) {
      color_log::error("Ball::initialize", "Please, define a positive mass.");
    }
    assert(fields_.radius_ > 0);
  }

  /**
   * @brief Convert accumulated forces to acceleration.
   * @details Updates acceleration based on forces and motion type.
   *          For force-driven motion, acc = forces / mass.
   *          For other motion types, acceleration is zero.
   * @param motion Driver motion parameters and constraints.
   */
  inline void force_to_accel(const Driver_params& motion) {
    if (is_force_motion(motion_type_)) {
      if (fields_.mass_ >= 1e100 || fields_.mass_ <= 0) {
        color_log::warning("Ball::force_to_accel", "The mass of the ball is set to " + std::to_string(fields_.mass_));
      }
      motion.update_forces(motion_type_, fields_.forces_);  // update forces in function of the motion type
      fields_.acc_ = fields_.forces_ / fields_.mass_;
    } else {
      fields_.acc_ = {0, 0, 0};
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
   * @brief Update ball kinematics (position, velocity, radius).
   * @details Handles different motion types:
   *          - Tabulated: reads from motion tables
   *          - Compressive: updates radius via rv and ra
   *          - Others: updates position via dt * velocity
   * @param motion Driver motion parameters
   * @param time Current simulation time
   * @param dt Time step
   */
  inline void push_f_v_r(const Driver_params& motion, const double time, const double dt) {
    if (is_tabulated(motion_type_)) {
      fields_.center_ = motion.tab_to_position(time);
      fields_.vel_ = motion.tab_to_velocity(time);
    } else if (!is_stationary(motion_type_)) {
      if (is_compressive(motion_type_)) {
        push_ra_rv_to_rad(dt);
      }
      fields_.center_ = fields_.center_ + dt * fields_.vel_;
    }
  }

  /**
   * @brief Update the position of the ball.
   * @param dt The time step.
   */
  inline void push_f_v(const Driver_params& motion, const double dt) {
    if (is_force_motion(motion_type_)) {
      fields_.vel_ = fields_.acc_ * dt;
    }

    if (motion_type_ == LINEAR_MOTION) {
      fields_.vel_ = motion.motion_vector_ * motion.const_vel_;  // I prefer reset it
    }

    if (is_compressive(motion_type_)) {
      if (motion_type_ == COMPRESSIVE_FORCE) {
        fields_.vel_ = {0, 0, 0};
      }
      push_ra_to_rv(motion, dt);
    }
  }
  /**
   * @brief Update the "velocity radius" of the ball.
   * @param t The time step.
   */
  inline void push_ra_to_rv(const Driver_params& motion, const double dt) {
    if (is_compressive(motion_type_) && motion.sigma_ != 0) {
      fields_.rv_ += dt * fields_.ra_;
    }
  }

  /**
   * @brief Update the "velocity raduis" of the ball.
   * @param t The time step.
   */
  inline void push_ra_rv_to_rad(const double dt) {
    if (is_compressive(motion_type_)) {
      fields_.radius_ += dt * fields_.rv_ + 0.5 * dt * dt * fields_.ra_;
    }
  }

  /**
   * @brief Compute the surface.
   */
  ONIKA_HOST_DEVICE_FUNC inline double surface() {
    const double pi = 4 * atan(1);
    return 4 * pi * fields_.radius_ * fields_.radius_;
  }

  /**
   * @brief Compute the surface.
   */
  double volume() {
    const double pi = 4 * atan(1);
    return 4 / 3 * pi * fields_.radius_ * fields_.radius_ * fields_.radius_;
  }
  
  /**
   * @brief Update the "velocity radius" of the ball.
   * @param t The time step.
   */
  inline void f_ra(const Driver_params& motion, const double dt) {
    if (is_compressive(motion_type_)) {
      constexpr double C = 0.5;  // I don't remember why, ask Lhassan
      if (motion.mass_ != 0) {
        const double s = surface();
        // forces and mass are defined in Driver_params
        fields_.ra_ = (exanb::norm(fields_.forces_) - motion.sigma_ * s - (motion.damprate_ * fields_.rv_)) / (motion.mass_ * C);
      }
    }
  }

  /**
   * @brief Filter function to check if a point is within a certain radius of the ball.
   * @param rcut The cut-off radius.
   * @param p The point to check.
   * @return True if the point is within the cut-off radius of the ball, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool filter(const double rcut, const exanb::Vec3d& p) {
    const exanb::Vec3d dist = fields_.center_ - p;
    double d = fields_.radius_ - norm(dist);
    return std::fabs(d) <= rcut;
  }

  /**
   * @brief Detects collision between a vertex and the ball.
   * @param rcut The cut-off radius.
   * @param p The point to check for collision.
   * @return A tuple containing:
   *         - A boolean indicating whether a collision occurred.
   *         - The penetration depth (negative if inside the ball).
   *         - The normal vector pointing from the collision point to the center of the ball.
   *         - The contact position on the surface of the ball.
   */
  ONIKA_HOST_DEVICE_FUNC
      inline std::tuple<bool, double, exanb::Vec3d, exanb::Vec3d> detector(const double rcut, const exanb::Vec3d& p) {
        exanb::Vec3d point_to_center = fields_.center_ - p;
        double d = norm(point_to_center);
        double dn;
        exanb::Vec3d n = point_to_center / d;
        if (d > fields_.radius_) {
          dn = d - fields_.radius_ - rcut;
          n = (-1) * n;
        } else {
          dn = fields_.radius_ - d - rcut;
        }

        if (dn > 0) {
          return {false, dn, exanb::Vec3d(), exanb::Vec3d()};
        } else {
          exanb::Vec3d contact_position = p - n * (rcut + 0.5 * dn);
          return {true, dn, n, contact_position};
        }
      }
};

template<>
struct DriverProperty<Ball> {
  static constexpr bool use_moment = false;
  static constexpr bool use_quaternion = false;
};
}  // namespace exaDEM

namespace onika {
namespace memory {

template <>
struct MemoryUsage<exaDEM::Ball> {
  static inline size_t memory_bytes(const exaDEM::Ball& obj) {
    return onika::memory::memory_bytes(obj);
  }
};

}  // namespace memory
}  // namespace onika
