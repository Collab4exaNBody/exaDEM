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
struct BallFields {
  double radius;                                        /**< Radius of the ball. */
  exanb::Vec3d center;                                  /**< Center position of the ball. */
  exanb::Vec3d vel = Vec3d{0, 0, 0};                    /**< Velocity of the ball. */
  exanb::Vec3d vrot = Vec3d{0, 0, 0};                   /**< Angular velocity of the ball. */
  double mass = std::numeric_limits<double>::max() / 4; /**< Mass of the ball */
  double rv = 0;                                        /**< */

  /** We don't need to save these values */
  exanb::Vec3d acc = {0, 0, 0}; /**< Acceleration of the ball. */
  double ra = 0;                /**< */
};
}  // namespace exaDEM

namespace YAML {
using exaDEM::BallFields;
using exaDEM::MotionType;
using exanb::lerr;
using onika::physics::Quantity;

template <>
struct convert<BallFields> {
  static bool decode(const Node& node, BallFields& v) {
    if (!node.IsMap()) {
      return false;
    }
    if (!check_error(node, "radius")) {
      return false;
    }
    if (!check_error(node, "center")) {
      return false;
    }
    v.radius = node["radius"].as<Quantity>().convert();
    v.center = node["center"].as<Vec3d>();
    if (check(node, "vel")) {
      v.vel = node["vel"].as<Vec3d>();
    }
    if (check(node, "vrot")) {
      v.vrot = node["vrot"].as<Vec3d>();
    }
    if (check(node, "rv")) {
      v.rv = node["rv"].as<double>();
    }
    if (check(node, "mass")) {
      v.mass = node["mass"].as<double>();
    }
    if (check(node, "density") && !check(node, "mass")) {
      const double pi = 4 * atan(1);
      v.mass = 4 / 3 * pi * v.radius * v.radius * v.radius;
    }
    return true;
  }
};
}  // namespace YAML

namespace exaDEM {

/**
 * @brief Struct representing a ball in the exaDEM simulation.
 */
struct Ball {
  BallFields fields;
  Driver_params motion;

  Ball(BallFields& bp, Driver_params& dp) : fields(bp), motion(dp) {}

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
    exanb::lout << "Radius: " << fields.radius << std::endl;
    exanb::lout << "Center: " << fields.center << std::endl;
    exanb::lout << "Vel   : " << fields.vel << std::endl;
    exanb::lout << "AngVel: " << fields.vrot << std::endl;
    if (motion.is_compressive()) {
      exanb::lout << "Radius acceleration: " << fields.ra << std::endl;
      exanb::lout << "Radius velocity: " << fields.rv << std::endl;
    }
    if (motion.is_force_motion()) {
      exanb::lout << "Mass: " << fields.mass << std::endl;
    }
    motion.print_driver_params();
  }

  /**
   * @brief Write ball data into a stream.
   */
  void dump_driver(int id, std::stringstream& stream) {
    stream << "  - register_ball:" << std::endl;
    stream << "     id: " << id << std::endl;
    stream << "     state: { radius:" << fields.radius;
    stream << ",center: [" << fields.center << "]";
    stream << ",vel: [" << fields.vel << "]";
    stream << ",vrot: [" << fields.vrot << "]";
    if (motion.is_compressive()) {
      stream << ",rv: " << fields.rv;
    }
    if (motion.is_force_motion()) {
      stream << ",mass: " << fields.mass;
    }
    stream << "}" << std::endl;
    motion.dump_driver_params(stream);
  }

  /**
   * @brief Initialize the ball.
   * @details This function asserts that the radius of the ball is greater than 0.
   */
  inline void initialize() {
    const std::vector<MotionType> ball_valid_motion_types = {
      STATIONARY,
      LINEAR_MOTION,
      COMPRESSIVE_FORCE,
      TABULATED};

    if (!motion.is_valid_motion_type(ball_valid_motion_types)) {
      color_log::error("Ball::initialize", "Invalid Motion Type.");
    } else if (!motion.check_motion_coherence()) {
      color_log::error("Ball::initialize", "Invalid Motion.");
    } else if (fields.mass <= 0.0) {
      color_log::error("Ball::initialize", "Please, define a positive mass.");
    }
    assert(fields.radius > 0);
  }

  /**
   * @brief Update the position of the ball.
   * @param dt The time step.
   */
  inline void force_to_accel() {
    if (motion.is_force_motion()) {
      if (fields.mass >= 1e100) {
        color_log::warning("f_to_a", "The mass of the ball is set to " + std::to_string(fields.mass));
      }
      fields.acc = motion.sum_forces() / fields.mass;
    } else {
      fields.acc = {0, 0, 0};
    }
  }

  /**
   * @brief return driver velocity
   */
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& get_vel() {
    return fields.vel;
  }

  /**
   * @brief Update the position of the ball.
   * @param dt The time step.
   */
  inline void push_f_v_r(const double time, const double dt) {
    if (motion.is_tabulated()) {
      fields.center = motion.tab_to_position(time);
      fields.vel = motion.tab_to_velocity(time);
    } else if (!motion.is_stationary()) {
      if (motion.is_compressive()) {
        push_ra_rv_to_rad(dt);
      }
      fields.center = fields.center + dt * fields.vel;
    }
  }

  /**
   * @brief Update the position of the ball.
   * @param dt The time step.
   */
  inline void push_f_v(const double dt) {
    if (motion.is_force_motion()) {
      fields.vel = fields.acc * dt;
    }

    if (motion.motion_type == LINEAR_MOTION) {
      fields.vel = motion.motion_vector * motion.const_vel;  // I prefere reset it
    }

    if (motion.is_compressive()) {
      if (motion.motion_type == COMPRESSIVE_FORCE) {
        fields.vel = {0, 0, 0};
      }
      push_ra_to_rv(dt);
    }
  }
  /**
   * @brief Update the "velocity raduis" of the ball.
   * @param t The time step.
   */
  inline void push_ra_to_rv(const double dt) {
    if (motion.is_compressive()) {
      if (motion.sigma != 0) {
        fields.rv += 0.5 * dt * fields.ra;
      }
    }
  }

  /**
   * @brief Update the "velocity raduis" of the ball.
   * @param t The time step.
   */
  inline void push_ra_rv_to_rad(const double dt) {
    if (motion.is_compressive()) {
      fields.radius += dt * fields.rv + 0.5 * dt * dt * fields.ra;
    }
  }

  /**
   * @brief Compute the surface.
   */
  ONIKA_HOST_DEVICE_FUNC inline double surface() {
    const double pi = 4 * atan(1);
    return 4 * pi * fields.radius * fields.radius;
  }

  /**
   * @brief Compute the surface.
   */
  double volume() {
    const double pi = 4 * atan(1);
    return 4 / 3 * pi * fields.radius * fields.radius * fields.radius;
  }
  /**
   * @brief Update the "velocity radius" of the ball.
   * @param t The time step.
   */
  inline void f_ra(const double dt) {
    if (motion.is_compressive()) {
      constexpr double C = 0.5;  // I don't remember why, ask Lhassan
      if (motion.weigth != 0) {
        const double s = surface();
        // forces and weigth are defined in Driver_params
        fields.ra = (exanb::norm(motion.forces) - motion.sigma * s - (motion.damprate * fields.rv)) / (motion.weigth * C);
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
    const exanb::Vec3d dist = fields.center - p;
    double d = fields.radius - norm(dist);
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
        Vec3d point_to_center = fields.center - p;
        double d = norm(point_to_center);
        double dn;
        Vec3d n = point_to_center / d;
        if (d > fields.radius) {
          dn = d - fields.radius - rcut;
          n = (-1) * n;
        } else {
          dn = fields.radius - d - rcut;
        }

        if (dn > 0) {
          return {false, dn, exanb::Vec3d(), exanb::Vec3d()};
        } else {
          exanb::Vec3d contact_position = p - n * (rcut + 0.5 * dn);
          return {true, dn, n, contact_position};
        }
      }
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
