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
  exanb::Vec3d vel = exanb::Vec3d{0, 0, 0};                    /**< Velocity of the ball. */
  exanb::Vec3d vrot = exanb::Vec3d{0, 0, 0};                   /**< Angular velocity of the ball. */
  exanb::Vec3d forces = {0, 0, 0}; /**< sum of the forces applied to the driver. */
  double mass = std::numeric_limits<double>::max() / 4; /**< Mass of the ball */
  double rv = 0;                                        /**< */

  /** We don't need to save these values */
  exanb::Vec3d acc = {0, 0, 0}; /**< Acceleration of the ball. */
  double ra = 0;                /**< */
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
    v.radius = node["radius"].as<Quantity>().convert();
    v.center = node["center"].as<exanb::Vec3d>();
    if (check(node, "vel")) {
      v.vel = node["vel"].as<exanb::Vec3d>();
    }
    if (check(node, "vrot")) {
      v.vrot = node["vrot"].as<exanb::Vec3d>();
    }
    if (check(node, "rv")) {
      v.rv = node["rv"].as<Quantity>().convert();
    }
    if (check(node, "mass")) {
      v.mass = node["mass"].as<Quantity>().convert();
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
  MotionType motion_type;

  Ball(BallFields& bp, MotionType& mt) : fields(bp), motion_type(mt) {}

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
    if (is_compressive(motion_type)) {
      exanb::lout << "Radius acceleration: " << fields.ra << std::endl;
      exanb::lout << "Radius velocity: " << fields.rv << std::endl;
    }
    if (is_force_motion(motion_type)) {
      exanb::lout << "Mass: " << fields.mass << std::endl;
    }
  }

  /**
   * @brief Write ball data into a stream.
   */
  void dump_driver(const Driver_params& motion, int id, std::stringstream& stream) {
    stream << "  - register_ball:" << std::endl;
    stream << "     id: " << id << std::endl;
    stream << "     state: { radius:" << fields.radius;
    stream << ",center: [" << fields.center << "]";
    stream << ",vel: [" << fields.vel << "]";
    stream << ",vrot: [" << fields.vrot << "]";
    if (is_compressive(motion_type)) {
      stream << ",rv: " << fields.rv;
    }
    if (is_force_motion(motion_type)) {
      stream << ",mass: " << fields.mass;
    }
    stream << "}" << std::endl;
    motion.dump_driver_params(motion_type, stream);
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

    if (!is_valid_motion_type(motion_type, ball_valid_motion_types)) {
      color_log::error("Ball::initialize", "Invalid Motion Type.");
    } else if (!motion.check_motion_coherence(motion_type)) {
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
  inline void force_to_accel(const Driver_params& motion) {
    if (is_force_motion(motion_type)) {
      if (fields.mass >= 1e100) {
        color_log::warning("f_to_a", "The mass of the ball is set to " + std::to_string(fields.mass));
      }
      motion.update_forces(motion_type, fields.forces);  // update forces in function of the motion type
      fields.acc = fields.forces / fields.mass;
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
  inline void push_f_v_r(const Driver_params& motion, const double time, const double dt) {
    if (is_tabulated(motion_type)) {
      fields.center = motion.tab_to_position(time);
      fields.vel = motion.tab_to_velocity(time);
    } else if (!is_stationary(motion_type)) {
      if (is_compressive(motion_type)) {
        push_ra_rv_to_rad(dt);
      }
      fields.center = fields.center + dt * fields.vel;
    }
  }

  /**
   * @brief Update the position of the ball.
   * @param dt The time step.
   */
  inline void push_f_v(const Driver_params& motion, const double dt) {
    if (is_force_motion(motion_type)) {
      fields.vel = fields.acc * dt;
    }

    if (motion_type == LINEAR_MOTION) {
      fields.vel = motion.motion_vector * motion.const_vel;  // I prefere reset it
    }

    if (is_compressive(motion_type)) {
      if (motion_type == COMPRESSIVE_FORCE) {
        fields.vel = {0, 0, 0};
      }
      push_ra_to_rv(motion, dt);
    }
  }
  /**
   * @brief Update the "velocity raduis" of the ball.
   * @param t The time step.
   */
  inline void push_ra_to_rv(const Driver_params& motion, const double dt) {
    if (is_compressive(motion_type)) {
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
    if (is_compressive(motion_type)) {
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
  inline void f_ra(const Driver_params& motion, const double dt) {
    if (is_compressive(motion_type)) {
      constexpr double C = 0.5;  // I don't remember why, ask Lhassan
      if (motion.weigth != 0) {
        const double s = surface();
        // forces and weigth are defined in Driver_params
        fields.ra = (exanb::norm(fields.forces) - motion.sigma * s - (motion.damprate * fields.rv)) / (motion.weigth * C);
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
        exanb::Vec3d point_to_center = fields.center - p;
        double d = norm(point_to_center);
        double dn;
        exanb::Vec3d n = point_to_center / d;
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
