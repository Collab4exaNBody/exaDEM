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
#include <exaDEM/color_log.hpp>
#include <exaDEM/driver_base.hpp>
#include <onika/physics/units.h>

namespace exaDEM {
struct SurfaceFields {
  /** Required */
  double offset = 0;               /**< Offset from the origin along the normal vector. */
  exanb::Vec3d normal = {0, 0, 1}; /**< Normal vector of the surface. */
  exanb::Vec3d center = {0, 0, 0}; /**< Center position of the surface. */
  /** optional */
  exanb::Vec3d vel = {0, 0, 0};                                /**< Velocity of the surface. */
  exanb::Vec3d vrot = exanb::Vec3d{0, 0, 0};                   /**< Angular velocity of the surface. */
  exanb::Vec3d forces = {0, 0, 0}; /**< sum of the forces applied to the driver. */
  double mass = std::numeric_limits<double>::max() / 4; /**< Mass of the ball */
  double surface = -1;
  /** no need to dump them */
  exanb::Vec3d center_proj; /**< Center position projected on the norm. */
  double acc = 0;
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
    v.offset = node["offset"].as<Quantity>().convert();
    v.normal = node["normal"].as<exanb::Vec3d>();
    if (check(node, "vel")) {
      v.vel = node["vel"].as<exanb::Vec3d>();
    }
    if (check(node, "vrot")) {
      v.vrot = node["vrot"].as<exanb::Vec3d>();
    }
    if (check(node, "mass")) {
      v.mass = node["mass"].as<Quantity>().convert();
    }
    if (check(node, "surface")) {
      v.surface = node["surface"].as<Quantity>().convert();
    }
    if (v.vrot != exanb::Vec3d{0, 0, 0}) {
      if (!check_error(node, "center")) return false;
      v.center = node["center"].as<exanb::Vec3d>();
    } else {
      if (check(node, "center")) {
        v.center = node["center"].as<exanb::Vec3d>();
      } else {
        v.center = v.offset * v.normal;
      }
    }
    return true;
  }
};
}  // namespace YAML

namespace exaDEM {
/**
 * @brief Struct representing a surface in the exaDEM simulation.
 */
struct Surface {
  SurfaceFields fields;
  MotionType motion_type;
  /**
   * @brief Get the type of the driver (in this case, SURFACE).
   * @return The type of the driver.
   */
  constexpr DRIVER_TYPE get_type() {
    return DRIVER_TYPE::SURFACE;
  }

  /**
   * @brief Print information about the surface.
   */
  inline void print() const {
    exanb::lout << "Driver Type: Surface" << std::endl;
    exanb::lout << "Offset: " << fields.offset << std::endl;
    exanb::lout << "Normal: " << fields.normal << std::endl;
    exanb::lout << "Center: " << fields.center << std::endl;
    exanb::lout << "Vel   : " << fields.vel << std::endl;
    exanb::lout << "AngVel: " << fields.vrot << std::endl;
    if (is_compressive(motion_type)) {
      exanb::lout << "Acceleration: " << fields.acc << std::endl;
      exanb::lout << "Surface Value [>0]: " << fields.surface << std::endl;
    }
  }

  /**
   * @brief Write surface data into a stream.
   */
  inline void dump_driver(const Driver_params& motion, int id, std::stringstream& stream) {
    stream << "  - register_surface:" << std::endl;
    stream << "     id: " << id << std::endl;
    stream << "     state: {offset: " << fields.offset;
    stream << ", center: [" << fields.center << "]";
    stream << ", normal: [" << fields.normal << "]";
    stream << ", vel: [" << fields.vel << "]";
    stream << ", vrot: [" << fields.vrot << "]";
    stream << ", surface: " << fields.surface;
    stream << "}" << std::endl;
    motion.dump_driver_params(motion_type, stream);
  }

  /**
   * @brief Initialize the surface.
   * @details Calculates the center position based on the normal and offset.
   */
  inline void initialize(Driver_params& motion) {
    const std::vector<MotionType> surface_valid_motion_types = {
      STATIONARY,
      LINEAR_MOTION,
      LINEAR_COMPRESSIVE_MOTION,
      SHAKER,
      PENDULUM_MOTION};

    fields.center_proj = fields.offset * fields.normal;

    if (motion_type == PENDULUM_MOTION) {
      if (fields.center != motion.pendulum_anchor_point) {
        color_log::warning("register_surface", "Surface center should be equal to pendulum_anchor_point");
        color_log::warning("register_surface",
                           "Surface center is update to [" + std::to_string(motion.pendulum_anchor_point) + "]");
        fields.center = motion.pendulum_anchor_point;
      }
      exanb::Vec3d axis = motion.pendulum_anchor_point - motion.pendulum_initial_position;
      if (exanb::dot(axis, motion.pendulum_swing_dir) >= 1e-14) {
        color_log::error("register_surface",
                         "The vector from pendulum_anchor_point to pendulum_initial_position must be orthogonal to "
                         "pendulum_swing_dir.");
      }
    }

    // checks
    if (exanb::dot(fields.center - fields.center_proj, fields.normal) >= 1e-14) {
      color_log::warning("register_surface", "The Center point (surface) is not correctly defined");
      fields.center = exanb::dot(fields.center_proj - fields.center, fields.normal) * fields.normal + fields.center_proj;
      color_log::warning("register_surface",
                         "center is re-computed because it doesn't fit with offset, new center is: [" +
                         std::to_string(fields.center) + "] and center_proj is: [" + std::to_string(fields.center_proj) + "]");
    }

    if (!is_valid_motion_type(motion_type, surface_valid_motion_types)) {
      color_log::error("register_surface", "Invalid Motion Type.");
    } else if (!motion.check_motion_coherence(motion_type)) {
      color_log::error("register_surface", "Invalid Coherency [Motion Type].");
    } else if (fields.mass <= 0.0) {
      color_log::error("register_surface", "Please, define a positive mass.");
    }
    if (is_linear(motion_type)) {
      // We do not accept that motion_vector is not equal to -normal for compression mode
      if (fields.normal != motion.motion_vector && (fields.normal != -motion.motion_vector && !is_compressive(motion_type))) {
        color_log::warning("register_surface",
                           "The motion vector of the surface has been adjusted to align with the normal vector, i.e. "
                           "the motion vecor[" +
                           std::to_string(motion.motion_vector) + "] is now equal to [" + std::to_string(fields.normal) + "].");
        motion.motion_vector = fields.normal;
      }
    }

    if (is_compressive(motion_type)) {
      if (fields.surface <= 0) {
        color_log::error("register_surface",
                         "The surface value must be positive for LINEAR_COMPRESSIVE_FORCE. You need to specify "
                         "surface: XX in the 'state' slot.");
      }
    }
  }

  void force_to_accel(const Driver_params& motion) {
    if (is_compressive(motion_type)) {
      constexpr double C = 0.5;
      if (motion.weigth != 0) {
        const double s = fields.surface;
        // acc = (exanb::norm(forces) - sigma * s - (damprate * exanb::norm(vel)) ) / (weigth * C);
        exanb::Vec3d tmp = (fields.forces - motion.sigma * s * motion.motion_vector) / (motion.weigth * C);
        // get acc into the motion vector axis
        fields.acc = exanb::dot(tmp, motion.motion_vector);
      } else {
        fields.acc = 0;
      }
    }
  }

  inline void push_f_v(const Driver_params& motion, const double dt) {
    if (is_stationary(motion_type)) {
      fields.vel = {0, 0, 0};
    } else {
      if (is_compressive(motion_type)) {
        if (motion.sigma != 0) {
          fields.vel += 0.5 * dt * fields.acc * motion.motion_vector;
        }
      }
      if (motion_type == LINEAR_MOTION) {
        fields.vel = motion.const_vel * motion.motion_vector;  // I prefere reset it
      }
    }
  }

  /**
   * Fields Getters
   */
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& position() { return fields.center; }
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& velocity() { return fields.vel; }
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& forces() { return fields.forces; }
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& angular_velocity() { return fields.vrot; }

  /**
   * @brief Update the position of the wall.
   * @param time Current physical time.
   * @param dt The time step.
   */
  inline void push_f_v_r(const Driver_params& motion, const double time, const double dt) {
    if (!is_stationary(motion_type)) {
      if (motion_type == LINEAR_MOTION) {
        assert(motion.vel == motion.const_vel * motion.motion_vector);
      }

      if (motion_type == PENDULUM_MOTION) {
        auto [Offset, Normal] = motion.compute_offset_normal_pendulum_motion(time + dt);
        fields.normal = Normal;
        fields.offset = Offset;
        fields.vel = motion.pendulum_velocity(time + dt);
        fields.center_proj = fields.normal * fields.offset;
        return;
      }

      double displ = dt * exanb::dot(fields.vel, fields.normal) + 0.5 * dt * dt * fields.acc;

      /** The shaker motion changes the displacement behavior */
      /** the shaker direction vector is ignored, the normal vector is used */
      if (motion_type == SHAKER) {
        double signal_next = motion.shaker_signal(time + dt);
        double signal_current = motion.shaker_signal(time);
        const double angle_factor = exanb::dot(motion.shaker_direction(), fields.normal);
        displ = (signal_next - signal_current) * angle_factor;
        fields.vel = motion.shaker_velocity(time + dt);
      }

      fields.center += displ * fields.normal;
      fields.offset += displ;
      fields.center_proj += displ * fields.normal;
    }
  }

  /**
   * @brief Filter function to check if a vertex is within a certain radius of the surface.
   * @param rcut The cut-off radius.
   * @param p The point to check.
   * @return True if the point is within the cut-off radius of the surface, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool filter(const double rcut, const exanb::Vec3d& p) {
    exanb::Vec3d proj = dot(p, fields.normal) * fields.normal;
    double d = norm(proj - fields.center_proj);
    return d <= rcut;
  }

  /**
   * @brief Detects collision between a vertex and the surface.
   * @param rcut The cut-off radius.
   * @param p The point to check for collision.
   * @return A tuple containing:
   *         - A boolean indicating whether a collision occurred.
   *         - The penetration depth (negative if inside the surface).
   *         - The normal vector pointing from the collision point to the surface.
   *         - The contact position on the surface.
   */
  ONIKA_HOST_DEVICE_FUNC
      inline std::tuple<bool, double, exanb::Vec3d, exanb::Vec3d> detector(const double rcut, const exanb::Vec3d& p) {
        exanb::Vec3d proj = dot(p, fields.normal) * fields.normal;
        exanb::Vec3d surface_to_point = -(fields.center_proj - proj);
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
}  // namespace exaDEM

namespace onika {
namespace memory {
template <>
struct MemoryUsage<exaDEM::Surface> {
  static inline size_t memory_bytes(const exaDEM::Surface& obj) {
    return onika::memory::memory_bytes(obj);
  }
};
}  // namespace memory
}  // namespace onika
