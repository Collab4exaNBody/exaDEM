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
#include <onika/math/quaternion.h>
#include <onika/math/quaternion_yaml.h>
#include <onika/physics/units.h>

#include <exaDEM/color_log.hpp>
#include <exaDEM/driver_numerical_scheme_kernel.hpp>
// plugin shape
#include <exaDEM/rshape_grid.hpp>
#include <exaDEM/shape.hpp>
#include <exaDEM/shape_reader.hpp>
#include <exaDEM/shape_writer.hpp>
#include <filesystem>

namespace exaDEM {
// Data fields for a rigid shape driver (polyhedron or complex geometry)
struct RShapeDriverFields {
  exanb::Vec3d center = exanb::Vec3d{0, 0, 0}; /**< Center position of the shape. */
  exanb::Vec3d vel = exanb::Vec3d{0, 0, 0};    /**< Linear velocity of the shape. */
  exanb::Vec3d vrot = exanb::Vec3d{0, 0, 0};   /**< Angular velocity (rotation) of the shape. */
  exanb::Vec3d forces = exanb::Vec3d{0, 0, 0}; /**< Accumulated forces applied to the shape from interactions. */
  exanb::Quaternion quat = {1, 0, 0, 0};       /**< Orientation quaternion of the shape. */
  exanb::Vec3d acc = {0, 0, 0};                /**< Linear acceleration of the shape. */
  double surface = -1;                         /**< Contact surface area (for compressive motion). */
  double mass = std::numeric_limits<double>::max() / 4; /**< Mass of the shape. */
  // Moment (torque) control mode
  bool drive_by_mom = false;                    /**< Whether shape is driven by applied moment. */
  exanb::Vec3d applied_mom;                     /**< Applied moment input (for moment-driven motion). */
  exanb::Vec3d mom;                             /**< Accumulated moment from interactions. */
  exanb::Vec3d inertia = exanb::Vec3d{0, 0, 0}; /**< Moment of inertia tensor components. */
  exanb::Vec3d mom_axis;                        /**< Normalized axis along which moment is applied. */
};
}  // namespace exaDEM

namespace YAML {
template <>
struct convert<exaDEM::RShapeDriverFields> {
  static bool decode(const Node& node, exaDEM::RShapeDriverFields& v) {
    if (!node.IsMap()) {
      return false;
    }
    if (check(node, "center")) {
      v.center = node["center"].as<exanb::Vec3d>();
    }
    if (check(node, "vel")) {
      v.vel = node["vel"].as<exanb::Vec3d>();
    }
    if (check(node, "vrot")) {
      v.vrot = node["vrot"].as<exanb::Vec3d>();
    }
    if (check(node, "mass")) {
      v.mass = node["mass"].as<Quantity>().convert();
    }
    if (check(node, "quat")) {
      v.quat = node["quat"].as<exanb::Quaternion>();
    }
    if (check(node, "surface")) {
      v.surface = node["surface"].as<Quantity>().convert();
    }
    if (check(node, "moment")) {
      v.applied_mom = node["moment"].as<exanb::Vec3d>();
      v.drive_by_mom = true;
      double mom_norm = exanb::norm(v.applied_mom);
      if (mom_norm > 1e-12) {
        v.mom_axis = v.applied_mom / mom_norm;
      } else {
        v.mom_axis = exanb::Vec3d{0, 0, 0};
      }
    }
    if (check(node, "inertia")) {
      v.inertia = node["inertia"].as<exanb::Vec3d>();
    }
    v.mom = exanb::Vec3d{0, 0, 0};
    return true;
  }
};
}  // namespace YAML

namespace exaDEM {

/**
 * @brief Rigid shape (polyhedron/complex geometry) driver for DEM simulations.
 *
 * Represents a complex rigid body defined by a set of vertices, edges, and faces.
 * Supports various motion types and can be driven by forces or moments.
 * Orientation is tracked using quaternions for rotation handling.
 */
struct RShapeDriver {
  RShapeDriverFields fields; /**< Contains specific driver parameters */
  MotionType motion_type;    /**< Contains motion type parameters */
  shape shp;                 /**< Shape of the R-Shape. */
  onika::memory::CudaMMVector<exanb::Vec3d>
      vertices;                             /**< Collection of vertices (computed from shp, quat, and center). */
  RShapeDriverGridCellIndexes grid_indexes; /**< Grid indexes for vertices, edges, and faces of the shape. */

  /**
   * @brief Get the type of the driver (in this case, RSHAPE).
   * @return The type of the driver.
   */
  constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::RSHAPE; }

  /** @brief Print information about the R-Shape.
   */
  inline void print() const {
    exanb::lout << "Driver Type: R-Shape" << std::endl;
    exanb::lout << "Name               = " << shp.m_name << std::endl;
    exanb::lout << "Center             = " << fields.center << std::endl;
    exanb::lout << "Minkowski         = " << shp.minkowski() << std::endl;
    exanb::lout << "Velocity           = " << fields.vel << std::endl;
    exanb::lout << "Angular Velocity   = " << fields.vrot << std::endl;
    exanb::lout << "Orientation        = " << fields.quat.w << " " << fields.quat.x << " " << fields.quat.y << " "
                << fields.quat.z << std::endl;
    if (fields.surface > 0.0) {
      exanb::lout << "Surface            = " << fields.surface << std::endl;
    }
    if (fields.drive_by_mom) {
      exanb::lout << "Applied moment     = " << fields.applied_mom << std::endl;
      exanb::lout << "Inertia            = " << fields.inertia << std::endl;
      exanb::lout << "normal(moment)     = " << fields.mom_axis << std::endl;
    } else if (motion_type == PARTICLE) {
      exanb::lout << "Mass               = " << fields.mass << std::endl;
      exanb::lout << "Inertia            = " << fields.inertia << std::endl;
    }
    exanb::lout << "Number of faces    = " << shp.get_number_of_faces() << std::endl;
    exanb::lout << "Number of edges    = " << shp.get_number_of_edges() << std::endl;
    exanb::lout << "Number of vertices = " << shp.get_number_of_vertices() << std::endl;
  }

  /**
   * @brief Print information about the R-Shape.
   */
  inline void initialize(Driver_params& motion) {
    const std::vector<MotionType> rshape_valid_motion_types = {
        STATIONARY, LINEAR_MOTION, LINEAR_FORCE_MOTION, LINEAR_COMPRESSIVE_MOTION, PARTICLE,
        TABULATED,  SHAKER,        EXPRESSION};
    // checks
    if (shp.get_number_of_faces() == 0 && shp.get_number_of_edges() == 0 && shp.get_number_of_vertices() == 0) {
      color_log::error("RShape::initialize",
                       "Your shape is not correctly defined, no vertex, no "
                       "edge, and no face.");
    }

    // resize and initialize vertices
    vertices.resize(shp.get_number_of_vertices());
    // #   pragma omp parallel for schedule(static)
    for (int i = 0; i < shp.get_number_of_vertices(); i++) {
      this->update_vertex(i);
    }

    // remove relative paths
    std::filesystem::path full_name = this->shp.m_name;
    this->shp.m_name = full_name.filename();
    // motion type
    if (!is_valid_motion_type(motion_type, rshape_valid_motion_types)) {
      std::exit(EXIT_FAILURE);
    } else if (!motion.check_motion_coherence(motion_type)) {
      std::exit(EXIT_FAILURE);
    } else if (fields.mass <= 0.0) {
      color_log::error("RShape::initialize", "Please, define a positive mass.");
    }

    if (is_compressive(motion_type)) {
      double s = shp.compute_surface();
      if (fields.surface <= 0) {
        color_log::warning("RShape::initialize",
                           "The surface value must be positive for LINEAR_COMPRESSIVE_FORCE. "
                           "You need to specify surface: XX in the 'state' slot.");
        color_log::error("RShape::initialize",
                         "The surface value must be positive for LINEAR_COMPRESSIVE_FORCE. "
                         "You need to specify surface: XX in the 'state' slot.",
                         false);
        color_log::error("RShape::initialize", "The computed surface of all faces is: " + std::to_string(s), true);
      }
      if (s - fields.surface > 1e-6) {
        color_log::warning("RShape::initialize", "The computed surface of all faces is: " + std::to_string(s));
      }
    }

    bool special_case = motion.expr.expr_use_mom;
    if (need_moment() || special_case) {
      if (fields.inertia == exanb::Vec3d{0, 0, 0}) {
        color_log::error("RShape::initialize", "Inertia should be defined, either params, either in a shape file.");
      }
    }
  }

  /**
   * @brief Convert accumulated forces to acceleration.
   * @details Updates acceleration based on forces and motion type.
   *          Handles compressive motion (with surface and damping) and force-driven motion.
   * @param motion Driver motion parameters and constraints.
   * @pre fields.forces must be populated from interaction calculations.
   * @pre For compressive motion, fields.surface must be positive.
   * @pre For force motion, fields.mass must be positive and finite.
   * @post fields.acc is updated based on motion type (compressive, force-driven, or stationary).
   * @post For compressive motion, acceleration is constrained to motion_vector direction.
   * @invariant If mass is near infinite (1e100+), warning is logged but computation proceeds.
   */
  inline void force_to_accel(const Driver_params& motion) {
    if (is_compressive(motion_type)) {
      // Compressive motion: only update acceleration when the particle has a non-zero mass.
      // (Zero mass would cause a division by zero — silently skipped here.)
      constexpr double C = 0.5;  // integration coefficient (Velocity Verlet half-step factor)

      if (fields.mass != 0) {
        const double surface = fields.surface;

        // Force balance along the compression axis:
        exanb::Vec3d tmp =
            (fields.forces - motion.sigma * surface - (motion.damprate * fields.vel)) / (fields.mass * C);

        // Project acceleration onto the motion axis: only the component along
        // motion_vector is physically meaningful for a uniaxial compression.
        fields.acc = exanb::dot(tmp, motion.motion_vector) * motion.motion_vector;
      }

    } else if (is_force_motion(motion_type)) {
      // Force-driven motion: warn if mass looks uninitialised (sentinel value ≥ 1e100).
      // A division by such a value would yield near-zero acceleration silently.
      if (fields.mass >= 1e100) {
        color_log::warning("f_to_a", "The mass of the rshape is set to " + std::to_string(fields.mass));
      } else if (fields.mass <= 0) {
        color_log::error("rshape::force_to_accel",
                         "The mass of the rshape is not defined correctly " + std::to_string(fields.mass));
      }

      // Apply any motion-type-specific force corrections before computing acceleration.
      motion.update_forces(motion_type, fields.forces);

      // Newton's second law: a = F / m
      fields.acc = fields.forces / fields.mass;

    } else {
      // Stationary, linear, shaker, etc.: kinematics are prescribed directly,
      // so acceleration from forces is irrelevant — reset to zero.
      fields.acc = {0, 0, 0};
    }
  }

  /**
   * @brief Update linear velocity based on acceleration.
   * @details Integrates acceleration to velocity, applies motion constraints
   *          (stationary, linear motion, compressive).
   * @param motion Driver motion parameters.
   * @param dt Time step.
   */
  inline void push_f_v(const Driver_params& motion, const double dt) {
    if (is_stationary(motion_type)) {
      fields.vel = exanb::Vec3d{0, 0, 0};
    } else {
      if (is_force_motion(motion_type)) {
        fields.vel += fields.acc * dt;
      }

      if (motion_type == MotionType::LINEAR_MOTION) {
        fields.vel = motion.motion_vector * motion.const_vel;
      }

      if (is_compressive(motion_type)) {
        if (motion.sigma != 0) {
          fields.vel += 0.5 * dt * fields.acc;
        }
      }
    }
  }

  /**
   * @brief Update position and velocity using kinematic integration.
   * @details Handles different motion types: tabulated, linear, shaker, and expression-based.
   * @param motion Driver motion parameters.
   * @param time Current simulation time.
   * @param dt Time step.
   */
  inline void push_f_v_r(const Driver_params& motion, const double time, const double dt) {
    if (is_tabulated(motion_type)) {
      // Tabulated motion: bypass integration entirely and interpolate position
      // and velocity directly from the pre-computed trajectory table.
      fields.center = motion.tab_to_position(time);
      fields.vel = motion.tab_to_velocity(time);
    } else if (!is_stationary(motion_type)) {
      // Only integrate non-stationary particles.
      if (motion_type == MotionType::LINEAR_MOTION) {
        // Sanity check: for purely linear motion the speed must remain constant.
        [[maybe_unused]] constexpr double VEL_TOLERANCE = 1e-12;
        assert(std::abs(exanb::norm(fields.vel) - motion.const_vel) < VEL_TOLERANCE);
      }

      if (motion_type == MotionType::SHAKER) {
        // Shaker motion: recompute velocity from the waveform at (time + dt)
        // and project it onto the shaker direction.
        // Acceleration is reset to zero because the shaker imposes kinematics directly.
        fields.vel = motion.shaker_velocity(time + dt) * motion.shaker_direction();
        fields.acc = exanb::Vec3d{0, 0, 0};  // reset acc
      }

      if (motion.is_expr(motion_type, time)) {
        if (motion.expr.expr_use_v) {
          fields.vel = motion.driver_expr_v(time);
        }
        if (motion.expr.expr_use_vrot) {
          fields.vrot = motion.driver_expr_vrot(time);
        }
      }

      // Standard Velocity Verlet position update:
      // Note: velocity may have been overridden above (shaker, expression).
      fields.center += dt * fields.vel + 0.5 * dt * dt * fields.acc;
    }
  }

  /** @brief Update angular velocity and orientation based on accumulated moments.
   * Computes angular acceleration, updates angular velocity, and integrates to orientation quaternion
   * @param motion Driver motion parameters.
   * @param time Current simulation time.
   * @param dt Time step.
   */
  inline void push_av_to_quat(const Driver_params& motion, double time, double dt) {
    if (need_moment()) {
      DriverPushToAngularAccelerationFunctor compute_arot = {};
      DriverPushToAngularVelocityFunctor compute_vrot = {dt * 0.5};
      DriverPushToQuaternionFunctor compute_quat_vrot = {dt, dt * 0.5, dt * dt * 0.5};

      if (motion.is_expr(motion_type, time)) {
        if (motion.expr.expr_use_mom) {
          fields.applied_mom = motion.driver_expr_mom(time);
          double mom_norm = exanb::norm(fields.applied_mom);
          if (mom_norm > 1e-12) {
            fields.mom_axis = fields.applied_mom / mom_norm;
          } else {
            fields.mom_axis = exanb::Vec3d{0, 0, 0};
          }
        }
      }

      exanb::Vec3d project_mom;
      exanb::Vec3d arot;

      // do not use applied_mom direction if the motion type is PARTICLE
      if (motion_type == MotionType::PARTICLE) {
        project_mom = fields.mom;
      } else {
        project_mom = dot(fields.applied_mom + fields.mom, fields.mom_axis) * fields.mom_axis;
      }

      compute_arot(fields.quat, project_mom, fields.vrot, arot, fields.inertia);
      compute_vrot(fields.vrot, arot);
      compute_quat_vrot(fields.quat, fields.vrot, arot);
      fields.mom = {0, 0, 0};
    } else {
      fields.quat = fields.quat + dot(fields.quat, fields.vrot) * dt;
      fields.quat = normalize(fields.quat);
    }
    exanb::ldbg << "Quat[rshape]: " << fields.quat.w << " " << fields.quat.x << " " << fields.quat.y << " "
                << fields.quat.z << std::endl;
  }

  /** @brief Update the position of a vertex based on the shape geometry and orientation.
   * @param i The index of the vertex to update.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline void update_vertex(int i) {
    // homothety = 1.0
    vertices[i] = shp.get_vertex(i, fields.center, 1.0, fields.quat);
  }

  /**
   * @brief Field accessors for position, velocity, forces, moments, and orientation.
   */
  // Position getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& position() { return fields.center; }
  // Linear velocity getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& velocity() { return fields.vel; }
  // Accumulated forces getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& forces() { return fields.forces; }
  // Accumulated moment (torque) getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& moment() { return fields.mom; }
  // Angular velocity getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Vec3d& angular_velocity() { return fields.vrot; }
  // Orientation (quaternion) getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Quaternion& orientation() { return fields.quat; }

  /**
   * @brief Check if moment-based rotation is enabled for this shape.
   * @return true if shape is driven by applied moment, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool need_moment() const {
    if (fields.drive_by_mom) {
      return true;
    } else if (motion_type == MotionType::PARTICLE) {
      return true;
    }
    return false;
  }

  /**
   * @brief Check if the shape is completely stationary (no linear or angular motion).
   * @return true if motion_type is STATIONARY and angular velocity is zero.
   */
  inline bool stationary() const { return is_stationary(motion_type) && (fields.vrot == exanb::Vec3d{0, 0, 0}); }

  /** @brief Dump driver information to a stream.
   * @param motion The motion parameters of the driver to include in the dump.
   * @param id The identifier of the driver (for labeling in the output).
   * @param path The directory path where the shape file should be written.
   * @param stream The output stream to write the YAML information to.
   */
  void dump_driver(const Driver_params& motion, int id, std::string path, std::stringstream& stream) {
    std::string filename = path + shp.m_name + ".shp";
    stream << "  - register_rshape:" << std::endl;
    stream << "     id: " << id << std::endl;
    stream << "     filename: " << filename << std::endl;
    stream << "     minkowski: " << shp.m_radius << std::endl;
    stream << "     state: {";
    stream << "center: [" << fields.center << "]";
    stream << ", vel: [" << fields.vel << "]";
    stream << ", vrot: [" << fields.vrot << "]";
    if (fields.surface > 0) {
      stream << ", surface: " << fields.surface;
    }
    if (fields.drive_by_mom) {
      stream << ", moment: " << fields.applied_mom << ", inertia: " << fields.inertia;
    }
    stream << ", quat: [" << fields.quat.w << "," << fields.quat.x << "," << fields.quat.y << "," << fields.quat.z
           << "]";
    if (is_force_motion(motion_type)) {
      stream << ",mass: " << fields.mass;
    }
    stream << "}" << std::endl;
    motion.dump_driver_params(motion_type, stream);
    write_shp(shp, filename);
  }
};

template <>
struct DriverProperty<RShapeDriver> {
  static constexpr bool use_moment = true;
  static constexpr bool use_quaternion = true;
};
}  // namespace exaDEM

namespace onika {
namespace memory {
template <>
struct MemoryUsage<exaDEM::RShapeDriver> {
  static inline size_t memory_bytes(const exaDEM::RShapeDriver& obj) { return onika::memory::memory_bytes(&obj); }
};
}  // namespace memory
}  // namespace onika