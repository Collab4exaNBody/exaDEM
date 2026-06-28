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
// Data fields_ for a rigid shape driver (polyhedron or complex geometry)
struct RShapeDriverFields {
  exanb::Vec3d center_ = exanb::Vec3d{0, 0, 0}; /**< Center position of the shape. */
  exanb::Vec3d vel_ = exanb::Vec3d{0, 0, 0};    /**< Linear velocity of the shape. */
  exanb::Vec3d vrot_ = exanb::Vec3d{0, 0, 0};   /**< Angular velocity (rotation) of the shape. */
  exanb::Vec3d forces_ = exanb::Vec3d{0, 0, 0}; /**< Accumulated forces applied to the shape from interactions. */
  exanb::Quaternion quat_ = {1, 0, 0, 0};       /**< Orientation quaternion of the shape. */
  exanb::Vec3d acc_ = {0, 0, 0};                /**< Linear acceleration of the shape. */
  double surface_ = -1;                         /**< Contact surface area (for compressive motion). */
  double mass_ = std::numeric_limits<double>::max() / 4; /**< Mass of the shape. */
  // Moment (torque) control mode
  bool drive_by_mom_ = false;                    /**< Whether shape is driven by applied moment. */
  exanb::Vec3d applied_mom_;                     /**< Applied moment input (for moment-driven motion). */
  exanb::Vec3d mom_;                             /**< Accumulated moment from interactions. */
  exanb::Vec3d inertia_ = exanb::Vec3d{0, 0, 0}; /**< Moment of inertia tensor components. */
  exanb::Vec3d mom_axis_;                        /**< Normalized axis along which moment is applied. */
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
      v.center_ = node["center"].as<exanb::Vec3d>();
    }
    if (check(node, "vel")) {
      v.vel_ = node["vel"].as<exanb::Vec3d>();
    }
    if (check(node, "vrot")) {
      v.vrot_ = node["vrot"].as<exanb::Vec3d>();
    }
    if (check(node, "mass")) {
      v.mass_ = node["mass"].as<Quantity>().convert();
    }
    if (check(node, "quat")) {
      v.quat_ = node["quat"].as<exanb::Quaternion>();
    }
    if (check(node, "surface")) {
      v.surface_ = node["surface"].as<Quantity>().convert();
    }
    if (check(node, "moment")) {
      v.applied_mom_ = node["moment"].as<exanb::Vec3d>();
      v.drive_by_mom_ = true;
      double mom_norm = exanb::norm(v.applied_mom_);
      if (mom_norm > 1e-12) {
        v.mom_axis_ = v.applied_mom_ / mom_norm;
      } else {
        v.mom_axis_ = exanb::Vec3d{0, 0, 0};
      }
    }
    if (check(node, "inertia")) {
      v.inertia_ = node["inertia"].as<exanb::Vec3d>();
    }
    v.mom_ = exanb::Vec3d{0, 0, 0};
    return true;
  }
};
}  // namespace YAML

namespace exaDEM {

/**
 * @brief Rigid shape (polyhedron/complex geometry) driver for DEM simulations.
 *
 * Represents a complex rigid body defined by a set of vertices_, edges, and faces.
 * Supports various motion types and can be driven by forces or moments.
 * Orientation is tracked using quaternions for rotation handling.
 */
struct RShapeDriver {
  RShapeDriverFields fields_; /**< Contains specific driver parameters */
  MotionType motion_type_;    /**< Contains motion type parameters */
  shape shp_;                 /**< Shape of the R-Shape. */
  onika::memory::CudaMMVector<exanb::Vec3d>
      vertices_;                             /**< Collection of vertices_ (computed from shp_, quat, and center). */
  RShapeDriverGridCellIndexes grid_indexes_; /**< Grid indexes for vertices_, edges, and faces of the shape. */

  /**
   * @brief Get the type of the driver (in this case, RSHAPE).
   * @return The type of the driver.
   */
  constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::RSHAPE; }

  /** @brief Print information about the R-Shape.
   */
  inline void print() const {
    exanb::lout << "Driver Type: R-Shape" << std::endl;
    exanb::lout << "Name               = " << shp_.name_ << std::endl;
    exanb::lout << "Center             = " << fields_.center_ << std::endl;
    exanb::lout << "Minkowski         = " << shp_.minkowski() << std::endl;
    exanb::lout << "Velocity           = " << fields_.vel_ << std::endl;
    exanb::lout << "Angular Velocity   = " << fields_.vrot_ << std::endl;
    exanb::lout << "Orientation        = " << fields_.quat_.w << " " << fields_.quat_.x << " " << fields_.quat_.y << " "
                << fields_.quat_.z << std::endl;
    if (fields_.surface_ > 0.0) {
      exanb::lout << "Surface            = " << fields_.surface_ << std::endl;
    }
    if (fields_.drive_by_mom_) {
      exanb::lout << "Applied moment     = " << fields_.applied_mom_ << std::endl;
      exanb::lout << "Inertia            = " << fields_.inertia_ << std::endl;
      exanb::lout << "normal(moment)     = " << fields_.mom_axis_ << std::endl;
    } else if (motion_type_ == PARTICLE) {
      exanb::lout << "Mass               = " << fields_.mass_ << std::endl;
      exanb::lout << "Inertia            = " << fields_.inertia_ << std::endl;
    }
    exanb::lout << "Number of faces    = " << shp_.get_number_of_faces() << std::endl;
    exanb::lout << "Number of edges    = " << shp_.get_number_of_edges() << std::endl;
    exanb::lout << "Number of vertices_ = " << shp_.get_number_of_vertices() << std::endl;
  }

  /**
   * @brief Print information about the R-Shape.
   */
  inline void initialize(Driver_params& motion) {
    const std::vector<MotionType> rshape_valid_motion_types = {
        STATIONARY, LINEAR_MOTION, LINEAR_FORCE_MOTION, LINEAR_COMPRESSIVE_MOTION, PARTICLE,
        TABULATED,  SHAKER,        EXPRESSION};
    // checks
    if (shp_.get_number_of_faces() == 0 && shp_.get_number_of_edges() == 0 && shp_.get_number_of_vertices() == 0) {
      color_log::error("RShape::initialize",
                       "Your shape is not correctly defined, no vertex, no "
                       "edge, and no face.");
    }

    // resize and initialize vertices_
    vertices_.resize(shp_.get_number_of_vertices());
    // #   pragma omp parallel for schedule(static)
    for (int i = 0; i < shp_.get_number_of_vertices(); i++) {
      this->update_vertex(i);
    }

    // remove relative paths
    std::filesystem::path full_name = this->shp_.name_;
    this->shp_.name_ = full_name.filename();
    // motion type
    if (!is_valid_motion_type(motion_type_, rshape_valid_motion_types)) {
      std::exit(EXIT_FAILURE);
    } else if (!motion.check_motion_coherence(motion_type_)) {
      std::exit(EXIT_FAILURE);
    } else if (fields_.mass_ <= 0.0) {
      color_log::error("RShape::initialize", "Please, define a positive mass.");
    }

    if (is_compressive(motion_type_)) {
      double s = shp_.compute_surface();
      if (fields_.surface_ <= 0) {
        color_log::warning("RShape::initialize",
                           "The surface value must be positive for LINEAR_COMPRESSIVE_FORCE. "
                           "You need to specify surface: XX in the 'state' slot.");
        color_log::error("RShape::initialize",
                         "The surface value must be positive for LINEAR_COMPRESSIVE_FORCE. "
                         "You need to specify surface: XX in the 'state' slot.",
                         false);
        color_log::error("RShape::initialize", "The computed surface of all faces is: " + std::to_string(s), true);
      }
      if (s - fields_.surface_ > 1e-6) {
        color_log::warning("RShape::initialize", "The computed surface of all faces is: " + std::to_string(s));
      }
    }

    bool special_case = motion.expr_.expr_use_mom_;
    if (need_moment() || special_case) {
      if (fields_.inertia_ == exanb::Vec3d{0, 0, 0}) {
        color_log::error("RShape::initialize", "Inertia should be defined, either params, either in a shape file.");
      }
    }
  }

  /**
   * @brief Convert accumulated forces to acceleration.
   * @details Updates acceleration based on forces and motion type.
   *          Handles compressive motion (with surface and damping) and force-driven motion.
   * @param motion Driver motion parameters and constraints.
   * @pre fields_.forces_ must be populated from interaction calculations.
   * @pre For compressive motion, fields_.surface_ must be positive.
   * @pre For force motion, fields_.mass_ must be positive and finite.
   * @post fields_.acc_ is updated based on motion type (compressive, force-driven, or stationary).
   * @post For compressive motion, acceleration is constrained to motion_vector direction.
   * @invariant If mass is near infinite (1e100+), warning is logged but computation proceeds.
   */
  inline void force_to_accel(const Driver_params& motion) {
    if (is_compressive(motion_type_)) {
      // Compressive motion: only update acceleration when the particle has a non-zero mass.
      // (Zero mass would cause a division by zero — silently skipped here.)
      constexpr double C = 0.5;  // integration coefficient (Velocity Verlet half-step factor)

      if (fields_.mass_ != 0) {
        const double surface = fields_.surface_;

        // Force balance along the compression axis:
        exanb::Vec3d tmp =
            (fields_.forces_ - motion.sigma_ * surface - (motion.damprate_ * fields_.vel_)) / (fields_.mass_ * C);

        // Project acceleration onto the motion axis: only the component along
        // motion_vector is physically meaningful for a uniaxial compression.
        fields_.acc_ = exanb::dot(tmp, motion.motion_vector_) * motion.motion_vector_;
      }

    } else if (is_force_motion(motion_type_)) {
      // Force-driven motion: warn if mass looks uninitialised (sentinel value ≥ 1e100).
      // A division by such a value would yield near-zero acceleration silently.
      if (fields_.mass_ >= 1e100) {
        color_log::warning("f_to_a", "The mass of the rshape is set to " + std::to_string(fields_.mass_));
      } else if (fields_.mass_ <= 0) {
        color_log::error("rshape::force_to_accel",
                         "The mass of the rshape is not defined correctly " + std::to_string(fields_.mass_));
      }

      // Apply any motion-type-specific force corrections before computing acceleration.
      motion.update_forces(motion_type_, fields_.forces_);

      // Newton's second law: a = F / m
      fields_.acc_ = fields_.forces_ / fields_.mass_;

    } else {
      // Stationary, linear, shaker, etc.: kinematics are prescribed directly,
      // so acceleration from forces is irrelevant — reset to zero.
      fields_.acc_ = {0, 0, 0};
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
    if (is_stationary(motion_type_)) {
      fields_.vel_ = exanb::Vec3d{0, 0, 0};
    } else {
      if (is_force_motion(motion_type_)) {
        fields_.vel_ += fields_.acc_ * dt;
      }

      if (motion_type_ == MotionType::LINEAR_MOTION) {
        fields_.vel_ = motion.motion_vector_ * motion.const_vel_;
      }

      if (is_compressive(motion_type_)) {
        if (motion.sigma_ != 0) {
          fields_.vel_ += 0.5 * dt * fields_.acc_;
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
    if (is_tabulated(motion_type_)) {
      // Tabulated motion: bypass integration entirely and interpolate position
      // and velocity directly from the pre-computed trajectory table.
      fields_.center_ = motion.tab_to_position(time);
      fields_.vel_ = motion.tab_to_velocity(time);
    } else if (!is_stationary(motion_type_)) {
      // Only integrate non-stationary particles.
      if (motion_type_ == MotionType::LINEAR_MOTION) {
        // Sanity check: for purely linear motion the speed must remain constant.
        [[maybe_unused]] constexpr double VEL_TOLERANCE = 1e-12;
        assert(std::abs(exanb::norm(fields_.vel_) - motion.const_vel_) < VEL_TOLERANCE);
      }

      if (motion_type_ == MotionType::SHAKER) {
        // Shaker motion: recompute velocity from the waveform at (time + dt)
        // and project it onto the shaker direction.
        // Acceleration is reset to zero because the shaker imposes kinematics directly.
        fields_.vel_ = motion.shaker_velocity(time + dt) * motion.shaker_direction();
        fields_.acc_ = exanb::Vec3d{0, 0, 0};  // reset acc
      }

      if (motion.is_expr(motion_type_, time)) {
        if (motion.expr_.expr_use_v_) {
          fields_.vel_ = motion.driver_expr_v(time);
        }
        if (motion.expr_.expr_use_vrot_) {
          fields_.vrot_ = motion.driver_expr_vrot(time);
        }
      }

      // Standard Velocity Verlet position update:
      // Note: velocity may have been overridden above (shaker, expression).
      fields_.center_ += dt * fields_.vel_ + 0.5 * dt * dt * fields_.acc_;
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

      if (motion.is_expr(motion_type_, time)) {
        if (motion.expr_.expr_use_mom_) {
          fields_.applied_mom_ = motion.driver_expr_mom(time);
          double mom_norm = exanb::norm(fields_.applied_mom_);
          if (mom_norm > 1e-12) {
            fields_.mom_axis_ = fields_.applied_mom_ / mom_norm;
          } else {
            fields_.mom_axis_ = exanb::Vec3d{0, 0, 0};
          }
        }
      }

      exanb::Vec3d project_mom;
      exanb::Vec3d arot;

      // do not use applied_mom direction if the motion type is PARTICLE
      if (motion_type_ == MotionType::PARTICLE) {
        project_mom = fields_.mom_;
      } else {
        project_mom = dot(fields_.applied_mom_ + fields_.mom_, fields_.mom_axis_) * fields_.mom_axis_;
      }

      compute_arot(fields_.quat_, project_mom, fields_.vrot_, arot, fields_.inertia_);
      compute_vrot(fields_.vrot_, arot);
      compute_quat_vrot(fields_.quat_, fields_.vrot_, arot);
      fields_.mom_ = {0, 0, 0};
    } else {
      fields_.quat_ = fields_.quat_ + dot(fields_.quat_, fields_.vrot_) * dt;
      fields_.quat_ = normalize(fields_.quat_);
    }
    exanb::ldbg << "Quat[rshape]: " << fields_.quat_.w << " " << fields_.quat_.x << " " << fields_.quat_.y << " "
                << fields_.quat_.z << std::endl;
  }

  /** @brief Update the position of a vertex based on the shape geometry and orientation.
   * @param i The index of the vertex to update.
   */
  ONIKA_HOST_DEVICE_FUNC
  inline void update_vertex(int i) {
    // homothety = 1.0
    vertices_[i] = shp_.get_vertex(i, fields_.center_, 1.0, fields_.quat_);
  }

  /**
   * @brief Field accessors for position, velocity, forces, moments, and orientation.
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
  // Orientation (quaternion) getter
  ONIKA_HOST_DEVICE_FUNC inline exanb::Quaternion& orientation() { return fields_.quat_; }

  /**
   * @brief Check if moment-based rotation is enabled for this shape.
   * @return true if shape is driven by applied moment, false otherwise.
   */
  ONIKA_HOST_DEVICE_FUNC inline bool need_moment() const {
    if (fields_.drive_by_mom_) {
      return true;
    } else if (motion_type_ == MotionType::PARTICLE) {
      return true;
    }
    return false;
  }

  /**
   * @brief Check if the shape is completely stationary (no linear or angular motion).
   * @return true if motion_type_ is STATIONARY and angular velocity is zero.
   */
  inline bool stationary() const { return is_stationary(motion_type_) && (fields_.vrot_ == exanb::Vec3d{0, 0, 0}); }

  /** @brief Dump driver information to a stream.
   * @param motion The motion parameters of the driver to include in the dump.
   * @param id The identifier of the driver (for labeling in the output).
   * @param path The directory path where the shape file should be written.
   * @param stream The output stream to write the YAML information to.
   */
  void dump_driver(const Driver_params& motion, int id, std::string path, std::stringstream& stream) {
    std::string filename = path + shp_.name_ + ".shp";
    stream << "  - register_rshape:" << std::endl;
    stream << "     id: " << id << std::endl;
    stream << "     filename: " << filename << std::endl;
    stream << "     minkowski: " << shp_.minkowski() << std::endl;
    stream << "     state: {";
    stream << "center: [" << fields_.center_ << "]";
    stream << ", vel: [" << fields_.vel_ << "]";
    stream << ", vrot: [" << fields_.vrot_ << "]";
    if (fields_.surface_ > 0) {
      stream << ", surface: " << fields_.surface_;
    }
    if (fields_.drive_by_mom_) {
      stream << ", moment: " << fields_.applied_mom_ << ", inertia: " << fields_.inertia_;
    }
    stream << ", quat: [" << fields_.quat_.w << "," << fields_.quat_.x << "," << fields_.quat_.y << "," << fields_.quat_.z
           << "]";
    if (is_force_motion(motion_type_)) {
      stream << ",mass: " << fields_.mass_;
    }
    stream << "}" << std::endl;
    motion.dump_driver_params(motion_type_, stream);
    write_shp(shp_, filename);
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