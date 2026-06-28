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

#include <climits>
#include <chrono>
#include <thread>
#include <onika/physics/units.h>

#include <exaDEM/color_log.hpp>
#include <exaDEM/normalize.hpp>
#include <exaDEM/expr.hpp>
#include <exaDEM/motion_type.hpp>

namespace exaDEM {
struct Driver_params {
  // Common motion stuff
  exanb::Vec3d motion_vector_ = {0, 0, 0};
  double motion_start_threshold_ = 0;
  double motion_end_threshold_ = 1e300;

  // Motion: Linear
  double const_vel_ = 0;
  double const_force_ = 0;

  // Motion: Compression
  double sigma_ = 0;         /**< used for compressive force */
  double damprate_ = 0;      /**< used for compressive force */
  double mass_ = 0;          /**< mass_ of the driver */

  // Motion: Tabulated
  std::vector<double> tab_time_;
  std::vector<exanb::Vec3d> tab_pos_;

  // Motion: Shaker
  double omega_ = 0;
  double amplitude_ = 0;
  exanb::Vec3d shaker_dir_ = exanb::Vec3d(0, 0, 1);

  // Motion: Pendulum (re-use both shaker motion members omega_ and amplitude_)
  exanb::Vec3d pendulum_anchor_point_;     /**< Fixed suspension point. */
  exanb::Vec3d pendulum_initial_position_; /**< Starting position of the pendulum mass_. */
  exanb::Vec3d pendulum_swing_dir_;        /**< Direction defining the pendulum's oscillation plane. */

  // Motion: Expression
  Driver_expr expr_;

  // Just for YAML
  MotionType input_motion_type_ = MotionType::STATIONARY;

  ONIKA_HOST_DEVICE_FUNC
      inline bool is_expr(MotionType motion_type, double time) const {
        // do nothing if time < start or time > end;
        return motion_type == MotionType::EXPRESSION && is_motion_triggered(time);
      }

  ONIKA_HOST_DEVICE_FUNC
      inline void update_forces(MotionType motion_type, exanb::Vec3d& forces) const {
        if (motion_type == LINEAR_FORCE_MOTION) {
          forces = (exanb::dot(forces, motion_vector_) + const_force_) * motion_vector_;
        } else if (motion_type != MotionType::PARTICLE) {
          forces = exanb::Vec3d{0, 0, 0};
        }
      }


  ONIKA_HOST_DEVICE_FUNC bool is_motion_triggered(double time) const {
    return ((time >= motion_start_threshold_) && (time <= motion_end_threshold_));
  }

  ONIKA_HOST_DEVICE_FUNC bool is_motion_triggered(uint64_t timesteps, double dt) const {
    const double time = timesteps * dt;
    return is_motion_triggered(time);
  }

  void tabulations_to_stream(MotionType motion_type, std::stringstream& times, std::stringstream& positions) const {
    if (is_tabulated(motion_type)) {
      times << "time: [";
      positions << "positions: [";

      assert(tab_time_.size() == tab_pos_.size());
      size_t last = tab_time_.size() - 1;
      for (size_t i = 0; i < last; i++) {
        times << tab_time_[i] << ",";
        positions << "[ " << tab_pos_[i] << " ],";
      }
      times << tab_time_[last] << "]";
      positions << "[ " << tab_pos_[last] << " ]]";
    }
  }


  bool check_motion_coherence(MotionType motion_type) {
    if (is_shaker(motion_type) || is_pendulum(motion_type)) {
      if (amplitude_ <= 0.0) {
        color_log::warning("Driver_params::check_motion_coherence",
                           "The \"amplitude\" input slot is not defined correctly.");
        return false;
      }
      if (omega_ <= 0.0) {
        color_log::warning("Driver_params::check_motion_coherence",
                           "The \"omega\" input slot is not defined correctly.");
        return false;
      }

      if (is_shaker(motion_type)) {
        if (exanb::dot(shaker_dir_, shaker_dir_) - 1 >= 1e-14) {
          exanb::Vec3d old = shaker_dir_;
          _normalize(shaker_dir_);
          color_log::warning("Driver_params::check_motion_coherence", "Your shaker_dir vector [" + std::to_string(old) +
                             "} has been normalized to [" +
                             std::to_string(shaker_dir_) + "]");
        }
      } else if (is_pendulum(motion_type)) {
        if (pendulum_anchor_point_ == pendulum_initial_position_) {
          color_log::error("Driver_params::check_motion_coherence",
                           "The point defined in pendulum_anchor_point and the one in pendulum_initial_position are "
                           "the same. It is impossible to define a motion type PENDULUM_MOTION. Point: [" +
                           std::to_string(pendulum_anchor_point_) + "]");
        }
        if (exanb::dot(pendulum_swing_dir_, pendulum_swing_dir_) - 1 >= 1e-14) {
          exanb::Vec3d old = pendulum_swing_dir_;
          _normalize(pendulum_swing_dir_);
          color_log::warning("Driver_params::check_motion_coherence",
                             "Your pendulum_swing_dir vector [" + std::to_string(old) + "} has been normalized to [" +
                             std::to_string(pendulum_swing_dir_) + "]");
        }
      }
    }
    if (is_tabulated(motion_type)) {
      if (tab_time_.size() == 0) {
        color_log::warning("Driver_params::check_motion_coherence",
                           "The \"time\" input slot is not defined while the tabulated motion is activated.");
        return false;
      } else if (tab_time_[0] != 0.0) {
        color_log::warning("Driver_params::check_motion_coherence",
                           "Please set the first element of your input time vector to 0.");
        return false;
      }
      if (tab_pos_.size() == 0) {
        color_log::warning("Driver_params::check_motion_coherence",
                           "The \"positions\" input slot is not defined while the tabulated motion is activated.");
        return false;
      }
      if (tab_time_.size() != tab_pos_.size()) {
        color_log::warning("Driver_params::check_motion_coherence",
                           "The \"positions\" and \"time\" input slot are not the same size.");
        return false;
      }
      if (!is_sorted(tab_time_.begin(), tab_time_.end())) {
        color_log::warning("Driver_params::check_motion_coherence",
                           "The \"time\" array used for a TABULATED motion is not sorted.");
        return false;
      }
    }
    if (is_linear(motion_type)) {
      // Check if motion vector is zero (invalid for linear motion)
      if (motion_vector_ == exanb::Vec3d{0, 0, 0}) {
        exanb::lout << ansi::yellow("Your motion type is a \"Linear Mode\" that requires a motion vector.") << std::endl;
        exanb::lout << ansi::yellow(
            "Please, define motion vector by adding \"motion_vector: [1,0,0]. It is defined to [0,0,0] by "
            "default.")
            << std::endl;
        return false;
      }
      // Normalize motion vector if its magnitude is not equal to 1
      if (dot(motion_vector_, motion_vector_) - 1 >= 1e-14) {
        exanb::Vec3d old = motion_vector_;
        _normalize(motion_vector_);
        color_log::warning("Driver_params::check_motion_coherence", "Your motion vector [" + std::to_string(old) +
                           "} has been normalized to [" +
                           std::to_string(motion_vector_) + "]");
      }
      if (motion_type == LINEAR_MOTION && const_vel_ == 0) {
        color_log::warning("Driver_params::check_motion_coherence",
                           "You have chosen constant linear motion with zero velocity, please use \"const_vel\" or use "
                           "the motion type \"STATIONARY\".");
      }
      if (motion_type == LINEAR_FORCE_MOTION && const_force_ == 0) {
        color_log::warning("Driver_params::check_motion_coherence",
                           "You have chosen constant linear force motion with zero force, please use \"const_force\" "
                           "or use the motion type \"STATIONARY\".");
      }
    }

    if (is_compressive(motion_type)) {
      if (sigma_ == 0) {
        color_log::warning("Driver_params::check_motion_coherence",
                           "Sigma is to 0.0 while the compressive motion type is set to true.");
      }
      if (damprate_ <= 0) {
        color_log::warning("Driver_params::check_motion_coherence",
                           "Dumprate is to 0.0 while the compressive motion type is set to true.");
      }
    }

    return true;  // Return true if the motion is coherent
  }

  void print_driver_params(MotionType motion_type) const {
    exanb::lout << "Motion type        : " << motion_type_to_string(motion_type) << std::endl;

    if (is_tabulated(motion_type)) {
      std::stringstream times;
      std::stringstream positions;
      tabulations_to_stream(motion_type, times, positions);
      exanb::lout << times.rdbuf() << std::endl;
      exanb::lout << positions.rdbuf() << std::endl;
    }

    if (!is_stationary(motion_type)) {
      if (motion_start_threshold_ != 0 || motion_end_threshold_ != std::numeric_limits<double>::max()) {
        if (motion_end_threshold_ != std::numeric_limits<double>::max()) {
          exanb::lout << "Motion duration    : [ " << motion_start_threshold_ << "s , " << motion_end_threshold_ << "s ]"
              << std::endl;
        } else {
          exanb::lout << "Motion duration    : [ " << motion_start_threshold_ << "s ,  inf s )" << std::endl;
        }
      }
      if (is_linear(motion_type)) {
        exanb::lout << "Motion vector      : " << motion_vector_ << std::endl;
        if (motion_type == LINEAR_MOTION) {
          exanb::lout << "Velocity (constant): " << const_vel_ << std::endl;
        }
        if (motion_type == LINEAR_FORCE_MOTION) {
          exanb::lout << "Force (constant)   : " << const_force_ << std::endl;
        }
      }
      if (is_compressive(motion_type)) {
        exanb::lout << "Sigma              : " << sigma_ << std::endl;
        exanb::lout << "Damprate           : " << damprate_ << std::endl;
      }
    }

    if (is_shaker(motion_type)) {
      exanb::lout << "Shaker.Omega       : " << omega_ << std::endl;
      exanb::lout << "Shaker.Amplitude   : " << amplitude_ << std::endl;
      exanb::lout << "Shaker.Direction   : [" << shaker_dir_ << "]" << std::endl;
    }

    if (exaDEM::is_expr(motion_type)) {
      expr_.expr_display(exanb::lout);
    }
  }

  /**
   * @brief Write Driver data into a stream.
   */
  void dump_driver_params(MotionType motion_type, std::stringstream& stream) const {
    stream << "     params: {";
    stream << " motion_type: " << motion_type_to_string(motion_type);
    stream << ", motion_vector: [" << motion_vector_ << "]";
    stream << ", motion_start_threshold: " << motion_start_threshold_;
    stream << ", motion_end_threshold: " << motion_end_threshold_;
    if (motion_type == MotionType::LINEAR_MOTION) {
      stream << ", const_vel: " << const_vel_;
    }
    if (motion_type == MotionType::LINEAR_FORCE_MOTION) {
      stream << ", const_force: " << const_force_;
    }
    if (is_compressive(motion_type)) {
      stream << ", sigma: " << sigma_;
      stream << ", damprate: " << damprate_;
    }
    if (motion_type == MotionType::TABULATED) {
      std::stringstream times;
      std::stringstream positions;
      tabulations_to_stream(motion_type, times, positions);
      stream << ", " << times.rdbuf() << ", " << positions.rdbuf();
    }
    if (motion_type == MotionType::SHAKER) {
      stream << ", omega: " << omega_;
      stream << ", amplitude: " << amplitude_;
      stream << ", shaker_dir: [" << shaker_dir_ << "]";
    }
    if (motion_type == MotionType::PENDULUM_MOTION) {
      stream << ", omega: " << omega_;
      stream << ", amplitude: " << amplitude_;
      stream << ", pendulum_anchor_point: [" << pendulum_anchor_point_ << "]";
      stream << ", pendulum_initial_position: [" << pendulum_initial_position_ << "]";
      stream << ", pendulum_swing_dir: [" << pendulum_swing_dir_ << "]";
    }

    if (exaDEM::is_expr(motion_type)) {
      expr_.expr_dump(stream);
    }
    stream << " }" << std::endl;
  }

  /* Tabulated motion routines */
  exanb::Vec3d tab_to_position(double time) const {
    assert(time >= 0.0);
    auto ite = std::lower_bound(tab_time_.begin(), tab_time_.end(), time);
    if (ite == tab_time_.end()) {
      return tab_pos_.back();
    } else {
      size_t idx_lower = ite - tab_time_.begin() - 1;
      size_t idx_upper = idx_lower + 1;
      assert(tab_time_[idx_lower] >= time);
      if (idx_upper >= tab_time_.size()) return tab_pos_.back();
      double Dt = (time - tab_time_[idx_lower]) / (tab_time_[idx_upper] - tab_time_[idx_lower]);
      exanb::Vec3d P = (tab_pos_[idx_upper] - tab_pos_[idx_lower]) * Dt + tab_pos_[idx_lower];
      return P;
    }
  }

  exanb::Vec3d tab_to_velocity(double time) const {
    assert(time >= 0.0);
    auto ite = std::lower_bound(tab_time_.begin(), tab_time_.end(), time);
    if (ite == tab_time_.end()) {
      return exanb::Vec3d(0, 0, 0);  // stationnary
    } else {
      size_t idx_lower = ite - tab_time_.begin() - 1;
      size_t idx_upper = idx_lower + 1;
      assert(tab_time_[idx_lower] >= time);
      if (idx_upper >= tab_time_.size()) return exanb::Vec3d(0, 0, 0);
      double Dt = tab_time_[idx_upper] - tab_time_[idx_lower];
      assert(Dt != 0.0);
      exanb::Vec3d V = (tab_pos_[idx_upper] - tab_pos_[idx_lower]) / Dt;
      return V;
    }
  }

  /* Shaker routines */
  exanb::Vec3d shaker_direction() const {
    return shaker_dir_;
  }

  double shaker_signal(double time) const {
    assert(motion_start_threshold_ >= 0);
    time -= motion_start_threshold_;
    return amplitude_ * sin(omega_ * time);
  }

  exanb::Vec3d shaker_velocity(double time) const {
    assert(motion_start_threshold_ >= 0);
    time -= motion_start_threshold_;
    return amplitude_ * omega_ * cos(omega_ * time) * shaker_direction();
  }

  /* Pendulum routines */
  exanb::Vec3d pendulum_direction() const {
    return pendulum_swing_dir_;
  }

  exanb::Vec3d pendulum_velocity(double time) const {
    return {0, 0, 0};
  }

  std::pair<double, exanb::Vec3d> compute_offset_normal_pendulum_motion(double time) const {
    if (time < motion_start_threshold_) {
      color_log::error("compute_normal_pendulum_motion",
                       "This call is ill-formed, please verify that time is superior to motion_start_threshold_.");
    }
    exanb::Vec3d v1 = pendulum_anchor_point_;
    exanb::Vec3d v2 = pendulum_initial_position_;
    exanb::Vec3d v3 = pendulum_initial_position_ + pendulum_direction() * pendulum_signal(time);

    // warning, if v2 = v3, we return an offset of  0 and the pendulum direction
    if (exanb::dot(v3, v2) < 1e-16) {
      return {0.0, pendulum_direction()};
    }

    exanb::Vec3d v1v3 = v3 - v1;
    v1v3 = v1v3 / exanb::norm(v1v3);
    exanb::Vec3d project_v2_v1v3 = exanb::dot(v1v3, v2 - v1) * v1v3 + v1;
    exanb::Vec3d dir_proj_v2_v2 = project_v2_v1v3 - v2;
    exanb::Vec3d normal = dir_proj_v2_v2 / exanb::norm(dir_proj_v2_v2);
    double offset = exanb::dot(v1, normal);
    return {offset, normal};
  }

  double pendulum_signal(double time) const {
    assert(motion_start_threshold_ >= 0);
    time -= motion_start_threshold_;
    return amplitude_ * sin(omega_ * time);
  }

  // Expression routines
  exanb::Vec3d driver_expr_v(double time) const{
    assert(motion_start_threshold_ >= 0);
    time -= motion_start_threshold_;
    return expr_.expr_v(time);
  }

  exanb::Vec3d driver_expr_vrot(double time) const {
    assert(motion_start_threshold_ >= 0);
    time -= motion_start_threshold_;
    return expr_.expr_vrot(time);
  }

  exanb::Vec3d driver_expr_mom(double time) const {
    assert(motion_start_threshold_ >= 0);
    time -= motion_start_threshold_;
    return expr_.expr_mom(time);
  }
};
}  // namespace exaDEM

namespace YAML {
template <>
struct convert<exaDEM::Driver_params> {
  static bool decode(const Node& node, exaDEM::Driver_params& v) {
    std::string function_name = "Driver_params::decode";
    if (!node.IsMap()) {
      return false;
    }
    if (!node["motion_type"]) {
      color_log::error(function_name, "mmotion_type is missing.", false);
      return false;
    }

    v = {};
    v.input_motion_type_ = exaDEM::string_to_motion_type(node["motion_type"].as<std::string>());
    if (is_linear(v.input_motion_type_)) {
      if (!node["motion_vector"]) {
        color_log::error(function_name, "motion_vector is missing.", false);
        return false;
      }
      v.motion_vector_ = node["motion_vector"].as<exanb::Vec3d>();

      if (v.input_motion_type_ == exaDEM::MotionType::LINEAR_MOTION) {
        if (!node["const_vel"]) {
          color_log::error(function_name, "const_vel is missing.", false);
          return false;
        }
        if (node["const_vel"]) {
          v.const_vel_ = node["const_vel"].as<Quantity>().convert();
        }
      }
      if (v.input_motion_type_ == exaDEM::MotionType::LINEAR_FORCE_MOTION) {
        if (!node["const_force"]) {
          color_log::error(function_name, "const_force is missing.", false);
          return false;
        }
        if (node["const_force"]) {
          v.const_force_ = node["const_force"].as<double>();
        }
      }
    }
    if (is_compressive(v.input_motion_type_)) {
      if (!node["sigma"]) {
        color_log::error(function_name, "sigma is missing.", false);
        return false;
      }
      v.sigma_ = node["sigma"].as<double>();
      if (!node["damprate"]) {
        color_log::error(function_name, "damprate is missing.", false);
        return false;
      }
      v.damprate_ = node["damprate"].as<double>();
    }
    // Tabulation
    if (is_tabulated(v.input_motion_type_)) {
      if (!node["time"]) {
        color_log::error(function_name, "time is missing.", false);
        return false;
      }
      v.tab_time_ = node["time"].as<std::vector<double>>();
      if (!node["positions"]) {
        color_log::error(function_name, "position is missing.", false);
        return false;
      }
      v.tab_pos_ = node["positions"].as<std::vector<exanb::Vec3d>>();
    }

    // Shaker && Pendulum
    if (is_shaker(v.input_motion_type_) || is_pendulum(v.input_motion_type_)) {
      if (!node["omega"]) {
        color_log::error(function_name, "omega is missing.", false);
        return false;
      }
      v.omega_ = node["omega"].as<Quantity>().convert();
      if (!node["amplitude"]) {
        color_log::error(function_name, "amplitude is missing.", false);
        return false;
      }
      v.amplitude_ = node["amplitude"].as<Quantity>().convert();

      if (is_shaker(v.input_motion_type_)) {
        if (!node["shaker_dir"]) {
          color_log::warning("Driver_params::decode", "shaker_dir is missing, default is [0,0,1].");
          v.shaker_dir_ = exanb::Vec3d{0, 0, 1};
        } else {
          v.shaker_dir_ = node["shaker_dir"].as<exanb::Vec3d>();
        }
      } else if (is_pendulum(v.input_motion_type_)) {
        if (!node["pendulum_anchor_point"]) {
          color_log::error(function_name, "pendulum_anchor_point is missing.", false);
          return false;
        }
        v.pendulum_anchor_point_ = node["pendulum_anchor_point"].as<exanb::Vec3d>();
        if (!node["pendulum_initial_position"]) {
          color_log::error(function_name, "pendulum_initial_position is missing.", false);
          return false;
        }
        v.pendulum_initial_position_ = node["pendulum_initial_position"].as<exanb::Vec3d>();
        if (!node["pendulum_swing_dir"]) {
          color_log::error(function_name, "pendulum_swing_dir is missing.", false);
          return false;
        }
        v.pendulum_swing_dir_ = node["pendulum_swing_dir"].as<exanb::Vec3d>();
      }
    } else if (is_expr(v.input_motion_type_)) {
      if (!node["expr"]) {
        color_log::error(function_name, "expr is missing while the motion type is set to EXPRESSION");
      }
      v.expr_ = node["expr"].as<exaDEM::Driver_expr>();
    }

    if (node["motion_start_threshold"]) {
      v.motion_start_threshold_ = node["motion_start_threshold"].as<Quantity>().convert();
    }

    if (node["motion_end_threshold"]) {
      v.motion_end_threshold_ = node["motion_end_threshold"].as<Quantity>().convert();
    }
    return true;
  }
};
}  // namespace YAML
