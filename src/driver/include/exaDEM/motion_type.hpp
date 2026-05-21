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

namespace exaDEM {
enum MotionType {
  STATIONARY,                /**< Stationary state, with no motion. */
  LINEAR_MOTION,             /**< Linear movement type, straight-line motion. */
  COMPRESSIVE_FORCE,         /**< Movement influenced by compressive forces. */
  LINEAR_FORCE_MOTION,       /**< Linear motion type influenced by applied forces. */
  PARTICLE,                  /**< General movement caused by applied forces. */
  LINEAR_COMPRESSIVE_MOTION, /**< Linear movement combined with compressive forces. */
  TABULATED,                 /**< Motion defined by precomputed or tabulated data. */
  SHAKER,                    /**< Oscillatory or vibratory motion, typically mimicking a shaking mechanism. */
  PENDULUM_MOTION,           /**< Oscillatory swinging around a suspension point (pendulum-like). */
  EXPRESSION,
  UNKNOWN
};

inline std::string motion_type_to_string(MotionType motion_type) {
  switch (motion_type) {
    case STATIONARY:
      return "STATIONARY";
    case LINEAR_MOTION:
      return "LINEAR_MOTION";
    case COMPRESSIVE_FORCE:
      return "COMPRESSIVE_FORCE";
    case LINEAR_FORCE_MOTION:
      return "LINEAR_FORCE_MOTION";
    case PARTICLE:
      return "PARTICLE";
    case LINEAR_COMPRESSIVE_MOTION:
      return "LINEAR_COMPRESSIVE_MOTION";
    case TABULATED:
      return "TABULATED";
    case SHAKER:
      return "SHAKER";
    case PENDULUM_MOTION:
      return "PENDULUM_MOTION";
    case EXPRESSION:
      return "EXPRESSION";
    default:
      return "UNKNOWN";
  }
}

inline MotionType string_to_motion_type(const std::string& str) {
  if (str == "STATIONARY") {
    return STATIONARY;
  } else if (str == "LINEAR_MOTION") {
    return LINEAR_MOTION;
  } else if (str == "COMPRESSIVE_FORCE") {
    return COMPRESSIVE_FORCE;
  } else if (str == "LINEAR_FORCE_MOTION") {
    return LINEAR_FORCE_MOTION;
  } else if (str == "PARTICLE") {
    return PARTICLE;
  } else if (str == "LINEAR_COMPRESSIVE_MOTION") {
    return LINEAR_COMPRESSIVE_MOTION;
  } else if (str == "TABULATED") {
    return TABULATED;
  } else if (str == "SHAKER") {
    return SHAKER;
  } else if (str == "PENDULUM_MOTION") {
    return PENDULUM_MOTION;
  } else if (str == "EXPRESSION") {
    return EXPRESSION;
  }

  // If the string doesn't match any valid MotionType, return a default value
  return UNKNOWN;  // Or some other default action like throwing an exception or logging
}

ONIKA_HOST_DEVICE_FUNC
inline bool is_stationary(MotionType motion_type) {
  return motion_type == MotionType::STATIONARY;
}

ONIKA_HOST_DEVICE_FUNC
inline bool is_tabulated(MotionType motion_type) {
  return motion_type == MotionType::TABULATED;
}

ONIKA_HOST_DEVICE_FUNC
inline bool is_shaker(MotionType motion_type) {
  return motion_type == MotionType::SHAKER;
}

ONIKA_HOST_DEVICE_FUNC
inline bool is_pendulum(MotionType motion_type) {
  return motion_type == MotionType::PENDULUM_MOTION;
}

ONIKA_HOST_DEVICE_FUNC
inline bool is_expr(MotionType motion_type) {
  return motion_type == MotionType::EXPRESSION;
}

ONIKA_HOST_DEVICE_FUNC
inline bool is_linear(MotionType motion_type) {
  return (motion_type == MotionType::LINEAR_MOTION || motion_type == MotionType::LINEAR_FORCE_MOTION ||
          motion_type == MotionType::LINEAR_COMPRESSIVE_MOTION);
}

ONIKA_HOST_DEVICE_FUNC
inline bool is_compressive(MotionType motion_type) {
  return (motion_type == MotionType::COMPRESSIVE_FORCE || motion_type == MotionType::LINEAR_COMPRESSIVE_MOTION);
}

ONIKA_HOST_DEVICE_FUNC
inline bool is_force_motion(MotionType motion_type) {
  return (motion_type == MotionType::PARTICLE || motion_type == MotionType::LINEAR_FORCE_MOTION);
}

ONIKA_HOST_DEVICE_FUNC
inline bool need_forces(MotionType motion_type) {
  // Need for LINEAR_FORCE_MOTION
  // No need for STATIONARY
  // No need for LINEAR_MOTION
  // Need for PARTICLE
  // Need for COMPRESSIVE_FORCE
  // Need for LINEAR_COMPRESSIVE_MOTION
  return is_compressive(motion_type) || motion_type == MotionType::PARTICLE || motion_type == MotionType::LINEAR_FORCE_MOTION;
}

// Checks
inline bool is_valid_motion_type(const MotionType motion_type, const std::vector<MotionType>& valid_motion_types) {
  auto it = std::find(valid_motion_types.begin(), valid_motion_types.end(), motion_type);
  if (it == valid_motion_types.end()) {
    color_log::warning(
        "Driver_params::is_valid_motion_type",
        "This motion type [" + motion_type_to_string(motion_type) + "] is not possible, MotionType availables are: ");
    for (const auto& motion : valid_motion_types) {
      exanb::lout << " " << ansi::yellow(motion_type_to_string(motion));
    }
    exanb::lout << std::endl;
    return false;
  }
  return true;
}
}  // namespace exaDEM
