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

#include "tinyexpr.h"

namespace exaDEM {
struct Driver_expr {
  bool expr_use_v_ = false;
  std::string expr_vx_ = "0";
  std::string expr_vy_ = "0";
  std::string expr_vz_ = "0";
  bool expr_use_vrot_ = false;
  std::string expr_vrotx_ = "0";
  std::string expr_vroty_ = "0";
  std::string expr_vrotz_ = "0";
  bool expr_use_mom_ = false;
  std::string expr_momx_ = "0";
  std::string expr_momy_ = "0";
  std::string expr_momz_ = "0";

  /**
   * @brief Compiles and evaluates an expression of the variable t.
   * @return The value of the expression, NaN if the expression is invalid.
   */
  static double expr_eval(const std::string& expr, double time) {
    te_variable the_t[] = {{"t", &time}};
    int err;
    te_expr* compiled_expr = te_compile(expr.c_str(), the_t, 1, &err);
    const double res = te_eval(compiled_expr);
    te_free(compiled_expr);
    return res;
  }

  exanb::Vec3d expr_v(double time) const {
    assert(expr_use_v_);
    return {expr_eval(expr_vx_, time), expr_eval(expr_vy_, time), expr_eval(expr_vz_, time)};
  }

  exanb::Vec3d expr_vrot(double time) const {
    assert(expr_use_vrot_);
    return {expr_eval(expr_vrotx_, time), expr_eval(expr_vroty_, time), expr_eval(expr_vrotz_, time)};
  }

  exanb::Vec3d expr_mom(double time) const {
    assert(expr_use_mom_);
    return {expr_eval(expr_momx_, time), expr_eval(expr_momy_, time), expr_eval(expr_momz_, time)};
  }

  template <typename Stream>
  void expr_display(Stream& stream) const {
    stream << "Expression Vx      : " << expr_vx_ << std::endl;
    stream << "Expression Vy      : " << expr_vy_ << std::endl;
    stream << "Expression Vz      : " << expr_vz_ << std::endl;
    stream << "Expression Vrotx   : " << expr_vrotx_ << std::endl;
    stream << "Expression Vroty   : " << expr_vroty_ << std::endl;
    stream << "Expression Vrotz   : " << expr_vrotz_ << std::endl;
    stream << "Expression Momx   : " << expr_momx_ << std::endl;
    stream << "Expression Momy   : " << expr_momy_ << std::endl;
    stream << "Expression Momz   : " << expr_momz_ << std::endl;
  }

  template <typename Stream>
  void expr_dump(Stream& stream) const {
    stream << ", expr: { ";
    stream << "vx: " << expr_vx_ << " , ";
    stream << "vy: " << expr_vy_ << " , ";
    stream << "vz: " << expr_vz_ << " , ";
    stream << "vrotx: " << expr_vrotx_ << " , ";
    stream << "vroty: " << expr_vroty_ << " , ";
    stream << "vrotz: " << expr_vrotz_ << " , ";
    stream << "momx: " << expr_momx_ << " , ";
    stream << "momy: " << expr_momy_ << " , ";
    stream << "momz: " << expr_momz_ << " } ";
  }
};
}  // namespace exaDEM

namespace YAML {
template <>
struct convert<exaDEM::Driver_expr> {
  static bool decode(const Node& node, exaDEM::Driver_expr& expr) {
    if (!node.IsMap()) {
      exanb::lout << "Please, define this driver motion as STATIONNARY" << std::endl;
      return false;
    }
    if (node["vx"]) {
      expr.expr_use_v_ = true;
      expr.expr_vx_ = node["vx"].as<std::string>();
    }
    if (node["vy"]) {
      expr.expr_use_v_ = true;
      expr.expr_vy_ = node["vy"].as<std::string>();
    }
    if (node["vz"]) {
      expr.expr_use_v_ = true;
      expr.expr_vz_ = node["vz"].as<std::string>();
    }
    if (node["vrotx"]) {
      expr.expr_use_vrot_ = true;
      expr.expr_vrotx_ = node["vrotx"].as<std::string>();
    }
    if (node["vroty"]) {
      expr.expr_use_vrot_ = true;
      expr.expr_vroty_ = node["vroty"].as<std::string>();
    }
    if (node["vrotz"]) {
      expr.expr_use_vrot_ = true;
      expr.expr_vrotz_ = node["vrotz"].as<std::string>();
    }
    if (node["momx"]) {
      expr.expr_use_mom_ = true;
      expr.expr_momx_ = node["momx"].as<std::string>();
    }
    if (node["momy"]) {
      expr.expr_use_mom_ = true;
      expr.expr_momy_ = node["momy"].as<std::string>();
    }
    if (node["momz"]) {
      expr.expr_use_mom_ = true;
      expr.expr_momz_ = node["momz"].as<std::string>();
    }
    return true;
  }
};
}  // namespace YAML
