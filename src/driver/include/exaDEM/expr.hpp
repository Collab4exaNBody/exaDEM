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
using exanb::Vec3d;
struct Driver_expr {
  bool expr_use_v = false;
  std::string expr_vx = "0";
  std::string expr_vy = "0";
  std::string expr_vz = "0";
  bool expr_use_vrot = false;
  std::string expr_vrotx = "0";
  std::string expr_vroty = "0";
  std::string expr_vrotz = "0";
  bool expr_use_mom = false;
  std::string expr_momx = "0";
  std::string expr_momy = "0";
  std::string expr_momz = "0";

  Vec3d expr_v(double time) {
    assert(expr_use_v);
    Vec3d res;
    te_variable the_t[] = {{"t", &time}};

    int err;
    te_expr* te_expr_vx = te_compile(expr_vx.c_str(), the_t, 1, &err);
    te_expr* te_expr_vy = te_compile(expr_vy.c_str(), the_t, 1, &err);
    te_expr* te_expr_vz = te_compile(expr_vz.c_str(), the_t, 1, &err);

    res.x = te_eval(te_expr_vx);
    res.y = te_eval(te_expr_vy);
    res.z = te_eval(te_expr_vz);
    return res;
  }

  Vec3d expr_vrot(double time) {
    assert(expr_use_vrot);
    Vec3d res;
    te_variable the_t[] = {{"t", &time}};

    int err;
    te_expr* te_expr_vrotx = te_compile(expr_vrotx.c_str(), the_t, 1, &err);
    te_expr* te_expr_vroty = te_compile(expr_vroty.c_str(), the_t, 1, &err);
    te_expr* te_expr_vrotz = te_compile(expr_vrotz.c_str(), the_t, 1, &err);

    res.x = te_eval(te_expr_vrotx);
    res.y = te_eval(te_expr_vroty);
    res.z = te_eval(te_expr_vrotz);
    return res;
  }

  Vec3d expr_mom(double time) {
    assert(expr_use_mom);
    Vec3d res;
    te_variable the_t[] = {{"t", &time}};

    int err;
    te_expr* te_expr_momx = te_compile(expr_momx.c_str(), the_t, 1, &err);
    te_expr* te_expr_momy = te_compile(expr_momy.c_str(), the_t, 1, &err);
    te_expr* te_expr_momz = te_compile(expr_momz.c_str(), the_t, 1, &err);

    res.x = te_eval(te_expr_momx);
    res.y = te_eval(te_expr_momy);
    res.z = te_eval(te_expr_momz);
    return res;
  }

  template <typename Stream>
  void expr_display(Stream& stream) const {
    stream << "Expression Vx      : " << expr_vx << std::endl;
    stream << "Expression Vy      : " << expr_vy << std::endl;
    stream << "Expression Vz      : " << expr_vz << std::endl;
    stream << "Expression Vrotx   : " << expr_vrotx << std::endl;
    stream << "Expression Vroty   : " << expr_vroty << std::endl;
    stream << "Expression Vrotz   : " << expr_vrotz << std::endl;
    stream << "Expression Momx   : " << expr_momx << std::endl;
    stream << "Expression Momy   : " << expr_momy << std::endl;
    stream << "Expression Momz   : " << expr_momz << std::endl;
  }

  template <typename Stream>
  void expr_dump(Stream& stream) const {
    stream << ", expr: { ";
    stream << "vx: " << expr_vx << " , ";
    stream << "vy: " << expr_vy << " , ";
    stream << "vz: " << expr_vz << " , ";
    stream << "vrotx: " << expr_vrotx << " , ";
    stream << "vroty: " << expr_vroty << " , ";
    stream << "vrotz: " << expr_vrotz << " , ";
    stream << "momx: " << expr_momx << " , ";
    stream << "momy: " << expr_momy << " , ";
    stream << "momz: " << expr_momz << " } ";
  }
};
}  // namespace exaDEM

namespace YAML {
using exaDEM::Driver_expr;
using exanb::lerr;
using exanb::lout;

template <>
struct convert<Driver_expr> {
  static bool decode(const Node& node, Driver_expr& expr) {
    if (!node.IsMap()) {
      lout << "Please, define this driver motion as STATIONNARY" << std::endl;
      return false;
    }
    if (node["vx"]) {
      expr.expr_use_v = true;
      expr.expr_vx = node["vx"].as<std::string>();
    }
    if (node["vy"]) {
      expr.expr_use_v = true;
      expr.expr_vy = node["vy"].as<std::string>();
    }
    if (node["vz"]) {
      expr.expr_use_v = true;
      expr.expr_vz = node["vz"].as<std::string>();
    }
    if (node["vrotx"]) {
      expr.expr_use_vrot = true;
      expr.expr_vrotx = node["vrotx"].as<std::string>();
    }
    if (node["vroty"]) {
      expr.expr_use_vrot = true;
      expr.expr_vroty = node["vroty"].as<std::string>();
    }
    if (node["vrotz"]) {
      expr.expr_use_vrot = true;
      expr.expr_vrotz = node["vrotz"].as<std::string>();
    }
    if (node["momx"]) {
      expr.expr_use_mom = true;
      expr.expr_momx = node["momx"].as<std::string>();
    }
    if (node["momy"]) {
      expr.expr_use_mom = true;
      expr.expr_momy = node["momy"].as<std::string>();
    }
    if (node["momz"]) {
      expr.expr_use_mom = true;
      expr.expr_momz = node["momz"].as<std::string>();
    }
    return true;
  }
};
}  // namespace YAML
