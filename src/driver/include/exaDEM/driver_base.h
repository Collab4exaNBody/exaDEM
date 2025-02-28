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
#include <exanb/core/basic_types.h>
#include <string.h>
#include <tuple>

namespace exaDEM
{
  using namespace exanb;
  /**
   * @brief Enumeration representing different types of drivers in the exaDEM simulation.
   */
  enum DRIVER_TYPE
  {
    CYLINDER, /**< Cylinder driver type. */
    SURFACE,  /**< Surface driver type. */
    BALL,     /**< Ball driver type. */
    STL_MESH, /**< STL mesh driver type. */
    UNDEFINED /**< Undefined driver type. */
  };

  constexpr int DRIVER_TYPE_SIZE = 5; /**< Size of the driver type enum. */

  /**
   * @brief Converts a DRIVER_TYPE enum value to its corresponding string representation.
   * @param type The DRIVER_TYPE enum value.
   * @return The string representation of the DRIVER_TYPE.
   */
  inline std::string print(DRIVER_TYPE type)
  {
    switch (type)
    {
    case DRIVER_TYPE::CYLINDER:
      return "Cylinder";
    case DRIVER_TYPE::SURFACE:
      return "Surface";
    case DRIVER_TYPE::BALL:
      return "Ball";
    case DRIVER_TYPE::STL_MESH:
      return "Stl_mesh";
    case DRIVER_TYPE::UNDEFINED:
      return "Undefined Driver";
    default:
      return "Undefined Driver";
    }
  }

  constexpr unsigned int str2int(const char *str, int h = 0) { return !str[h] ? 5381 : (str2int(str, h + 1) * 33) ^ str[h]; }

  /**
   * @brief Converts a string representation of a driver type to its corresponding DRIVER_TYPE enum value.
   * @param driver_name The name of the driver as a string.
   * @return The corresponding DRIVER_TYPE enum value.
   */
  inline DRIVER_TYPE get_type(std::string driver_name)
  {
    switch (str2int(driver_name.c_str()))
    {
    case str2int("CYLINDER"):
      return DRIVER_TYPE::CYLINDER;
    case str2int("SURFACE"):
      return DRIVER_TYPE::SURFACE;
    case str2int("BALL"):
      return DRIVER_TYPE::BALL;
    case str2int("STL_MESH"):
      return DRIVER_TYPE::STL_MESH;
    default:
      std::cout << "error, no driver " << driver_name << " found" << std::endl;
      std::cout << "Use: CYLINDER, SURFACE, or BALL" << std::endl;
      std::abort();
    }
  }

  struct Cylinder;
  struct Surface;
  struct Ball;
  struct Stl_mesh;
  struct UndefinedDriver;

  /**
   * @brief Template function to get the DRIVER_TYPE of a given type.
   * @tparam T The type whose DRIVER_TYPE is to be obtained.
   * @return The DRIVER_TYPE corresponding to the type T.
   */
  template <typename T> constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::UNDEFINED; }
  template <> constexpr DRIVER_TYPE get_type<exaDEM::Cylinder>() { return DRIVER_TYPE::CYLINDER; }
  template <> constexpr DRIVER_TYPE get_type<exaDEM::Surface>() { return DRIVER_TYPE::SURFACE; }
  template <> constexpr DRIVER_TYPE get_type<exaDEM::Ball>() { return DRIVER_TYPE::BALL; }
  template <> constexpr DRIVER_TYPE get_type<exaDEM::Stl_mesh>() { return DRIVER_TYPE::STL_MESH; }
  template <> constexpr DRIVER_TYPE get_type<exaDEM::UndefinedDriver>() { return DRIVER_TYPE::UNDEFINED; }

  /**
   * @brief Abstract base class representing a driver in the exaDEM simulation.
   */
  struct Driver
  {
    constexpr DRIVER_TYPE get_type();
    virtual void print();
    virtual bool filter(const double, const Vec3d &);
    virtual std::tuple<bool, double, Vec3d, Vec3d> dectector(const double, const Vec3d &);
  };
} // namespace exaDEM
