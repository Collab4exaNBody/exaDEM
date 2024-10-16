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
#include <exaDEM/driver_base.h>

namespace exaDEM
{
  using namespace exanb;

  /**
   * @brief Struct representing a surface in the exaDEM simulation.
   */
  struct Surface
  {
    double offset;            /**< Offset from the origin along the normal vector. */
    exanb::Vec3d normal;      /**< Normal vector of the surface. */
    exanb::Vec3d center;      /**< Center position of the surface. */
    double vel;               /**< Velocity of the surface. */
    exanb::Vec3d vrot;        /**< Angular velocity of the surface. */
    exanb::Vec3d center_proj; /**< Center position projected on the norm. */

    /**
     * @brief Get the type of the driver (in this case, SURFACE).
     * @return The type of the driver.
     */
    constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::SURFACE; }

    /**
     * @brief Print information about the surface.
     */
    void print()
    {
      lout << "Driver Type: Surface" << std::endl;
      lout << "Offset: " << offset << std::endl;
      lout << "Normal: " << normal << std::endl;
      lout << "Center: " << center << std::endl;
      lout << "Vel   : " << vel << std::endl;
      lout << "AngVel: " << vrot << std::endl;
    }

    /**
     * @brief Write surface data into a stream.
     */
    void dump_driver(int id, std::stringstream &stream)
    {
      stream << "  - add_surface:" << std::endl;
      stream << "     id: " << id << std::endl;
      stream << "     offset: " << this->offset << std::endl;
      stream << "     center: [" << this->center << "]" << std::endl;
      stream << "     normal: [" << this->normal << "]" << std::endl;
      stream << "     velocity: " << this->vel << std::endl;
      stream << "     angular_velocity: [" << this->vrot << "]" << std::endl;
    }

    /**
     * @brief Initialize the surface.
     * @details Calculates the center position based on the normal and offset.
     */
    inline void initialize()
    {
      center_proj = normal * offset;
      // checks
      if (exanb::dot(center, normal) != exanb::dot(center_proj, normal))
        lout << "Warning, center point (surface) is not correctly defined" << std::endl;

      if (exanb::dot(center, normal) != exanb::dot(center_proj, normal))
      {
        center += (offset - exanb::dot(center, normal)) * normal;
        lout << "center is re-computed because it doesn't fit with offset, new center is: " << center << " and center_proj is: " << center_proj << std::endl;
      }

      // if( exanb::dot(normal,normal) != 1 )  lout << "Warning, normal vector (surface) is not correctly defined" << std::endl;
    }

    /**
     * @brief Compute offset if we ignore forces apply on this surface.
     * @param t The time step.
     */
    ONIKA_HOST_DEVICE_FUNC inline double compute_pos_from_vel(const double t) { return offset + t * vel; }

    /**
     * @brief return driver velocity
     */
    ONIKA_HOST_DEVICE_FUNC inline Vec3d get_vel() { return normal * vel; }

    /**
     * @brief Update the position of the wall.
     * @param t The time step.
     */
    ONIKA_HOST_DEVICE_FUNC inline void push_v_to_r(const double t)
    {
      center = center + t * vel * normal;
      center_proj = center_proj + t * vel * normal;
      offset += t * vel;
    }

    /**
     * @brief Filter function to check if a vertex is within a certain radius of the surface.
     * @param rcut The cut-off radius.
     * @param p The point to check.
     * @return True if the point is within the cut-off radius of the surface, false otherwise.
     */
    ONIKA_HOST_DEVICE_FUNC inline bool filter(const double rcut, const exanb::Vec3d &p)
    {
      Vec3d proj = dot(p, normal) * normal;
      double d = norm(proj - center_proj);
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
    ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detector(const double rcut, const Vec3d &p)
    {
      Vec3d proj = dot(p, normal) * normal;
      Vec3d surface_to_point = -(center_proj - proj);
      double d = norm(surface_to_point);
      double dn = d - rcut;
      if (dn > 0)
      {
        return {false, 0.0, Vec3d(), Vec3d()};
      }
      else
      {
        Vec3d n = surface_to_point / d;
        Vec3d contact_position = p - n * (rcut + 0.5 * dn);
        return {true, dn, n, contact_position};
      }
    }
  };
} // namespace exaDEM
