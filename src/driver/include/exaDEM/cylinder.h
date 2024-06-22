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

#include <exaDEM/driver_base.h>

namespace exaDEM
{
  using namespace exanb;

  /**
   * @brief Struct representing a cylinder in the exaDEM simulation.
   */
  struct Cylinder
  {
    double radius;          /**< Radius of the cylinder. */
    exanb::Vec3d axis;      /**< Axis direction of the cylinder. */
    exanb::Vec3d center;    /**< Center position of the cylinder. */
    exanb::Vec3d vel;       /**< Velocity of the cylinder. */
    exanb::Vec3d vrot;      /**< Angular velocity of the cylinder. */

    /**
     * @brief Get the type of the driver (in this case, CYLINDER).
     * @return The type of the driver.
     */
    constexpr DRIVER_TYPE get_type() {return DRIVER_TYPE::CYLINDER;}

    /**
     * @brief Print information about the cylinder.
     */
    void print()
    {
      std::cout << "Driver Type: Cylinder" << std::endl;
      std::cout << "Radius: " << radius << std::endl;
      std::cout << "Axis  : " << axis << std::endl;
      std::cout << "Center: " << center << std::endl;
      std::cout << "Vel   : " << vel << std::endl;
      std::cout << "AngVel: " << vrot << std::endl;
    }


    /**
     * @brief return driver velocity
     */
    ONIKA_HOST_DEVICE_FUNC inline Vec3d& get_vel()
    {
      return vel;
    }

    /**
     * @brief Filter function to check if a point is within a certain radius of the cylinder.
     * @param rcut The cut-off radius. Note: rcut = rverlet + r shape
     * @param vi The vector representing the point to check.
     * @return True if the point is within the cut-off radius of the cylinder, false otherwise.
     */
    ONIKA_HOST_DEVICE_FUNC inline bool filter( const double rcut, const Vec3d& vi)
    {
      const Vec3d proj = vi * axis;

      // === direction
      const auto dir = proj - center;

      // === interpenetration
      const double d = norm(dir);
      const double dn = radius - (d + rcut);
      return dn <= 0;
    }

    /**
     * @brief Detects the intersection between a vertex of a polyhedron and a cylinder.
     *
     * This function checks if a vertex, represented by its position 'pi' and orientation 'oi',
     * intersects with a cylindrical shape defined by its center projection 'center_proj', axis 'axis',
     * and radius 'radius'.
     *
     * @param rcut The shape radius. 
     * @param pi The position of the vertex.
     *
     * @return A tuple containing:
     *   - A boolean indicating whether there is an intersection (true) or not (false).
     *   - The penetration depth in case of intersection.
     *   - The contact normal at the intersection point.
     *   - The contact point between the vertex and the cylinder.
     */
    // rcut = r shape
    ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detector(const double rcut, const Vec3d& pi)
    {
      // === project the vertex in the plan as the cylinder center
      const Vec3d proj = pi * axis;

      // === direction
      const Vec3d dir = center - proj;

      // === interpenetration
      const double d = exanb::norm(dir);

      // === compute interpenetration
      const double dn = radius - (rcut + d);

      if ( dn > 0 )
      {
        return {false, 0.0, Vec3d(), Vec3d()};
      }
      else
      {
        // === compute contact normal 
        const Vec3d n = dir / d;

        // === compute contact point
        const Vec3d contact_position = pi - n * (rcut + 0.5 * dn);

        return {true, dn, n, contact_position};
      }
    }
  };
}
