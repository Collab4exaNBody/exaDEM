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
   * @brief Struct representing a ball in the exaDEM simulation.
   */
  struct Ball
  {
    double radius;          /**< Radius of the ball. */
    exanb::Vec3d center;    /**< Center position of the ball. */
    exanb::Vec3d vel;       /**< Velocity of the ball. */
    exanb::Vec3d vrot;      /**< Angular velocity of the ball. */

    /**
     * @brief Get the type of the driver (in this case, BALL).
     * @return The type of the driver.
     */
    constexpr DRIVER_TYPE get_type() {return DRIVER_TYPE::BALL;}

    /**
     * @brief Print information about the ball.
     */
    void print()
    {
      std::cout << "Driver Type: Ball"  << std::endl;
      std::cout << "Radius: " << radius << std::endl;
      std::cout << "Center: " << center << std::endl;
      std::cout << "Vel   : " << vel    << std::endl;
      std::cout << "AngVel: " << vrot   << std::endl;
    }

    /**
     * @brief Initialize the ball.
     * @details This function asserts that the radius of the ball is greater than 0.
     */
    inline void initialize ()
    {
      assert (radius > 0 );
    }

    /**
     * @brief Initialize the ball.
     * @details This function asserts that the radius of the ball is greater than 0.
     */
    inline void update_radius (const double incr)
    {
      radius += incr;
    }

    /**
     * @brief Update the position of the ball.
     * @param t The time step.
     */
    inline void update_position ( const double t )
    {
      center = center + t * vel; 
    }

    /**
     * @brief Filter function to check if a point is within a certain radius of the ball.
     * @param rcut The cut-off radius.
     * @param p The point to check.
     * @return True if the point is within the cut-off radius of the ball, false otherwise.
     */
    inline bool filter( const double rcut , const exanb::Vec3d& p)
    {
      const Vec3d dist = center - p;
      double d = radius - norm ( dist );
      return std::fabs(d) <= rcut;
    }

    /**
     * @brief Detects collision between a vertex and the ball.
     * @param rcut The cut-off radius.
     * @param p The point to check for collision.
     * @return A tuple containing:
     *         - A boolean indicating whether a collision occurred.
     *         - The penetration depth (negative if inside the ball).
     *         - The normal vector pointing from the collision point to the center of the ball.
     *         - The contact position on the surface of the ball.
     */
    inline std::tuple<bool, double, Vec3d, Vec3d> detector( const double rcut , const Vec3d& p)
    {
      Vec3d point_to_center = center - p;
      double d = norm ( point_to_center );
      double dn; 
      Vec3d n = point_to_center / d;
      if ( d > radius )
      { 
        dn = d - radius - rcut;
        n = (-1) * n;
      }
      else dn = radius - d - rcut;

      if( dn > 0 )
      {
        return {false, 0.0, Vec3d(), Vec3d()};
      }
      else
      {
        Vec3d contact_position = p - n * ( rcut + 0.5 * dn ); 
        return {true, dn, n, contact_position};
      }
    }
  };
}
