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
#include <exaDEM/driver_base.h>


namespace exaDEM
{
  using namespace exanb;
  struct Ball_params
  {
    double radius;                    /**< Radius of the ball. */
    exanb::Vec3d center;              /**< Center position of the ball. */
    exanb::Vec3d vel = Vec3d{0,0,0};  /**< Velocity of the ball. */
    exanb::Vec3d vrot = Vec3d{0,0,0}; /**< Angular velocity of the ball. */
    double mass = std::numeric_limits<double>::max() / 4; /**< Mass of the ball */
    double rv = 0;                    /**< */

    /** We don't need to save these values */
    exanb::Vec3d acc = {0,0,0};       /**< Acceleration of the ball. */
    double ra = 0;                    /**< */

    /**
     * @brief Compute the surface.
     */
    double volume()
    {
      const double pi = 4*atan(1);
      return 4/3 * pi * radius * radius * radius;
    }
  };
}

namespace YAML
{
  using exaDEM::Ball_params;
  using exaDEM::MotionType;
  using exanb::lerr;
  using exanb::Quantity;
  using exanb::UnityConverterHelper;

  template <> struct convert<Ball_params>
  {
    static bool decode(const Node &node, Ball_params &v)
    {
      if (!node.IsMap())
      {
        return false;
      }
      if( !check_error(node, "radius") ) return false;
      if( !check_error(node, "center") ) return false;
      v.radius = node["radius"].as<Quantity>().convert();
      v.center = node["center"].as<Vec3d>();
      if( check(node, "vel") ) { v.vel = node["vel"].as<Vec3d>(); }
      if( check(node, "vrot") ) { v.vrot = node["vrot"].as<Vec3d>(); }
      if( check(node, "rv") ) { v.rv= node["rv"].as<double>(); }
      if( check(node, "mass") ) { v.mass = node["mass"].as<double>(); }
      if( check(node, "density") && !check(node, "mass") ) { v.mass = v.volume()*node["density"].as<double>();; }
      return true;
    }
  };
}


namespace exaDEM
{
  using namespace exanb;

  const std::vector<MotionType> ball_valid_motion_types = { STATIONARY, LINEAR_MOTION, COMPRESSIVE_FORCE};


  /**
   * @brief Struct representing a ball in the exaDEM simulation.
   */
  struct Ball : public Ball_params, Driver_params
  {
    Ball(Ball_params& bp, Driver_params& dp) : Ball_params{bp}, Driver_params() 
    { 
      Driver_params::set_params(dp);
    }

    /**
     * @brief Get the type of the driver (in this case, BALL).
     * @return The type of the driver.
     */
    constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::BALL; }

    /**
     * @brief Print information about the ball.
     */
    void print()
    {
      lout << "Driver Type: Ball" << std::endl;
      lout << "Radius: " << radius << std::endl;
      lout << "Center: " << center << std::endl;
      lout << "Vel   : " << vel << std::endl;
      lout << "AngVel: " << vrot << std::endl;
      if ( is_compressive() )
      {
        lout << "Radius acceleration: "<< ra << std::endl;
        lout << "Radius velocity: "<< rv << std::endl;
      }
      if ( is_force_motion() )
      {
        lout << "Mass: " << this->mass << std::endl;
      }
      Driver_params::print_driver_params();
    }

    /**
     * @brief Write ball data into a stream.
     */
    void dump_driver(int id, std::stringstream &stream)
    {
      stream << "  - register_ball:" << std::endl;
      stream << "     id: " << id << std::endl;
      stream << "     state: { radius:" << this->radius;
      stream << ",center: [" << this->center << "]";
      stream << ",vel: [" << this->vel << "]";
      stream << ",vrot: [" << this->vrot << "]";
      if ( is_compressive() )
      {
        stream << ",rv: " << this->rv;
      }
      if ( is_force_motion() )
      {
        stream << ",mass: " << this->mass;
      }
      stream <<"}" << std::endl;
      Driver_params::dump_driver_params(stream);
    }

    /**
     * @brief Initialize the ball.
     * @details This function asserts that the radius of the ball is greater than 0.
     */
    inline void initialize() 
    {
      if( !Driver_params::is_valid_motion_type(ball_valid_motion_types)) std::exit(EXIT_FAILURE);
      if( !Driver_params::check_motion_coherence()) std::exit(EXIT_FAILURE);
      if( mass <= 0.0 ) 
      {
        lout << "Please, define a positive mass." << std::endl;
        std::exit(EXIT_FAILURE);
      }
      assert(radius > 0); 
    }

    /**
     * @brief Update the position of the ball.
     * @param dt The time step.
     */
    inline void force_to_accel()
    {
      if( is_force_motion() )
      {
        if( mass >= 1e100 ) lout << "Warning, the mass of the ball is set to " << mass << std::endl;
        acc = Driver_params::sum_forces() / mass; 
      }
      else
      {
        acc = {0,0,0};
      }
    }

    /**
     * @brief return driver velocity
     */
    ONIKA_HOST_DEVICE_FUNC inline Vec3d &get_vel() { return vel; }

    /**
     * @brief Update the position of the ball.
     * @param dt The time step.
     */
    inline void push_f_v_r(const double dt) 
    {
      if( !is_stationary() )
      {
        if( is_compressive() )
        {
          push_ra_rv_to_rad(dt);
        }
        center = center + dt * vel; 
      }
    }

    /**
     * @brief Update the position of the ball.
     * @param dt The time step.
     */
    inline void push_f_v(const double dt)
    {
			if( is_force_motion() )
			{
				vel = acc * dt;
			}

			if( motion_type == LINEAR_MOTION )
			{
				vel = motion_vector * const_vel; // I prefere reset it 
			}

			if( is_compressive() )
			{
				if( motion_type == COMPRESSIVE_FORCE )
				{
					vel = {0,0,0};
				}
				push_ra_to_rv(dt);
			}
		}
		/**
		 * @brief Update the "velocity raduis" of the ball.
		 * @param t The time step.
		 */
		inline void push_ra_to_rv(const double dt) 
		{
			if( is_compressive() )
			{
				if( sigma != 0 ) rv += 0.5 * dt * ra;
			}
		}

		/**
		 * @brief Update the "velocity raduis" of the ball.
		 * @param t The time step.
		 */
		inline void push_ra_rv_to_rad(const double dt) 
		{
			if( is_compressive() )
			{
				radius += dt * rv + 0.5 * dt * dt * ra;
			}
		}

		/**
		 * @brief Compute the surface.
		 */
		ONIKA_HOST_DEVICE_FUNC inline 
			double surface()
			{
				const double pi = 4*atan(1);
				return 4 * pi * radius * radius;
			}

		/**
		 * @brief Update the "velocity radius" of the ball.
		 * @param t The time step.
		 */
		inline void f_ra(const double dt) 
		{
			if( is_compressive() )
			{
				constexpr double C = 0.5; // I don't remember why, ask Lhassan
				if( weigth != 0 ) 
				{
					const double s = surface();
					// forces and weigth are defined in Driver_params
					ra = ( exanb::norm(forces) - sigma * s - (damprate * rv) ) / (weigth * C); 
				}
			}
		}


		/**
		 * @brief Filter function to check if a point is within a certain radius of the ball.
		 * @param rcut The cut-off radius.
		 * @param p The point to check.
		 * @return True if the point is within the cut-off radius of the ball, false otherwise.
		 */
		ONIKA_HOST_DEVICE_FUNC inline bool filter(const double rcut, const exanb::Vec3d &p)
		{
			const Vec3d dist = center - p;
			double d = radius - norm(dist);
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
		ONIKA_HOST_DEVICE_FUNC inline std::tuple<bool, double, Vec3d, Vec3d> detector(const double rcut, const Vec3d &p)
		{
			Vec3d point_to_center = center - p;
			double d = norm(point_to_center);
			double dn;
			Vec3d n = point_to_center / d;
			if (d > radius)
			{
				dn = d - radius - rcut;
				n = (-1) * n;
			}
			else
				dn = radius - d - rcut;

			if (dn > 0)
			{
				return {false, 0.0, Vec3d(), Vec3d()};
			}
			else
			{
				Vec3d contact_position = p - n * (rcut + 0.5 * dn);
				return {true, dn, n, contact_position};
			}
		}
	};
} // namespace exaDEM
