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
#include <onika/physics/units.h>

namespace exaDEM
{
	using namespace exanb;
	struct Surface_params
	{
		/** Required */
		double offset = 0;                /**< Offset from the origin along the normal vector. */
		exanb::Vec3d normal = {0,0,1};    /**< Normal vector of the surface. */
		exanb::Vec3d center = {0,0,0};    /**< Center position of the surface. */
		/** optional */
		Vec3d vel = {0, 0, 0};                   /**< Velocity of the surface. */
		exanb::Vec3d vrot = Vec3d{0,0,0}; /**< Angular velocity of the surface. */
		double mass = std::numeric_limits<double>::max() / 4; /**< Mass of the ball */
		double surface = -1;
		/** no need to dump them */
		exanb::Vec3d center_proj;         /**< Center position projected on the norm. */
		double acc = 0;
	};
}

namespace YAML
{
	using exaDEM::Surface_params;
	using exaDEM::MotionType;
	using exanb::lerr;
	using onika::physics::Quantity;

	template <> struct convert<Surface_params>
	{
		static bool decode(const Node &node, Surface_params &v)
		{
			if (!node.IsMap())
			{
				return false;
			}
			if( !check_error(node, "offset") ) return false;
			if( !check_error(node, "normal") ) return false;
			v.offset = node["offset"].as<Quantity>().convert();
			v.normal = node["normal"].as<Vec3d>();
			if( check(node, "vel") ) { v.vel = node["vel"].as<Vec3d>(); }
			if( check(node, "vrot") ) { v.vrot = node["vrot"].as<Vec3d>(); }
			if( check(node, "mass") ) { v.mass = node["mass"].as<Quantity>().convert(); }
			if( check(node, "surface") ) { v.surface = node["surface"].as<double>(); }
			if( v.vrot != Vec3d{0,0,0} )
			{ 
				if( !check_error(node, "center") ) return false;
				v.center = node["center"].as<Vec3d>();
			}
			else
			{
				if( check(node, "center") ) 
				{ 
					v.center = node["center"].as<Vec3d>(); 
				}
				else
				{
					v.center = v.offset * v.normal;
				}
			}
			return true;
		}
	};
}

namespace exaDEM
{
	using namespace exanb;

	const std::vector<MotionType> surface_valid_motion_types = { STATIONARY, LINEAR_MOTION, LINEAR_COMPRESSIVE_MOTION, SHAKER};

	/**
	 * @brief Struct representing a surface in the exaDEM simulation.
	 */
	struct Surface : public Surface_params, Driver_params
	{
		/*
			 Surface(Surface_params& bp, Driver_params& dp) : Surface_params{bp}, Driver_params()
			 {
			 Driver_params::set_params(dp);
			 }
		 */
		/**
		 * @brief Get the type of the driver (in this case, SURFACE).
		 * @return The type of the driver.
		 */
		constexpr DRIVER_TYPE get_type() { return DRIVER_TYPE::SURFACE; }

		/**
		 * @brief Print information about the surface.
		 */
		inline void print() const
		{
			lout << "Driver Type: Surface" << std::endl;
			lout << "Offset: " << offset << std::endl;
			lout << "Normal: " << normal << std::endl;
			lout << "Center: " << center << std::endl;
			lout << "Vel   : " << vel << std::endl;
			lout << "AngVel: " << vrot << std::endl;
			if ( is_compressive() )
			{
				lout << "Acceleration: "<< acc << std::endl;
				lout << "Surface Value [>0]: "<< surface << std::endl;
			}
			Driver_params::print_driver_params();
		}

		/**
		 * @brief Write surface data into a stream.
		 */
		inline void dump_driver(int id, std::stringstream &stream)
		{
			stream << "  - register_surface:" << std::endl;
			stream << "     id: " << id << std::endl;
			stream << "     state: {offset: " << this->offset;;
			stream << ", center: [" << this->center << "]";
			stream << ", normal: [" << this->normal << "]";;
			stream << ", vel: [" << this->vel << "]"; 
			stream << ", vrot: [" << this->vrot << "]";
			stream << ", surface: " << surface;
			stream << "}" << std::endl;
			Driver_params::dump_driver_params(stream);
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
				lout << "[register_surface, WARNING] The Center point (surface) is not correctly defined" << std::endl;

			if (exanb::dot(center, normal) != exanb::dot(center_proj, normal))
			{
				center += (offset - exanb::dot(center, normal)) * normal;
				lout << "center is re-computed because it doesn't fit with offset, new center is: " << center << " and center_proj is: " << center_proj << std::endl;
			}
			if( !Driver_params::is_valid_motion_type(surface_valid_motion_types)) std::exit(EXIT_FAILURE);
			if( !Driver_params::check_motion_coherence()) std::exit(EXIT_FAILURE);
			if( mass <= 0.0 )
			{
				lout << "[register_surface, ERROR] Please, define a positive mass." << std::endl;
				std::exit(EXIT_FAILURE);
			}
			if( is_linear() )
			{
				if( normal != motion_vector )
				{
					lout << "\033[32m[register_surface, WARNING] The motion vector of the surface has been adjusted to align with the normal vector, i.e. the motion vecor[" << motion_vector<<"] is now equal to ["<<normal<<"].\033[0m" <<std::endl;
					motion_vector = normal;
				}
			}
			if( is_compressive() )
			{
				if( surface <= 0 )
				{
					lout << "\033[31m[register_surface, ERROR] The surface value must be positive for LINEAR_COMPRESSIVE_FORCE. You need to specify surface: XX in the 'state' slot.\033[0m" << std::endl; 
					std::exit(EXIT_FAILURE);
				}
			}
		}


		inline void force_to_accel() 
		{
			if( is_compressive() )
			{ 
				constexpr double C = 0.5;
				if( weigth != 0 )
				{
					const double s = surface;
					acc = (exanb::norm(forces) - sigma * s - (damprate * exanb::norm(vel)) ) / (weigth * C);
				}
				else
				{
					acc = 0; 
				}
			}
		}

		inline void push_f_v(const double dt) 
		{
			if ( is_stationary() )
			{
				vel =  {0,0,0};
			}
			else
			{
				if( is_compressive() )
				{
					if( this->sigma != 0 ) vel += 0.5 * dt * acc * normal; 
				}
				if( motion_type == LINEAR_MOTION )
				{
					vel = this->const_vel * this->motion_vector; // I prefere reset it 
				}
			}
		}

		/**
		 * @brief return driver velocity
		 */
		ONIKA_HOST_DEVICE_FUNC inline const Vec3d& get_vel() const 
		{
			return vel; 
		}

		/**
		 * @brief Update the position of the wall.
		 * @param time Current physical time.
		 * @param dt The time step.
		 */
		inline void push_f_v_r(const double time, const double dt)
		{
			if( !is_stationary() )
			{
				if( motion_type == LINEAR_MOTION )
				{
					assert( vel == this->const_vel * this->motion_vector );  
				}

				double displ = dt * exanb::norm(vel) + 0.5 * dt * dt * acc;

				/** The shaker motion changes the displacement behavior */
				/** the shaker direction vector is ignored, the normal vector is used */
				if( motion_type == SHAKER )
				{
					double signal_next = shaker_signal(time + dt);
					double signal_current = shaker_signal(time);
          const double angle_factor = exanb::dot(shaker_direction(), normal); 
					displ = (signal_next - signal_current) * angle_factor;           
					vel = shaker_velocity(time + dt);
				}

				center += displ * normal;
				offset += displ; 
				center_proj +=  displ * normal;
			}
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


namespace onika { namespace memory
	{

		template<>
			struct MemoryUsage< exaDEM::Surface >
			{
				static inline size_t memory_bytes(const exaDEM::Surface& obj)
				{
					const exaDEM::Surface_params * cparms = &obj;
					const exaDEM::Driver_params * dparms = &obj;
					return onika::memory::memory_bytes( *cparms , *dparms );
				}
			};

	} }


