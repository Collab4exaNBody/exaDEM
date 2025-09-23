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

#include <exaDEM/color_log.hpp>
#include <exaDEM/normalize.hpp>
#include <climits>
#include <chrono>
#include <thread>
#include <onika/physics/units.h>


namespace exaDEM
{
  using namespace exanb;


  enum MotionType
  {
    STATIONARY,                /**< Stationary state, with no motion. */
    LINEAR_MOTION,             /**< Linear movement type, straight-line motion. */
    COMPRESSIVE_FORCE,         /**< Movement influenced by compressive forces. */
    LINEAR_FORCE_MOTION,       /**< Linear motion type influenced by applied forces. */
    FORCE_MOTION,              /**< General movement caused by applied forces. */
    LINEAR_COMPRESSIVE_MOTION, /**< Linear movement combined with compressive forces. */
    TABULATED,                 /**< Motion defined by precomputed or tabulated data. */
    SHAKER,                    /**< Oscillatory or vibratory motion, typically mimicking a shaking mechanism. */
    PENDULUM_MOTION,           /**< Oscillatory swinging around a suspension point (pendulum-like). */
    UNKNOWN
  };


  inline std::string motion_type_to_string(MotionType motion_type)
  {
    switch(motion_type)
    {
      case STATIONARY: return "STATIONARY";
      case LINEAR_MOTION: return "LINEAR_MOTION";
      case COMPRESSIVE_FORCE: return "COMPRESSIVE_FORCE";
      case LINEAR_FORCE_MOTION: return "LINEAR_FORCE_MOTION";
      case FORCE_MOTION: return "FORCE_MOTION";
      case LINEAR_COMPRESSIVE_MOTION: return "LINEAR_COMPRESSIVE_MOTION";
      case TABULATED: return "TABULATED";
      case SHAKER: return "SHAKER";
      case PENDULUM_MOTION: return "PENDULUM_MOTION";
      default: return "UNKNOWN";
    }
  }

  inline MotionType string_to_motion_type(const std::string& str)
  {
    if (str == "STATIONARY") return STATIONARY;
    if (str == "LINEAR_MOTION") return LINEAR_MOTION;
    if (str == "COMPRESSIVE_FORCE") return COMPRESSIVE_FORCE;
    if (str == "LINEAR_FORCE_MOTION") return LINEAR_FORCE_MOTION;
    if (str == "FORCE_MOTION") return FORCE_MOTION;
    if (str == "LINEAR_COMPRESSIVE_MOTION") return LINEAR_COMPRESSIVE_MOTION;
    if (str == "TABULATED") return TABULATED;
    if (str == "SHAKER") return SHAKER;
    if (str == "PENDULUM_MOTION") return PENDULUM_MOTION;

    // If the string doesn't match any valid MotionType, return a default value
    return UNKNOWN;  // Or some other default action like throwing an exception or logging
  }

  struct Driver_params
  {
    // Common motion stuff
    MotionType motion_type = STATIONARY;
    Vec3d motion_vector = {0,0,0};
    double motion_start_threshold = 0;
    double motion_end_threshold = 1e300;

    // Motion: Linear
    double const_vel = 0;
    double const_force = 0;

    // Motion: Compression
    double sigma = 0;       /**< used for compressive force */
    double damprate = 0;    /**< used for compressive force */
    Vec3d forces = {0,0,0}; /**< sum of the forces applied to the driver. */
    double weigth = 0;     /**< cumulated sum of particle weigth into the simulation or in the driver */

    // Motion: Tabulated
    std::vector<double> tab_time;
    std::vector<Vec3d> tab_pos;

    // Motion: Shaker
    double omega = 0;
    double amplitude = 0;
    Vec3d shaker_dir = Vec3d(0,0,1);

    // Motion: Pendulum (re-use both shaker motion members omega and amplitude)
		Vec3d pendulum_anchor_point;     /**< Fixed suspension point. */
		Vec3d pendulum_initial_position; /**< Starting position of the pendulum mass. */
		Vec3d pendulum_swing_dir;        /**< Direction defining the pendulum's oscillation plane. */

		inline bool is_stationary() const { return motion_type == STATIONARY; }
		inline bool is_tabulated() const { return motion_type == TABULATED; }
		inline bool is_shaker() const { return motion_type == SHAKER; }
		inline bool is_pendulum() const { return motion_type == PENDULUM_MOTION; }

		void set_params(Driver_params& in)
		{ 
			(*this) = in;
		}

		ONIKA_HOST_DEVICE_FUNC inline bool is_linear() const
		{
			return (motion_type == LINEAR_MOTION || motion_type == LINEAR_FORCE_MOTION || motion_type == LINEAR_COMPRESSIVE_MOTION);
		}

		ONIKA_HOST_DEVICE_FUNC inline bool is_compressive() const
		{
			return (motion_type == COMPRESSIVE_FORCE || motion_type == LINEAR_COMPRESSIVE_MOTION);
		}

		ONIKA_HOST_DEVICE_FUNC inline bool is_force_motion() const
		{
			return ( motion_type == FORCE_MOTION || motion_type == LINEAR_FORCE_MOTION );
		}

		ONIKA_HOST_DEVICE_FUNC inline bool need_forces() const
		{
			// Need for LINEAR_FORCE_MOTION
			// No need for STATIONARY
			// No need for LINEAR_MOTION
			// Need for FORCE_MOTION
			// Need for COMPRESSIVE_FORCE
			// Need for LINEAR_COMPRESSIVE_MOTION
			return is_compressive() || motion_type == FORCE_MOTION ||  motion_type == LINEAR_FORCE_MOTION; 
		}

		// Getter
		inline Vec3d sum_forces()
		{
			if( motion_type == FORCE_MOTION )
			{
				return forces;
			}
			if( motion_type == LINEAR_FORCE_MOTION )
			{
				forces = (exanb::dot(forces, motion_vector) + const_force) * motion_vector;
				return forces;
			}
			return Vec3d{0,0,0};
		}

		// Checks
		inline bool is_valid_motion_type(const std::vector<MotionType>& valid_motion_types) const
		{
			auto it = std::find(valid_motion_types.begin(), valid_motion_types.end(), motion_type);
			if( it == valid_motion_types.end() )
			{
				color_log::warning("Driver_params::is_valid_motion_type", "This motion type [" + motion_type_to_string(motion_type) + "] is not possible, MotionType availables are: ");
				for(const auto& motion: valid_motion_types)
				{
					lout << " " << ansi::yellow(motion_type_to_string(motion));
				}
				lout << std::endl;
				return false;
			}
			return true;
		}

		bool check_motion_coherence()
		{
			if( is_shaker() || is_pendulum() )
			{
				if( amplitude <= 0.0 ) 
				{
					color_log::warning("Driver_params::check_motion_coherence", "The \"amplitude\" input slot is not defined correctly."); 
					return false;
				}
				if( omega <= 0.0 ) 
				{
					color_log::warning("Driver_params::check_motion_coherence", "The \"omega\" input slot is not defined correctly."); 
					return false;
				}

        if( is_shaker())
        {
				  if( exanb::dot(shaker_dir, shaker_dir) - 1 >= 1e-14 ) 
				  {
				    Vec3d old = shaker_dir;
				    exanb::_normalize(shaker_dir);
				    color_log::warning("Driver_params::check_motion_coherence", "Your shaker_dir vector [" + std::to_string(old) + "} has been normalized to ["  + std::to_string(shaker_dir) + "]"); 
				  }
        }
        else if( is_pendulum())
        {
          if( pendulum_anchor_point == pendulum_initial_position )
          {
				    color_log::error("Driver_params::check_motion_coherence", "The point defined in pendulum_anchor_point and the one in pendulum_initial_position are the same. It is impossible to define a motion type PENDULUM_MOTION. Point: [" + std::to_string(pendulum_anchor_point) + "]"); 
          }
				  if( exanb::dot(pendulum_swing_dir, pendulum_swing_dir) - 1 >= 1e-14 ) 
				  {
				    Vec3d old = pendulum_swing_dir;
				    exanb::_normalize(pendulum_swing_dir);
				    color_log::warning("Driver_params::check_motion_coherence", "Your pendulum_swing_dir vector [" + std::to_string(old) + "} has been normalized to ["  + std::to_string(pendulum_swing_dir) + "]"); 
				  }
        }
			}
			if( is_tabulated() )
			{
				if(tab_time.size() == 0)
				{
					color_log::warning("Driver_params::check_motion_coherence", "The \"time\" input slot is not defined while the tabulated motion is activated."); 
					return false;
				}
				else if(tab_time[0] != 0.0)
				{
					color_log::warning("Driver_params::check_motion_coherence", "Please set the first element of your input time vector to 0."); 
					return false;
				}
				if(tab_pos.size() == 0)
				{
					color_log::warning("Driver_params::check_motion_coherence", "The \"positions\" input slot is not defined while the tabulated motion is activated."); 
					return false;
				}
				if(tab_time.size() != tab_pos.size())
				{
					color_log::warning("Driver_params::check_motion_coherence", "The \"positions\" and \"time\" input slot are not the same size."); 
					return false;
				}
				if(!is_sorted(tab_time.begin(), tab_time.end()))
				{
					color_log::warning("Driver_params::check_motion_coherence", "The \"time\" array used for a TABULATED motion is not sorted."); 
					return false;
				}
			}
			if( is_linear() )
			{
				// Check if motion vector is zero (invalid for linear motion)
				if( motion_vector == Vec3d{0,0,0} )
				{
					lout << ansi::yellow("Your motion type is a \"Linear Mode\" that requires a motion vector.") << std::endl;
					lout << ansi::yellow("Please, define motion vector by adding \"motion_vector: [1,0,0]. It is defined to [0,0,0] by default.") << std::endl;
					return false;
				}
				// Normalize motion vector if its magnitude is not equal to 1
				if( dot(motion_vector, motion_vector) - 1 >= 1e-14 )
				{
					Vec3d old = motion_vector;
					exanb::_normalize(motion_vector);
					color_log::warning("Driver_params::check_motion_coherence", "Your motion vector [" + std::to_string(old) + "} has been normalized to [" + std::to_string(motion_vector) + "]"); 
				}
				if( motion_type == LINEAR_MOTION && const_vel == 0 )
				{
					color_log::warning("Driver_params::check_motion_coherence", "You have chosen constant linear motion with zero velocity, please use \"const_vel\" or use the motion type \"STATIONARY\"."); 
				}
				if( motion_type == LINEAR_FORCE_MOTION && const_force == 0 )
				{
					color_log::warning("Driver_params::check_motion_coherence", "You have chosen constant linear force motion with zero force, please use \"const_force\" or use the motion type \"STATIONARY\"."); 
				}
			}

			if( is_compressive() )
			{
				if( sigma == 0 ) 
				{
					color_log::warning("Driver_params::check_motion_coherence", "Sigma is to 0.0 while the compressive motion type is set to true."); 
				}
				if( damprate <= 0 )
				{
					color_log::warning("Driver_params::check_motion_coherence", "Dumprate is to 0.0 while the compressive motion type is set to true."); 
				}
			}

			return true;  // Return true if the motion is coherent
		}

		bool is_motion_triggered(double time) const
		{
			return ( (time >= motion_start_threshold) && (time <= motion_end_threshold) );  
		}

		bool is_motion_triggered(uint64_t timesteps, double dt) const
		{
			const double time = timesteps * dt;
			return is_motion_triggered(time);
		}

		void tabulations_to_stream(std::stringstream& times, std::stringstream& positions) const
		{
			if( is_tabulated() )
			{
				times << "time: [";
				positions << "positions: [";

				assert(tab_time.size() == tab_pos.size());
				size_t last = tab_time.size() - 1;
				for(size_t i = 0; i < last ; i++)
				{
					times     << tab_time[i] << ",";
					positions << "[ " << tab_pos[i] << " ],";
				}
				times     << tab_time[last] << "]";
				positions << "[ " << tab_pos[last]  << " ]]";
			}
		}

		void print_driver_params() const
		{
			lout << "Motion type: " << motion_type_to_string(motion_type) << std::endl;

			if( is_tabulated() )
			{
				std::stringstream times;
				std::stringstream positions;
				tabulations_to_stream(times, positions);
				lout << times.rdbuf() << std::endl;
				lout << positions.rdbuf() << std::endl;
			}

			if( !is_stationary() )
			{
				if( motion_start_threshold != 0 || motion_end_threshold != std::numeric_limits<double>::max() )
				{
					if( motion_end_threshold != std::numeric_limits<double>::max() )
					{
						lout << "Motion duration: [ " << motion_start_threshold << "s , " <<motion_end_threshold << "s ]" << std::endl;
					}
					else
					{
						lout << "Motion duration: [ " << motion_start_threshold << "s ,  inf s )" << std::endl;
					}
				}
				if( is_linear() )
				{
					lout << "Motion vector: " << motion_vector << std::endl;
					if( motion_type == LINEAR_MOTION)
					{ 
						lout << "Velocity (constant): " << const_vel << std::endl;
					}
					if(motion_type == LINEAR_FORCE_MOTION)
					{
						lout << "Force (constant): " << const_force << std::endl;
					}
				}
				if (is_compressive() )
				{
					lout << "Sigma: " << sigma << std::endl;
					lout << "Damprate: " << damprate << std::endl;
				}
			}

			if( is_shaker() )
			{
				lout << "Shaker.Omega: "     << omega << std::endl;
				lout << "Shaker.Amplitude: " << amplitude << std::endl;
				lout << "Shaker.Direction: [" << shaker_dir << "]" << std::endl;
			}
		};

		/**
		 * @brief Write ball data into a stream.
		 */
		void dump_driver_params(std::stringstream &stream)
		{
			stream << "     params: {";
			stream << " motion_type: "            << motion_type_to_string(motion_type);
			stream << ", motion_vector: ["         << motion_vector << "]"; 
			stream << ", motion_start_threshold: " << motion_start_threshold;
			stream << ", motion_end_threshold: "   << motion_end_threshold;
			if(motion_type == LINEAR_MOTION)
			{
				stream << ", const_vel: "   << const_vel;
			}
			if(motion_type == LINEAR_FORCE_MOTION)
			{
				stream << ", const_force: " << const_force; 
			}
			if( is_compressive() )
			{
				stream << ", sigma: " << sigma;
				stream << ", damprate: " << damprate;
			}
			if(motion_type == TABULATED)
			{
				std::stringstream times;
				std::stringstream positions;
				tabulations_to_stream(times, positions);
				stream << ", " << times.rdbuf() << ", " << positions.rdbuf();
			}
			if(motion_type == SHAKER)
			{
				stream << ", omega: " << omega;
				stream << ", amplitude: " << amplitude;
				stream << ", shaker_dir: [" << shaker_dir << "]";
			}
      if(motion_type == PENDULUM_MOTION)
      {
				stream << ", omega: " << omega;
				stream << ", amplitude: " << amplitude;
				stream << ", pendulum_anchor_point: [" << pendulum_anchor_point << "]";
				stream << ", pendulum_initial_position: [" << pendulum_initial_position << "]";
				stream << ", pendulum_swing_dir: [" << pendulum_swing_dir << "]";
      }
			stream  <<" }" << std::endl;
		}

		/* Tabulated motion routines */
		Vec3d tab_to_position(double time)
		{
			assert(time >= 0.0);
			auto ite = std::lower_bound(tab_time.begin(), tab_time.end(), time);
			if(ite == tab_time.end())
			{
				return tab_pos.back();
			}
			else
			{
				size_t idx_lower = ite - tab_time.begin() - 1;
				size_t idx_upper = idx_lower + 1;
				assert(tab_time[idx_lower] >= time);
				if(idx_upper >= tab_time.size()) return tab_pos.back();
				double Dt = (time - tab_time[idx_lower]) / (tab_time[idx_upper] - tab_time[idx_lower]);
				Vec3d P = (tab_pos[idx_upper] - tab_pos[idx_lower]) * Dt + tab_pos[idx_lower];
				return P;
			}
		}

		Vec3d tab_to_velocity(double time)
		{
			assert(time >= 0.0);
			auto ite = std::lower_bound(tab_time.begin(), tab_time.end(), time);
			if(ite == tab_time.end())
			{
				return Vec3d(0,0,0); // stationnary
			}
			else
			{
				size_t idx_lower = ite - tab_time.begin() - 1;
				size_t idx_upper = idx_lower + 1;
				assert(tab_time[idx_lower] >= time);
				if(idx_upper >= tab_time.size()) return Vec3d(0,0,0);
				double Dt = tab_time[idx_upper] - tab_time[idx_lower];
				assert(Dt != 0.0);
				Vec3d V = (tab_pos[idx_upper] - tab_pos[idx_lower])/Dt;
				return V;
			}
		}

		/* Shaker routines */
		Vec3d shaker_direction()
		{
			return shaker_dir;
		}

		double shaker_signal(double time)
		{
			assert(motion_start_threshold >= 0);
			time -= motion_start_threshold;
			return amplitude * sin(omega * time);
		}

		Vec3d shaker_velocity(double time)
		{
			assert(motion_start_threshold >= 0);
			time -= motion_start_threshold;
			return amplitude * omega * cos(omega * time) * shaker_direction();
		}

		/* Pendulum routines */
		Vec3d pendulum_direction()
		{
			return pendulum_swing_dir;
		}

		Vec3d pendulum_velocity(double time)
		{
      return {0,0,0};
		}

    std::pair<double,  Vec3d> compute_offset_normal_pendulum_motion(double time)
    {
      if( time <  motion_start_threshold ) color_log::error("compute_normal_pendulum_motion", "This call is ill-formed, please verify that time is superior to motion_start_threshold.");
      Vec3d v1 = pendulum_anchor_point;
      Vec3d v2 = pendulum_initial_position;
      Vec3d v3 = pendulum_initial_position + pendulum_direction() * pendulum_signal(time);

      // warning, if v2 = v3, we return an offset of  0 and the pendulum direction
      if( exanb::dot(v3,v2) < 1e-16) return {0.0, pendulum_direction()};

      Vec3d v1v3 = v3 - v1;
      v1v3 = v1v3 / exanb::norm(v1v3);
      Vec3d project_v2_v1v3 = exanb::dot(v1v3, v2-v1) * v1v3 + v1;
      Vec3d dir_proj_v2_v2 = project_v2_v1v3 - v2;
      Vec3d normal = dir_proj_v2_v2 / exanb::norm(dir_proj_v2_v2);
      double offset = exanb::dot(v1, normal);
      return {offset, normal};
    }

		double pendulum_signal(double time)
		{
			assert(motion_start_threshold >= 0);
			time -= motion_start_threshold;
			return amplitude * sin(omega * time);
		}
	};
}


namespace YAML
{
	using exaDEM::Driver_params;
	using exaDEM::MotionType;
	using onika::physics::Quantity;

	template <> struct convert<Driver_params>
	{
		static bool decode(const Node &node, Driver_params &v)
		{
			std::string function_name = "Driver_params::decode";
			if (!node.IsMap())
			{
				return false;
			}
			if (!node["motion_type"])
			{
				color_log::error(function_name, "mmotion_type is missing.", false);
				return false;
			}

			v = {};
			v.motion_type = exaDEM::string_to_motion_type(node["motion_type"].as<std::string>());
			if( v.is_linear() )
			{
				if (!node["motion_vector"])
				{
					color_log::error(function_name, "motion_vector is missing.", false);
					return false;
				}
				v.motion_vector = node["motion_vector"].as<Vec3d>();

				if( v.motion_type == MotionType::LINEAR_MOTION )
				{
					if (!node["const_vel"])
					{
						color_log::error(function_name, "const_vel is missing.", false);
						return false;
					}
					if( node["const_vel"] ) v.const_vel = node["const_vel"].as<double>();
				}
				if( v.motion_type == MotionType::LINEAR_FORCE_MOTION )
				{
					if (!node["const_force"])
					{
						color_log::error(function_name, "const_force is missing.", false);
						return false;
					}
					if( node["const_force"] ) v.const_force = node["const_force"].as<double>();
				}
			}
			if( v.is_compressive() )
			{ 
				if (!node["sigma"])
				{
					color_log::error(function_name, "sigma is missing.", false);
					return false;
				}
				v.sigma = node["sigma"].as<double>(); 
				if (!node["damprate"])
				{
					color_log::error(function_name, "damprate is missing.", false);
					return false;
				}
				v.damprate = node["damprate"].as<double>(); 
			}
			// Tabulation
			if( v.is_tabulated() )
			{ 
				if (!node["time"])
				{
					color_log::error(function_name, "time is missing.", false);
					return false;
				}
				v.tab_time= node["time"].as<std::vector<double>>(); 
				if (!node["positions"])
				{
					color_log::error(function_name, "position is missing.", false);
					return false;
				}
				v.tab_pos = node["positions"].as<std::vector<Vec3d>>(); 
			}

			// Shaker && Pendulum
			if( v.is_shaker() || v.is_pendulum() )
			{
				if (!node["omega"])
				{
					color_log::error(function_name, "omega is missing.", false);
					return false;
				}
				v.omega = node["omega"].as<double>();
				if (!node["amplitude"])
				{
					color_log::error(function_name, "amplitude is missing.", false);
					return false;
				}
				v.amplitude = node["amplitude"].as<double>();

        if( v.is_shaker() )
        {
				  if (!node["shaker_dir"])
				  {
				  	color_log::warning("Driver_params::decode", "shaker_dir is missing, default is [0,0,1].");
				  	v.shaker_dir = Vec3d{0,0,1};
				  }
				  else
				  {
					  v.shaker_dir = node["shaker_dir"].as<Vec3d>();
				  }
        }
        else if( v.is_pendulum() )
        {
				  if (!node["pendulum_anchor_point"])
				  {
					  color_log::error(function_name, "pendulum_anchor_point is missing.", false);
					  return false;
			  	}
				  v.pendulum_anchor_point = node["pendulum_anchor_point"].as<Vec3d>();
				  if (!node["pendulum_initial_position"])
				  {
					  color_log::error(function_name, "pendulum_initial_position is missing.", false);
					  return false;
			  	}
				  v.pendulum_initial_position = node["pendulum_initial_position"].as<Vec3d>();
				  if (!node["pendulum_swing_dir"])
				  {
					  color_log::error(function_name, "pendulum_swing_dir is missing.", false);
					  return false;
			  	}
				  v.pendulum_swing_dir = node["pendulum_swing_dir"].as<Vec3d>();
        }
			}

			if( node["motion_start_threshold"] ) v.motion_start_threshold = node["motion_start_threshold"].as<double>();
			if( node["motion_end_threshold"] ) v.motion_end_threshold = node["motion_end_threshold"].as<double>();
			return true;
		}
	};
}
