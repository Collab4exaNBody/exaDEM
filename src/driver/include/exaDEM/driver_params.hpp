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

    // If the string doesn't match any valid MotionType, return a default value
    return UNKNOWN;  // Or some other default action like throwing an exception or logging
  }

  struct Driver_params
  {
    // motio, part
    MotionType motion_type = STATIONARY;
    Vec3d motion_vector = {0,0,0};
    double motion_start_threshold = 0;
    double motion_end_threshold = std::numeric_limits<double>::max();

    // Motion: Linear
    double const_vel = 0;
    double const_force = 0;

    // Motion: Compression
    double sigma = 0;       /**< used for compressive force */
    double damprate = 0;    /**< used for compressive force */
    Vec3d forces = {0,0,0}; /**< sum of the forces applied to the driver. */
    double weigth = 0;     /**< cumulated sum of particle weigth into the simulation or in the driver */

    inline bool is_stationary() { return motion_type == STATIONARY; }

    void set_params(Driver_params& in)
    { 
      (*this) = in;
    }

    ONIKA_HOST_DEVICE_FUNC inline bool is_linear()
    {
      return (motion_type == LINEAR_MOTION || motion_type == LINEAR_FORCE_MOTION || motion_type == LINEAR_COMPRESSIVE_MOTION);
    }

    ONIKA_HOST_DEVICE_FUNC inline bool is_compressive()
    {
      return (motion_type == COMPRESSIVE_FORCE || motion_type == LINEAR_COMPRESSIVE_MOTION);
    }

    ONIKA_HOST_DEVICE_FUNC inline bool is_force_motion()
    {
      return ( motion_type == FORCE_MOTION || motion_type == LINEAR_FORCE_MOTION );
    }

    ONIKA_HOST_DEVICE_FUNC inline bool need_forces()
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
    Vec3d sum_forces()
    {
      if( motion_type == FORCE_MOTION )
      {
        return forces;
      }
      if( motion_type == LINEAR_FORCE_MOTION )
      {
        //lout <<  " avant " << forces <<std::endl;
        forces = (exanb::dot(forces, motion_vector) + const_force) * motion_vector;
        //lout <<  " apres " << forces <<std::endl;
        return forces;
      }
      return Vec3d{0,0,0};
    }

    // Checks
    bool is_valid_motion_type(const std::vector<MotionType>& valid_motion_types)
    {
      auto it = std::find(valid_motion_types.begin(), valid_motion_types.end(), motion_type);
      if( it == valid_motion_types.end() )
      {
        lout << "\033[31mThis motion type [" << motion_type_to_string(motion_type) << "] is not possible, MotionType availables are: ";
        for(auto& motion: valid_motion_types)
        {
          lout << " " << motion_type_to_string(motion);
        }
        lout  << "\033[0m" << std::endl;
        return false;
      }
      return true;
    }

    bool check_motion_coherence()
    {
      if( is_linear() )
      {
        // Check if motion vector is zero (invalid for linear motion)
        if( motion_vector == Vec3d{0,0,0} )
        {
          lout << "\033[31mYour motion type is a \"Linear Mode\" that requires a motion vector." << std::endl;
          lout << "\033[31mPlease, define motion vector by adding \"motion_vector: [1,0,0]. It is defined to [0,0,0] by default.\033[0m" << std::endl;
          return false;
        }
        // Normalize motion vector if its magnitude is not equal to 1
        if( dot(motion_vector, motion_vector) - 1 >= 1e-14 )
        {
          Vec3d old = motion_vector;
          exanb::_normalize(motion_vector);
          lout << "\033[31m[Warning] Your motion vector [" << old <<"} has been normalized to [" << motion_vector << "]\033[0m" << std::endl;
        }
        if( motion_type == LINEAR_MOTION && const_vel == 0 )
        {
          lout << "\033[31m[Warning] You have chosen constant linear motion with zero velocity, please use \"const_vel\" or use the motion type \"STATIONARY\"\033[0m" << std::endl;
        }
        if( motion_type == LINEAR_FORCE_MOTION && const_force == 0 )
        {
          lout << "\033[31m[Warning] You have chosen constant linear force motion with zero force, please use \"const_force\" or use the motion type \"STATIONARY\"\033[0m" << std::endl;
        }
      }

      if( is_compressive() )
      {
        if( sigma == 0 ) 
        {
          lout << "\033[31m[Warning] Sigma is to 0.0 while the compressive motion type is set to true.\033[0m" << std::endl;
        }
        if( damprate <= 0 )
        {
          lout << "\033[31m[Warning] Dumprate is to 0.0 while the compressive motion type is set to true.\033[0m" << std::endl;
        }
      }

      return true;  // Return true if the motion is coherent
    }

    bool is_motion_triggered(uint64_t timesteps, double dt) 
    {
      const double time = timesteps * dt;
      return ( (time >= motion_start_threshold) && (time <= motion_end_threshold) );  
    }

    void print_driver_params()
    {
      lout << "Motion type: " << motion_type_to_string(motion_type) << std::endl;
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
    };

    /**
     * @brief Write ball data into a stream.
     */
    void dump_driver_params(std::stringstream &stream)
    {
      stream << "     driver_params: { ";
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
        stream << ", sigma; " << sigma;
        stream << ", damprate; " << damprate;
      }
      stream  <<" }" << std::endl;
    }

    /*
       void operator=(const Driver_params& in)
       {
       motion_type            = in.motion_type;
       motion_vector          = in.motion_vector;
       motion_start_threshold = in.motion_start_threshold;
       motion_end_threshold   = in.motion_end_threshold;
       const_vel              = in.const_vel;
       sigma                  = in.sigma;
       }*/
  };
}


namespace YAML
{
  using exaDEM::Driver_params;
  using exaDEM::MotionType;
  using exanb::lerr;
  using onika::physics::Quantity;

  template <> struct convert<Driver_params>
  {
    static bool decode(const Node &node, Driver_params &v)
    {
      if (!node.IsMap())
      {
        return false;
      }
      if (!node["motion_type"])
      {
        lerr << "\033[31mmotion_type is missing\033[0m\n";
        return false;
      }
      v = {};
      v.motion_type = exaDEM::string_to_motion_type(node["motion_type"].as<std::string>());
      if( v.is_linear() )
      {
        if (!node["motion_vector"])
        {
          lerr << "\033[31mmotion_vector is missing \033[0m\n";
          return false;
        }
        v.motion_vector = node["motion_vector"].as<Vec3d>();

        if( v.motion_type == MotionType::LINEAR_MOTION )
        {
          if (!node["const_vel"])
          {
            lerr << "\033[31mconst_vel is missing \033[0m\n";
            return false;
          }
          if( node["const_vel"] ) v.const_vel = node["const_vel"].as<double>();
        }
        if( v.motion_type == MotionType::LINEAR_FORCE_MOTION )
        {
          if (!node["const_force"])
          {
            lerr << "\033[31mconst_force is missing \033[0m\n";
            return false;
          }
          if( node["const_force"] ) v.const_force = node["const_force"].as<double>();
        }
      }
      if( v.is_compressive() )
      { 
        if (!node["sigma"])
        {
          lerr << "\033[31msigma \033[0m\n";
          return false;
        }
        v.sigma = node["sigma"].as<double>(); 
        if (!node["damprate"])
        {
          lerr << "\033[31mdamprate is missing \033[0m\n";
          return false;
        }
        v.damprate = node["damprate"].as<double>(); 
      }
      if( node["motion_start_threshold"] ) v.motion_start_threshold = node["motion_start_threshold"].as<double>();
      if( node["motion_end_threshold"] ) v.motion_start_threshold = node["motion_end_threshold"].as<double>();
      return true;
    }
  };
}
