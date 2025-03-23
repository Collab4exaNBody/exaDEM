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
    TABULATED,                 /**< Motion defined by precomputed or tabulated data. */
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

    // If the string doesn't match any valid MotionType, return a default value
    return UNKNOWN;  // Or some other default action like throwing an exception or logging
  }

  struct Driver_params
  {
    // Common motion stuff
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

    // Motion: Tabulated
    std::vector<double> tab_time;
    std::vector<Vec3d> tab_pos;

//    Driver_params() = default;
//    ~Driver_params() = default;

    inline bool is_stationary() const { return motion_type == STATIONARY; }
    inline bool is_tabulated() const { return motion_type == TABULATED; }

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
        lout << "\033[31mThis motion type [" << motion_type_to_string(motion_type) << "] is not possible, MotionType availables are: ";
        for(const auto& motion: valid_motion_types)
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
      if( is_tabulated() )
      {
        if(tab_time.size() == 0)
        {
          lout << "\033[31m[Warning] The \"time\" input slot is not defined while the tabulated motion is activated.\033[0m" << std::endl;
          return false;
        }
        else if(tab_time[0] != 0.0)
        {
          lout << "\033[31m[Warning] Please set the first element of your input time vector to 0.\033[0m" << std::endl;
          return false;
        }
        if(tab_pos.size() == 0)
        {
          lout << "\033[31m[Warning] The \"positions\" input slot is not defined while the tabulated motion is activated.\033[0m" << std::endl;
          return false;
        }
        if(tab_time.size() != tab_pos.size())
        {
          lout << "\033[31m[Warning] The \"positions\" and \"time\" input slot are not the same size.\033[0m" << std::endl;
          return false;
        }
      }
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

    bool is_motion_triggered(uint64_t timesteps, double dt) const
    {
      const double time = timesteps * dt;
      return ( (time >= motion_start_threshold) && (time <= motion_end_threshold) );  
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
        //std::cout << idx_lower  << " " << tab_time[idx_lower]  << " " << time << std::endl;
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
      if( v.is_tabulated() )
      { 
        if (!node["time"])
        {
          lerr << "\033[31m time is missing \033[0m\n";
          return false;
        }
        v.tab_time= node["time"].as<std::vector<double>>(); 
        if (!node["positions"])
        {
          lerr << "\033[31m positions is missing \033[0m\n";
          return false;
        }
        v.tab_pos = node["positions"].as<std::vector<Vec3d>>(); 
      }
      if( node["motion_start_threshold"] ) v.motion_start_threshold = node["motion_start_threshold"].as<double>();
      if( node["motion_end_threshold"] ) v.motion_start_threshold = node["motion_end_threshold"].as<double>();
      return true;
    }
  };
}
