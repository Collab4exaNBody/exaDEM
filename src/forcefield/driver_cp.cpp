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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/particle_type_id.h>
#include <memory>
#include <exaDEM/multimat_parameters.h>
#include <exaDEM/drivers.h>

namespace exaDEM
{
  using namespace exanb;

  class DriversContactParams : public OperatorNode
  {
    ADD_SLOT(ParticleTypeMap, particle_type_map, INPUT, REQUIRED );
    ADD_SLOT(MultiMatParamsT<ContactParams>, multimat_cp, INPUT_OUTPUT, REQUIRED, DocString{"List of contact parameters for simulations with multiple materials"});
    ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(std::vector<std::string>,  mat, INPUT, OPTIONAL, DocString{"List of materials."});
    ADD_SLOT(std::vector<int>,    driver_id, INPUT, OPTIONAL, DocString{"List of drivers."});
    ADD_SLOT(std::vector<double>,     dncut, INPUT, OPTIONAL, DocString{"List of dncut values."});
    ADD_SLOT(std::vector<double>,        kn, INPUT, OPTIONAL, DocString{"List of ln values."});
    ADD_SLOT(std::vector<double>,        kt, INPUT, OPTIONAL, DocString{"List of kt values."});
    ADD_SLOT(std::vector<double>,        kr, INPUT, OPTIONAL, DocString{"List of kr values."});
    ADD_SLOT(std::vector<double>,        mu, INPUT, OPTIONAL, DocString{"List of mu values."});
    ADD_SLOT(std::vector<double>,        fc, INPUT, OPTIONAL, DocString{"List of fc values."});
    ADD_SLOT(std::vector<double>,  damprate, INPUT, OPTIONAL, DocString{"List of damprate values."});
    ADD_SLOT(ContactParams, default_config, INPUT, OPTIONAL, DocString{"Contact parameters for sphere interactions"});      // can be re-used for to dump contact network

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator fills the list of contact parameters between particles and drivers. 

        YAML example:

          - drivers_contact_params:
             mat:       [  Type1, Type2 ]
             driver_id: [      0,     0 ]
             kn:        [  10000, 15000 ]
             kt:        [   8000, 12000 ]
             kr:        [    0.0,   0.1 ]
             mu:        [    0.5,   0.5 ]
             damprate:  [  0.999, 0.999 ]

        )EOF";
    }

    inline std::string operator_name() { return "drivers_contact_params"; }

    public:
    inline void execute() override final
    {
      const auto& type_map = *particle_type_map; 
      int n_types = type_map.size();
      MultiMatParamsT<ContactParams>& cp = *multimat_cp;

      // We use the same data map to for drivers
      ParticleTypeMap driver_map;
      Drivers& drvs = *drivers;

      /** Build the driver map */
      for(int i = 0 ; i < int(drvs.get_size()) ; i++)
      {
        auto driver_type = drvs.type(i);
        std::string driver_name;
        if(driver_type == DRIVER_TYPE::STL_MESH)
        {
          Stl_mesh& D = drvs.get_typed_driver<Stl_mesh>(i);
          driver_name = D.shp.m_name;
        }
        else
        {
          driver_name = print(driver_type) + "_" + std::to_string(i);
        }
        driver_map[driver_name] = i;
      }

      ContactParams params;
      if(default_config.has_value()) params = *default_config;
      cp.setup_drivers(driver_map, params);

      if( n_types == 1 && drvs.get_size() == 1)
      {

        color_log::warning(operator_name(), "Advice: You are defining contact parameters while there is only one type of particle.");
        color_log::warning(operator_name(), "You should use 'contact_force' or 'contact_force_singlemat' as the operator, and avoid using 'drivers_contact_params'.");
      }

      // check input slots
      if(mat.has_value()) /** not default config */
      {
        auto& material_types    = *mat;
        auto& drv_id  = *driver_id;
        auto& normal_coeffs     = *kn;
        auto& tangential_coeffs = *kt;
        auto& rotational_coeffs = *kr;
        auto& frictional_coeffs = *mu;
        auto& damping_coeffs    = *damprate;

        std::vector<double> cohesion_coeffs;
        std::vector<double> dncut_coeffs;

        int number_of_pairs = material_types.size();

        auto check_lengths_match = [number_of_pairs, this]<typename Vec> (Vec& list, std::string cp_field_name) -> bool 
        {
          if( number_of_pairs != int(list.size()) ) 
          {
            std::string msg = "The length of the field \"" + cp_field_name;
            msg            += "\" does not match the size of the other fields. ";
            msg            += "mat1.size() = " + std::to_string(number_of_pairs);
            msg            += ", while " + cp_field_name + ".size() = " + std::to_string(list.size());
            color_log::error(this->operator_name(), msg);
          }
          return true;
        };

        check_lengths_match(drv_id, "drvs");
        check_lengths_match(normal_coeffs, "kn");
        check_lengths_match(tangential_coeffs, "kt");
        check_lengths_match(rotational_coeffs, "kr");
        check_lengths_match(frictional_coeffs, "mu");
        check_lengths_match(damping_coeffs, "damprate");

        bool fill_cohesion_part = false;

        if(fc.has_value() && dncut.has_value())
        {
          cohesion_coeffs = *fc;
          dncut_coeffs    = *dncut;
          check_lengths_match(cohesion_coeffs, "fc");
          check_lengths_match(dncut_coeffs, "dncut");
          fill_cohesion_part = true;
        } 

        /** check types / materials */
        for(auto& type_name : material_types)
        {
          if( type_map.find(type_name) == type_map.end())
          {
            color_log::error(operator_name(), "The type [" + type_name + "] is not defined", false);
            std::string msg = "Available types are = "; 
            for(auto& it : type_map) msg += it.first + " ";
            msg += ".";
            color_log::error(operator_name(), msg);
          }
        }

        /** Check input driver indexes */
        for(auto did : drv_id)
        {
          if(did >= int(drvs.get_size()))
          {
            color_log::error(operator_name(), "driver id: " + std::to_string(did) + " is out of range of the driver list: " + std::to_string(drvs.get_size()));
          }
          auto driver_type = drvs.type(did);
          if( driver_type == DRIVER_TYPE::UNDEFINED )
          {
            color_log::error(operator_name(),"The driver type is undefined for id: " + std::to_string(did));
          }
        }

        for(int p = 0 ; p < number_of_pairs ; p++)
        {
          std::string m1 = material_types[p];
          int64_t type = type_map.at(m1);
          int64_t drv  = drv_id[p];
          params.kn = normal_coeffs[p];
          params.kt = tangential_coeffs[p];
          params.kr = rotational_coeffs[p];
          params.mu = frictional_coeffs[p];
          params.damp_rate = damping_coeffs[p];

          if(fill_cohesion_part)
          {
            params.fc = cohesion_coeffs[p];
            params.dncut = dncut_coeffs[p];
          } 
          else
          {
            params.fc = 0.0;
            params.dncut = 0.0;
          }
          cp.register_driver(type, drv, params);
        } 
      }

      bool multimat_mode = false;
      bool driver_mode = true;

      cp.check_completeness(multimat_mode, driver_mode);
      cp.display(multimat_mode, driver_mode);
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(multimat_contact_params) { OperatorNodeFactory::instance()->register_factory("drivers_contact_params", make_simple_operator<DriversContactParams>); }
} // namespace exaDEM
