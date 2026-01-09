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
#include <exaDEM/forcefield/contact_parameters.h>
#include <exaDEM/forcefield/multimat_parameters.h>

namespace exaDEM {
using namespace exanb;

class MultiMatContactParams : public OperatorNode {
  ADD_SLOT(ParticleTypeMap, particle_type_map, INPUT, REQUIRED );
  ADD_SLOT(MultiMatParamsT<ContactParams>, multimat_cp, OUTPUT, DocString{"List of contact parameters for simulations with multiple materials"});
  ADD_SLOT(std::vector<std::string>, mat1, INPUT, OPTIONAL, DocString{"List of materials."});
  ADD_SLOT(std::vector<std::string>, mat2, INPUT, OPTIONAL, DocString{"List of materials."});
  ADD_SLOT(std::vector<double>,     dncut, INPUT, OPTIONAL, DocString{"List of dncut values."});
  ADD_SLOT(std::vector<double>,        kn, INPUT, OPTIONAL, DocString{"List of ln values."});
  ADD_SLOT(std::vector<double>,        kt, INPUT, OPTIONAL, DocString{"List of kt values."});
  ADD_SLOT(std::vector<double>,        kr, INPUT, OPTIONAL, DocString{"List of kr values."});
  ADD_SLOT(std::vector<double>,        mu, INPUT, OPTIONAL, DocString{"List of mu values."});
  ADD_SLOT(std::vector<double>,        fc, INPUT, OPTIONAL, DocString{"List of fc values."});
  ADD_SLOT(std::vector<double>,     gamma, INPUT, OPTIONAL, DocString{"List of gamma values."});
  ADD_SLOT(std::vector<double>,  damprate, INPUT, OPTIONAL, DocString{"List of damprate values."});
  ADD_SLOT(ContactParams,  default_config, INPUT, OPTIONAL, DocString{"Contact parameters for sphere interactions"});      // can be re-used for to dump contact network
  ADD_SLOT(bool, verbosity, INPUT, false, DocString{"Print force field parameter details"});

  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const override final {
    return R"EOF(
        This operator fills type id to all particles. 

        YAML example:

          - multimat_contact_params:
             mat1:      [  Type1, Type1, Type2 ]
             mat2:      [  Type1, Type2, Type2 ]
             kn:        [   5000, 10000, 15000 ]
             kt:        [   4000,  8000, 12000 ]
             kr:        [    0.0,   0.0,   0.0 ]
             mu:        [    0.1,   0.2,   0.3 ]
             damprate:  [  0.999, 0.999, 0.999 ]

        )EOF";
  }

  inline std::string operator_name() { 
    return "multimat_contact_params";
  }

 public:
  inline void execute() override final
  {
    const auto& type_map = *particle_type_map; 
    MultiMatParamsT<ContactParams>& cp = *multimat_cp;

    if(default_config.has_value()) {
      auto& params = *default_config;
      cp.setup_multimat(type_map, params);
    }
    else {
      cp.setup_multimat(type_map);
    }

    if(mat1.has_value() && mat2.has_value()) {
      int n_types = type_map.size();
      if( n_types == 1 ) {
        lout << "\033[1;32mAdvice: You are defining contact parameters while there is only one type of particle. "
            << "You should use 'contact_force' or 'contact_force_singlemat' as the operator, "
            << "and avoid using 'multimat_contact_params'.\033[0m" << std::endl;
      }

      // check input slots
      auto& material_types_1 = *mat1;
      auto& material_types_2 = *mat2;
      auto& normal_coeffs = *kn;
      auto& tangential_coeffs = *kt;
      auto& rotational_coeffs = *kr;
      auto& frictional_coeffs = *mu;
      auto& damping_coeffs = *damprate;

      std::vector<double> cohesion_coeffs;
      std::vector<double> dncut_coeffs;
      std::vector<double> gamma_coeffs;

      int number_of_pairs = material_types_1.size();

      auto check_lengths_match = [number_of_pairs, this]<typename Vec> (Vec& list, std::string cp_field_name) -> bool {
        if(number_of_pairs != int(list.size())) {
          color_log::error(this->operator_name(), "The length of the field \"" + cp_field_name + "\" does not match the size of the other fields. mat1.size() = " + std::to_string(number_of_pairs) + ", while " + cp_field_name + ".size() = " + std::to_string(list.size()) + ".");
        }
        return true;
      };

      check_lengths_match(material_types_2, "type2");
      check_lengths_match(normal_coeffs, "kn");
      check_lengths_match(tangential_coeffs, "kt");
      check_lengths_match(rotational_coeffs, "kr");
      check_lengths_match(frictional_coeffs, "mu");
      check_lengths_match(damping_coeffs, "damprate");

      bool fill_cohesion_part = false;

      if(fc.has_value() && dncut.has_value()) {
        cohesion_coeffs = *fc;
        dncut_coeffs = *dncut;
        check_lengths_match(cohesion_coeffs, "fc");
        check_lengths_match(dncut_coeffs, "dncut");
        fill_cohesion_part = true;
      } 

      bool fill_DMT_part = false;

      if(gamma.has_value()) {
        gamma_coeffs = *gamma;
        check_lengths_match(gamma_coeffs, "gamma");
        fill_DMT_part = true;
      }

      /** check types / materials */
      for(auto& type_name : material_types_1) {
        if( type_map.find(type_name) == type_map.end()) {
          color_log::error(operator_name(), "The type [" + type_name + "] is not defined", false);
          std::string msg = "Available types are = ";
          for(auto& it : type_map) msg += it.first + " ";
          msg += ".";
          color_log::error(operator_name(), msg);
        }
      }

      for(auto& type_name : material_types_2) {
        if( type_map.find(type_name) == type_map.end()) {
          color_log::error(operator_name(), "The type [" + type_name + "] is not defined", false);
          std::string msg = "Available types are = ";
          for(auto& it : type_map) msg += it.first + " ";
          msg += ".";
          color_log::error(operator_name(), msg);
        }
      }

      for(int p = 0 ; p < number_of_pairs ; p++) {
        std::string m1 = material_types_1[p];
        std::string m2 = material_types_2[p];
        int64_t type_1 = type_map.at(m1);
        int64_t type_2 = type_map.at(m2);
        ContactParams params;
        params.kn = normal_coeffs[p];
        params.kt = tangential_coeffs[p];
        params.kr = rotational_coeffs[p];
        params.mu = frictional_coeffs[p];
        params.damp_rate = damping_coeffs[p];

        if(fill_cohesion_part) {
          params.fc = cohesion_coeffs[p];
          params.dncut = dncut_coeffs[p];
        } 
        else {
          params.fc = 0.0;
          params.dncut = 0.0;
        }

        if(fill_DMT_part) {
          params.gamma = gamma_coeffs[p];
        } 
        else {
          params.gamma = 0.0;
        }
        cp.register_multimat(type_1, type_2, params);
      }
    }

    bool multimat_mode = true;
    bool driver_mode = false;

    cp.check_completeness(multimat_mode, driver_mode);
    if(*verbosity) cp.display(multimat_mode, driver_mode);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(multimat_contact_params) { 
  OperatorNodeFactory::instance()->register_factory("multimat_contact_params", make_simple_operator<MultiMatContactParams>); 
}
} // namespace exaDEM
