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
#include <exaDEM/forcefield/inner_bond_parameters.h>
#include <exaDEM/forcefield/multimat_parameters.h>

namespace exaDEM
{
  using namespace exanb;

  class InnerBondParamsOp : public OperatorNode
  {
    ADD_SLOT(ParticleTypeMap, particle_type_map, INPUT, REQUIRED );
    ADD_SLOT(MultiMatParamsT<InnerBondParams>, multimat_ibp, OUTPUT, DocString{"List of contact parameters for simulations with multiple materials"});
    ADD_SLOT(std::vector<std::string>, mat1, INPUT, OPTIONAL, DocString{"List of materials."});
    ADD_SLOT(std::vector<std::string>, mat2, INPUT, OPTIONAL, DocString{"List of materials."});
    ADD_SLOT(std::vector<double>,        kn, INPUT, OPTIONAL, DocString{"List of ln values."});
    ADD_SLOT(std::vector<double>,        kt, INPUT, OPTIONAL, DocString{"List of kt values."});
    ADD_SLOT(std::vector<double>,        mu, INPUT, OPTIONAL, DocString{"List of mu values."});
    ADD_SLOT(std::vector<double>,        en, INPUT, OPTIONAL, DocString{"List of en2 values."});
    ADD_SLOT(std::vector<double>,       pow, INPUT, OPTIONAL, DocString{"List of pow values."});
    ADD_SLOT(std::vector<double>,         g, INPUT, OPTIONAL, DocString{"List of g values."});
    ADD_SLOT(InnerBondParams, default_config, INPUT, OPTIONAL, DocString{"Contact parameters for sphere interactions"});      // can be re-used for to dump contact network

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator fills type id to all particles. 

        YAML example:

          - inner_bond_params:
             mat1:      [  Type1, Type1, Type2 ]
             mat2:      [  Type1, Type2, Type2 ]
             kn:        [   5000, 10000, 15000 ]
             kt:        [   4000,  8000, 12000 ]
             kr:        [    0.0,   0.0,   0.0 ]
             mu:        [    0.1,   0.2,   0.3 ]
             damprate:  [  0.999, 0.999, 0.999 ]

        )EOF";
    }

    inline std::string operator_name() { return "inner_bond_params"; }

    public:
    inline void execute() override final
    {
      const auto& type_map = *particle_type_map; 
      MultiMatParamsT<InnerBondParams>& ibp = *multimat_ibp;

      int n_types = type_map.size();
      if( n_types == 1 )
      {
        lout << "\033[1;32mAdvice: You are defining contact parameters while there is only one type of particle. "
          << "You should use 'contact_force' or 'contact_force_singlemat' as the operator, "
          << "and avoid using 'multimat_contact_params'.\033[0m" << std::endl;
      }

      if( !mat1.has_value() && !mat2.has_value())
      {
        if(!default_config.has_value())
        {
          color_log::error(this->operator_name(),"You must define either a list for material combinations (mat1, mat2) or a list of default parameters \'devault_config\'.");
        }
      }

      if(default_config.has_value()) 
      {
        auto& params = *default_config;
        ibp.setup_multimat(type_map, params);
      }
      else
      {
        ibp.setup_multimat(type_map);
      }

      if(mat1.has_value() && mat2.has_value())
      {

        // check input slots
        auto& material_types_1 = *mat1;
        auto& material_types_2 = *mat2;
        auto& normal_coeffs = *kn;
        auto& tangential_coeffs = *kt;
        auto& frictional_coeffs = *mu;
        auto& en2_coeffs = *en;
        auto& pow_coeffs = *pow;
        auto& g_coeffs = *g;

        int number_of_pairs = material_types_1.size();

        auto check_lengths_match = [number_of_pairs, this]<typename Vec> (Vec& list, std::string ibp_field_name) -> bool 
        {
          if( number_of_pairs != int(list.size()) ) 
          {
            color_log::error(this->operator_name(), "The length of the field \"" + ibp_field_name + "\" does not match the size of the other fields. mat1.size() = " + std::to_string(number_of_pairs) + ", while " + ibp_field_name + ".size() = " + std::to_string(list.size()) + ".");
          }
          return true;
        };

        check_lengths_match(material_types_2, "type2");
        check_lengths_match(normal_coeffs, "kn");
        check_lengths_match(tangential_coeffs, "kt");
        check_lengths_match(frictional_coeffs, "mu");
        check_lengths_match(en2_coeffs, "en");
        check_lengths_match(pow_coeffs, "pow");
        check_lengths_match(g_coeffs, "g");

        /** check types / materials */
        for(auto& type_name : material_types_1)
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

        for(auto& type_name : material_types_2)
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

        for(int p = 0 ; p < number_of_pairs ; p++)
        {
          std::string m1 = material_types_1[p];
          std::string m2 = material_types_2[p];
          int64_t type_1 = type_map.at(m1);
          int64_t type_2 = type_map.at(m2);
          InnerBondParams params;
          params.kn = normal_coeffs[p];
          params.kt = tangential_coeffs[p];
          params.mu = frictional_coeffs[p];
          params.en = en2_coeffs[p];
          params.pow = pow_coeffs[p];
          params.g = g_coeffs[p];

          ibp.register_multimat(type_1, type_2, params);
        } 
      }


      bool multimat_mode = true;
      bool driver_mode = false;

      ibp.check_completeness(multimat_mode, driver_mode);
      ibp.display(multimat_mode, driver_mode);
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(inner_bond_params) { OperatorNodeFactory::instance()->register_factory("inner_bond_params", make_simple_operator<InnerBondParamsOp>); }
} // namespace exaDEM
