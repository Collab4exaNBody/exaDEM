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

#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/particle_type_id.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/forcefield/inner_bond_parameters.hpp>
#include <exaDEM/forcefield/multimat_parameters.hpp>

namespace exaDEM {

class InnerBondParamsOp : public OperatorNode {
  ADD_SLOT(MultiMatParamsT<InnerBondParams>, multimat_ibp, OUTPUT,
           DocString{"List of contact parameters for simulations with multiple materials"});
  ADD_SLOT(std::vector<uint32_t>, group1, INPUT, OPTIONAL, DocString{"List of group indices for the first particle."});
  ADD_SLOT(std::vector<uint32_t>, group2, INPUT, OPTIONAL, DocString{"List of group indices for the second particle."});
  ADD_SLOT(std::vector<double>, kn, INPUT, OPTIONAL, DocString{"List of ln values."});
  ADD_SLOT(std::vector<double>, kt, INPUT, OPTIONAL, DocString{"List of kt values."});
  ADD_SLOT(std::vector<double>, damp_rate, INPUT, OPTIONAL, DocString{"List of en2 values."});
  ADD_SLOT(std::vector<double>, g, INPUT, OPTIONAL,
           DocString{"List of g values (mixed mode fracture energy release rate). Mutually exclusive with gn/gt."});
  ADD_SLOT(std::vector<double>, gn, INPUT, OPTIONAL,
           DocString{"List of gn values (separate modes normal fracture energy release rate). Requires gt."});
  ADD_SLOT(std::vector<double>, gt, INPUT, OPTIONAL,
           DocString{"List of gt values (separate modes tangential fracture energy release rate). Requires gn."});
  ADD_SLOT(std::vector<double>, sigma, INPUT, OPTIONAL,
           DocString{"List of stress criterion values (stress separate mode). Requires g."});

  ADD_SLOT(InnerBondParams, default_config, INPUT, OPTIONAL,
           DocString{"Contact parameters for sphere interactions"});  // can be re-used for to dump contact network
  ADD_SLOT(bool, verbosity, INPUT, false, DocString{"Print force field parameter details"});

  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
        This operator fills innerbond input parameters. 

        YAML example:

          - inner_bond_params:
             group1:    [      0,     0,     1 ]
             group2:    [      0,     1,     1 ]
             kn:        [   5000, 10000, 15000 ]
             kt:        [   4000,  8000, 12000 ]
             damp_rate: [  0.999, 0.999, 0.999 ]
             gn:        [   1e-5,  1e-5,  1e-5 ]
             gt:        [   1e-5,  1e-5,  1e-5 ]

        Either provide "g" (mixed mode fracture criterion) or both "gn" and "gt"
        (separate modes fracture criterion) for every pair, but not both at once:

          - inner_bond_params:
             group1:    [      0,     1 ]
             group2:    [      0,     1 ]
             kn:        [   5000, 15000 ]
             kt:        [   4000, 12000 ]
             damp_rate: [  0.999, 0.999 ]
             g:         [   1e-5,  1e-5 ]
        )EOF";
  }

  inline std::string operator_name() { return "inner_bond_params"; }

 public:
  inline void execute() final {
    MultiMatParamsT<InnerBondParams>& ibp = *multimat_ibp;

    if (!group1.has_value() && !group2.has_value()) {
      if (!default_config.has_value()) {
        color_log::error(
            this->operator_name(),
            "You must define either a list for group combinations (group1, group2) or a default config.");
      }
    }

    if (group1.has_value() && group2.has_value()) {
      auto& groups_1 = *group1;
      auto& groups_2 = *group2;
      auto& normal_coeffs = *kn;
      auto& tangential_coeffs = *kt;
      auto& damprate_coeffs = *damp_rate;

      int number_of_pairs = groups_1.size();

      auto check_lengths_match = [number_of_pairs, this]<typename Vec>(Vec& list, std::string ibp_field_name) -> bool {
        if (number_of_pairs != int(list.size())) {
          color_log::error(
              this->operator_name(),
              "The length of the field \"" + ibp_field_name +
                  "\" does not match the size of the other fields. group1.size() = " + std::to_string(number_of_pairs) +
                  ", while " + ibp_field_name + ".size() = " + std::to_string(list.size()) + ".");
        }
        return true;
      };

      check_lengths_match(groups_2, "group2");
      check_lengths_match(normal_coeffs, "kn");
      check_lengths_match(tangential_coeffs, "kt");
      check_lengths_match(damprate_coeffs, "damp_rate");

      if (g.has_value() && (gn.has_value() || gt.has_value())) {
        color_log::error(this->operator_name(), "\"g\" is mutually exclusive with \"gn\"/\"gt\".");
      }
      if (g.has_value()) {
        check_lengths_match(*g, "g");
        if (sigma.has_value()) {
          check_lengths_match(*sigma, "sigma");
        }
      } else if (gn.has_value() && gt.has_value()) {
        check_lengths_match(*gn, "gn");
        check_lengths_match(*gt, "gt");
      } else {
        color_log::error(this->operator_name(),
                         "You must define either \"g\", or both \"gn\" and \"gt\", or \"g\" and \"sigma\".");
      }

      // Compute number of groups as max(group1, group2) + 1
      uint32_t max_group = 0;
      for (auto g : groups_1) max_group = std::max(max_group, g);
      for (auto g : groups_2) max_group = std::max(max_group, g);
      int n_groups = static_cast<int>(max_group) + 1;

      if (default_config.has_value()) {
        ibp.setup_multimat(n_groups, *default_config);
      } else {
        ibp.setup_multimat(n_groups);
      }

      for (int p = 0; p < number_of_pairs; p++) {
        int g1 = static_cast<int>(groups_1[p]);
        int g2 = static_cast<int>(groups_2[p]);
        InnerBondParams params;
        params.kn_ = normal_coeffs[p];
        params.kt_ = tangential_coeffs[p];
        params.damp_rate_ = damprate_coeffs[p];
        if (g.has_value()) {
          if (!sigma.has_value()) {
            params.mode_ = RuptureMode::EnergyMixedMode;
            params.crit1_ = (*g)[p];
            params.crit2_ = 0.0;
          } else {
            params.mode_ = RuptureMode::StressEnergySeparateMode;
            params.crit1_ = (*g)[p];
            params.crit2_ = (*sigma)[p];
          }
        } else {
          params.mode_ = RuptureMode::EnergySeparateMode;
          params.crit1_ = (*gn)[p];
          params.crit2_ = (*gt)[p];
        }

        ibp.register_multimat(g1, g2, params);
      }
    } else if (default_config.has_value()) {
      ibp.setup_multimat(1, *default_config);
    }

    bool multimat_mode = group1.has_value();
    bool driver_mode = false;

    ibp.check_completeness(multimat_mode, driver_mode);
    if (*verbosity) {
      ibp.display(multimat_mode, driver_mode);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(inner_bond_params) {
  OperatorNodeFactory::instance()->register_factory("inner_bond_params", make_simple_operator<InnerBondParamsOp>);
}
}  // namespace exaDEM
