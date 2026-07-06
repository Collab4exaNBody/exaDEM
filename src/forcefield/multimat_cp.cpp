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
#include <exaDEM/forcefield/contact_parameters.hpp>
#include <exaDEM/forcefield/multimat_parameters.hpp>

namespace exaDEM {

class MultiMatContactParams : public OperatorNode {
  ADD_SLOT(MultiMatParamsT<ContactParams>, multimat_cp, OUTPUT,
           DocString{"List of contact parameters for simulations with multiple materials"});
  ADD_SLOT(std::vector<uint32_t>, group1, INPUT, OPTIONAL, DocString{"List of group indices for the first particle."});
  ADD_SLOT(std::vector<uint32_t>, group2, INPUT, OPTIONAL, DocString{"List of group indices for the second particle."});
  ADD_SLOT(std::vector<double>, dncut, INPUT, OPTIONAL, DocString{"List of dncut values."});
  ADD_SLOT(std::vector<double>, kn, INPUT, OPTIONAL, DocString{"List of ln values."});
  ADD_SLOT(std::vector<double>, kt, INPUT, OPTIONAL, DocString{"List of kt values."});
  ADD_SLOT(std::vector<double>, kr, INPUT, OPTIONAL, DocString{"List of kr values."});
  ADD_SLOT(std::vector<double>, mu, INPUT, OPTIONAL, DocString{"List of mu values."});
  ADD_SLOT(std::vector<double>, fc, INPUT, OPTIONAL, DocString{"List of fc values."});
  ADD_SLOT(std::vector<double>, gamma, INPUT, OPTIONAL, DocString{"List of gamma values."});
  ADD_SLOT(std::vector<double>, damprate, INPUT, OPTIONAL, DocString{"List of damprate values."});
  ADD_SLOT(ContactParams, default_config, INPUT, OPTIONAL,
           DocString{"Contact parameters for sphere interactions"});  // can be re-used for to dump contact network
  ADD_SLOT(bool, verbosity, INPUT, false, DocString{"Print force field parameter details"});

  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
        This operator fills type id to all particles. 

        YAML example:

          - multimat_contact_params:
             group1:    [      0,     0,     1 ]
             group2:    [      0,     1,     1 ]
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
  inline void execute() final {
    MultiMatParamsT<ContactParams>& cp = *multimat_cp;

    if (group1.has_value() && group2.has_value()) {
      auto& groups_1 = *group1;
      auto& groups_2 = *group2;
      auto& normal_coeffs = *kn;
      auto& tangential_coeffs = *kt;
      auto& rotational_coeffs = *kr;
      auto& frictional_coeffs = *mu;
      auto& damping_coeffs = *damprate;

      std::vector<double> cohesion_coeffs;
      std::vector<double> dncut_coeffs;
      std::vector<double> gamma_coeffs;

      int number_of_pairs = groups_1.size();

      auto check_lengths_match = [number_of_pairs, this]<typename Vec>(Vec& list, std::string cp_field_name) -> bool {
        if (number_of_pairs != int(list.size())) {
          color_log::error(this->operator_name(), "The length of the field \"" + cp_field_name +
                                                      "\" does not match the size of the other fields. group1.size() = " +
                                                      std::to_string(number_of_pairs) + ", while " + cp_field_name +
                                                      ".size() = " + std::to_string(list.size()) + ".");
        }
        return true;
      };

      check_lengths_match(groups_2, "group2");
      check_lengths_match(normal_coeffs, "kn");
      check_lengths_match(tangential_coeffs, "kt");
      check_lengths_match(rotational_coeffs, "kr");
      check_lengths_match(frictional_coeffs, "mu");
      check_lengths_match(damping_coeffs, "damprate");

      bool fill_cohesion_part = false;

      if (fc.has_value() && dncut.has_value()) {
        cohesion_coeffs = *fc;
        dncut_coeffs = *dncut;
        check_lengths_match(cohesion_coeffs, "fc");
        check_lengths_match(dncut_coeffs, "dncut");
        fill_cohesion_part = true;
      }

      bool fill_DMT_part = false;

      if (gamma.has_value()) {
        gamma_coeffs = *gamma;
        check_lengths_match(gamma_coeffs, "gamma");
        fill_DMT_part = true;
      }

      // Compute number of groups as max(group1, group2) + 1
      uint32_t max_group = 0;
      for (auto g : groups_1) max_group = std::max(max_group, g);
      for (auto g : groups_2) max_group = std::max(max_group, g);
      int n_groups = static_cast<int>(max_group) + 1;

      if (default_config.has_value()) {
        cp.setup_multimat(n_groups, *default_config);
      } else {
        cp.setup_multimat(n_groups);
      }

      for (int p = 0; p < number_of_pairs; p++) {
        int g1 = static_cast<int>(groups_1[p]);
        int g2 = static_cast<int>(groups_2[p]);
        ContactParams params;
        params.kn_ = normal_coeffs[p];
        params.kt_ = tangential_coeffs[p];
        params.kr_ = rotational_coeffs[p];
        params.mu_ = frictional_coeffs[p];
        params.damp_rate_ = damping_coeffs[p];

        if (fill_cohesion_part) {
          params.fc_ = cohesion_coeffs[p];
          params.dncut_ = dncut_coeffs[p];
        } else {
          params.fc_ = 0.0;
          params.dncut_ = 0.0;
        }

        if (fill_DMT_part) {
          params.gamma_ = gamma_coeffs[p];
        } else {
          params.gamma_ = 0.0;
        }
        cp.register_multimat(g1, g2, params);
      }
    } else if (default_config.has_value()) {
      cp.setup_multimat(1, *default_config);
    }

    bool multimat_mode = group1.has_value();
    bool driver_mode = false;

    cp.check_completeness(multimat_mode, driver_mode);
    if (*verbosity) {
      cp.display(multimat_mode, driver_mode);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(multimat_contact_params) {
  OperatorNodeFactory::instance()->register_factory("multimat_contact_params",
                                                    make_simple_operator<MultiMatContactParams>);
}
}  // namespace exaDEM
