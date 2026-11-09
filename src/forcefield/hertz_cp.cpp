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

#include <algorithm>
#include <exaDEM/color_log.hpp>
#include <exaDEM/drivers.hpp>
#include <exaDEM/forcefield/hertz_parameters.hpp>
#include <exaDEM/forcefield/multimat_parameters.hpp>

namespace exaDEM {

namespace hertz_cp_detail {
/** Optional list of values: the value of the pair p, or the default value if the list is not defined. */
inline double value_or(const std::vector<double>* list, size_t p, double default_value = 0.0) {
  return list != nullptr ? (*list)[p] : default_value;
}

/** Fills the parameters of the pair number p from the (optional) lists. */
inline HertzParams make_pair_params(size_t p, const std::vector<double>& kt, const std::vector<double>* kr,
                                    const std::vector<double>& mu, const std::vector<double>& damprate,
                                    const std::vector<double>* fc, const std::vector<double>* dncut,
                                    const std::vector<double>* gamma) {
  HertzParams params;
  params.kt_ = kt[p];
  params.kr_ = value_or(kr, p);
  params.mu_ = mu[p];
  params.damp_rate_ = damprate[p];
  params.fc_ = (fc != nullptr && dncut != nullptr) ? (*fc)[p] : 0.0;
  params.dncut_ = (fc != nullptr && dncut != nullptr) ? (*dncut)[p] : 0.0;
  params.gamma_ = value_or(gamma, p);
  return params;
}
}  // namespace hertz_cp_detail

/**
 * Hertz contact parameters between the groups (materials) of particles.
 * See MultiMatContactParams for the Hooke law.
 */
class MultiMatHertzContactParams : public OperatorNode {
  ADD_SLOT(MultiMatParamsT<HertzParams>, multimat_hertz_cp, OUTPUT,
           DocString{"List of Hertz contact parameters for simulations with multiple materials"});
  ADD_SLOT(std::vector<uint32_t>, group1, INPUT, OPTIONAL, DocString{"List of group indices for the first particle."});
  ADD_SLOT(std::vector<uint32_t>, group2, INPUT, OPTIONAL, DocString{"List of group indices for the second particle."});
  ADD_SLOT(uint32_t, n_groups, INPUT_OUTPUT, 1,
           DocString{"Number of distinct groups. If already set to more than 1 (e.g. by set_group or "
                     "read_conf_rockable), it is checked against group1/group2; otherwise it is computed from "
                     "group1/group2 (max group index + 1) and written back."});
  ADD_SLOT(std::vector<double>, young, INPUT, OPTIONAL,
           DocString{"Young modulus of each group material (one value per group). Requires poisson."});
  ADD_SLOT(std::vector<double>, poisson, INPUT, OPTIONAL,
           DocString{"Poisson ratio of each group material (one value per group). Requires young."});
  ADD_SLOT(std::vector<double>, dncut, INPUT, OPTIONAL, DocString{"List of dncut values."});
  ADD_SLOT(std::vector<double>, kt, INPUT, OPTIONAL, DocString{"List of kt values."});
  ADD_SLOT(std::vector<double>, kr, INPUT, OPTIONAL, DocString{"List of kr values."});
  ADD_SLOT(std::vector<double>, mu, INPUT, OPTIONAL, DocString{"List of mu values."});
  ADD_SLOT(std::vector<double>, fc, INPUT, OPTIONAL, DocString{"List of fc values."});
  ADD_SLOT(std::vector<double>, gamma, INPUT, OPTIONAL, DocString{"List of gamma values."});
  ADD_SLOT(std::vector<double>, damprate, INPUT, OPTIONAL, DocString{"List of damprate values."});
  ADD_SLOT(HertzParams, default_config, INPUT, OPTIONAL,
           DocString{"Hertz contact parameters of the pairs of groups which are not listed."});
  ADD_SLOT(bool, verbosity, INPUT, false, DocString{"Print force field parameter details"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator fills the Hertz contact parameters between the groups (materials) of spheres.
        The Young modulus and the Poisson ratio are defined for each group (one value per group), the
        reduced Young modulus E* of each pair of groups is computed from them. The other lists have one
        value per listed pair of groups. Use it with the contact_sphere_hertz_[none|cohesive|dmt] operators.
        See multimat_contact_params for the Hooke law.

        YAML example:

          - multimat_contact_params_hertz:
             group1:    [      0,     0,     1 ]
             group2:    [      0,     1,     1 ]
             young:     [  70e9, 200e9 ]
             poisson:   [  0.33,  0.30 ]
             kt:        [   4000,  8000, 12000 ]
             kr:        [    0.0,   0.0,   0.0 ]
             mu:        [    0.1,   0.2,   0.3 ]
             damprate:  [  0.999, 0.999, 0.999 ]

        )EOF";
  }

  inline std::string operator_name() { return "multimat_contact_params_hertz"; }

 public:
  inline void execute() final {
    MultiMatParamsT<HertzParams>& cp = *multimat_hertz_cp;

    if (group1.has_value() && group2.has_value()) {
      auto& groups_1 = *group1;
      auto& groups_2 = *group2;
      const size_t number_of_pairs = groups_1.size();

      auto check_lengths_match = [number_of_pairs, this]<typename Vec>(Vec& list, std::string cp_field_name) {
        if (number_of_pairs != list.size()) {
          color_log::error(
              this->operator_name(),
              "The length of the field \"" + cp_field_name +
                  "\" does not match the size of the other fields. group1.size() = " + std::to_string(number_of_pairs) +
                  ", while " + cp_field_name + ".size() = " + std::to_string(list.size()) + ".");
        }
      };
      if (!kt.has_value() || !mu.has_value() || !damprate.has_value()) {
        color_log::error(operator_name(), "kt, mu and damprate are required with group1 and group2.");
      }
      check_lengths_match(groups_2, "group2");
      check_lengths_match(*kt, "kt");
      check_lengths_match(*mu, "mu");
      check_lengths_match(*damprate, "damprate");
      if (kr.has_value()) check_lengths_match(*kr, "kr");
      if (fc.has_value() != dncut.has_value()) {
        color_log::error(operator_name(), "fc and dncut must be defined together.");
      }
      if (fc.has_value()) {
        check_lengths_match(*fc, "fc");
        check_lengths_match(*dncut, "dncut");
      }
      if (gamma.has_value()) check_lengths_match(*gamma, "gamma");

      // Compute number of groups as max(group1, group2) + 1
      uint32_t max_group = 0;
      for (auto g : groups_1) max_group = std::max(max_group, g);
      for (auto g : groups_2) max_group = std::max(max_group, g);
      const uint32_t n_groups_from_fields = max_group + 1;
      const uint32_t n_groups_upstream = *n_groups;
      *n_groups = std::max(n_groups_upstream, n_groups_from_fields);

      if (n_groups_upstream > n_groups_from_fields && !default_config.has_value()) {
        color_log::error(operator_name(),
                         "n_groups = " + std::to_string(n_groups_upstream) +
                             " is larger than the number of groups covered by group1/group2 (" +
                             std::to_string(n_groups_from_fields) +
                             "). A default_config must be defined to cover the remaining group pairs.");
      }

      if (default_config.has_value()) {
        cp.setup_multimat(static_cast<int>(*n_groups), *default_config);
      } else {
        cp.setup_multimat(static_cast<int>(*n_groups));
      }

      for (size_t p = 0; p < number_of_pairs; p++) {
        HertzParams params = hertz_cp_detail::make_pair_params(
            p, *kt, kr.has_value() ? &*kr : nullptr, *mu, *damprate, fc.has_value() ? &*fc : nullptr,
            dncut.has_value() ? &*dncut : nullptr, gamma.has_value() ? &*gamma : nullptr);
        cp.register_multimat(static_cast<int>(groups_1[p]), static_cast<int>(groups_2[p]), params);
      }
    } else if (default_config.has_value()) {
      cp.setup_multimat(static_cast<int>(*n_groups), *default_config);
    } else {
      color_log::error(operator_name(), "Please define group1, group2 and the parameters, or default_config.");
    }

    // Reduced Young modulus of each pair of groups, from the materials of the groups
    if (young.has_value() || poisson.has_value()) {
      if (!young.has_value() || !poisson.has_value()) {
        color_log::error(operator_name(), "young and poisson must be defined together.");
      }
      const int n = cp.number_of_groups_;
      if (static_cast<int>(young->size()) != n || static_cast<int>(poisson->size()) != n) {
        color_log::error(operator_name(),
                         "young and poisson must have one value per group (n_groups = " + std::to_string(n) +
                             "), but young.size() = " + std::to_string(young->size()) +
                             " and poisson.size() = " + std::to_string(poisson->size()) + ".");
      }
      // The material of a group is kept in the parameters of the pair (group, group)
      for (int g1 = 0; g1 < n; g1++) {
        for (int g2 = 0; g2 < n; g2++) {
          HertzParams& params = cp.multimat_cp_[cp.get_idx_multimat(g1, g2)];
          params.estar_ = compute_reduced_young_modulus((*young)[g1], (*poisson)[g1], (*young)[g2], (*poisson)[g2]);
          if (g1 == g2) {
            params.young_ = (*young)[g1];
            params.poisson_ = (*poisson)[g1];
          }
        }
      }
    }

    cp.check_completeness(true, false);
    if (*verbosity) {
      cp.display(true, false);
    }
  }
};

/**
 * Hertz contact parameters between the groups (materials) of particles and the drivers.
 * See DriversContactParams for the Hooke law.
 */
class DriversHertzContactParams : public OperatorNode {
  ADD_SLOT(MultiMatParamsT<HertzParams>, multimat_hertz_cp, INPUT_OUTPUT, REQUIRED,
           DocString{"List of Hertz contact parameters for simulations with multiple materials"});
  ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(std::vector<uint32_t>, group, INPUT, OPTIONAL, DocString{"List of particle group indices."});
  ADD_SLOT(std::vector<int>, driver_id, INPUT, OPTIONAL, DocString{"List of drivers."});
  ADD_SLOT(std::vector<double>, driver_young, INPUT, OPTIONAL,
           DocString{"Young modulus of each driver material (one value per driver). Requires driver_poisson, and the "
                     "materials of the groups (young, poisson of multimat_contact_params_hertz)."});
  ADD_SLOT(std::vector<double>, driver_poisson, INPUT, OPTIONAL,
           DocString{"Poisson ratio of each driver material (one value per driver). Requires driver_young."});
  ADD_SLOT(std::vector<double>, dncut, INPUT, OPTIONAL, DocString{"List of dncut values."});
  ADD_SLOT(std::vector<double>, kt, INPUT, OPTIONAL, DocString{"List of kt values."});
  ADD_SLOT(std::vector<double>, kr, INPUT, OPTIONAL, DocString{"List of kr values."});
  ADD_SLOT(std::vector<double>, mu, INPUT, OPTIONAL, DocString{"List of mu values."});
  ADD_SLOT(std::vector<double>, fc, INPUT, OPTIONAL, DocString{"List of fc values."});
  ADD_SLOT(std::vector<double>, gamma, INPUT, OPTIONAL, DocString{"List of gamma values."});
  ADD_SLOT(std::vector<double>, damprate, INPUT, OPTIONAL, DocString{"List of damprate values."});
  ADD_SLOT(HertzParams, default_config, INPUT, OPTIONAL,
           DocString{"Hertz contact parameters of the pairs (group, driver) which are not listed."});
  ADD_SLOT(bool, verbosity, INPUT, false, DocString{"Print force field parameter details"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator fills the Hertz contact parameters between the groups (materials) of spheres and the drivers.
        The materials of the groups come from multimat_contact_params_hertz (young, poisson), the materials of the
        drivers are given with driver_young and driver_poisson (one value per driver).
        See drivers_contact_params for the Hooke law.

        YAML example:

          - drivers_contact_params_hertz:
             group:          [      0,     1 ]
             driver_id:      [      0,     0 ]
             kt:             [   8000, 12000 ]
             kr:             [    0.0,   0.1 ]
             mu:             [    0.5,   0.5 ]
             damprate:       [  0.999, 0.999 ]
             driver_young:   [ 200e9 ]
             driver_poisson: [  0.30 ]

        )EOF";
  }

  inline std::string operator_name() { return "drivers_contact_params_hertz"; }

 public:
  inline void execute() final {
    MultiMatParamsT<HertzParams>& cp = *multimat_hertz_cp;

    // We use the same data map to for drivers
    ParticleTypeMap driver_map;
    Drivers& drvs = *drivers;

    /** Build the driver map */
    for (int i = 0; i < int(drvs.get_size()); i++) {
      auto driver_type = drvs.type(i);
      std::string driver_name;
      if (driver_type == DRIVER_TYPE::RSHAPE) {
        RShapeDriver& D = drvs.get_typed_driver<RShapeDriver>(i);
        driver_name = D.shp_.name_;
      } else {
        driver_name = print(driver_type) + "_" + std::to_string(i);
      }
      driver_map[driver_name] = i;
    }

    HertzParams default_params;
    if (default_config.has_value()) default_params = *default_config;
    cp.setup_drivers(driver_map, default_params);

    if (group.has_value()) { /** not default config */
      auto& particle_groups = *group;
      const size_t number_of_pairs = particle_groups.size();

      auto check_lengths_match = [number_of_pairs, this]<typename Vec>(Vec& list, std::string cp_field_name) {
        if (number_of_pairs != list.size()) {
          color_log::error(
              this->operator_name(),
              "The length of the field \"" + cp_field_name +
                  "\" does not match the size of the other fields. group.size() = " + std::to_string(number_of_pairs) +
                  ", while " + cp_field_name + ".size() = " + std::to_string(list.size()) + ".");
        }
      };
      if (!driver_id.has_value() || !kt.has_value() || !mu.has_value() || !damprate.has_value()) {
        color_log::error(operator_name(), "driver_id, kt, mu and damprate are required with group.");
      }
      auto& drv_id = *driver_id;
      check_lengths_match(drv_id, "driver_id");
      check_lengths_match(*kt, "kt");
      check_lengths_match(*mu, "mu");
      check_lengths_match(*damprate, "damprate");
      if (kr.has_value()) check_lengths_match(*kr, "kr");
      if (fc.has_value() != dncut.has_value()) {
        color_log::error(operator_name(), "fc and dncut must be defined together.");
      }
      if (fc.has_value()) {
        check_lengths_match(*fc, "fc");
        check_lengths_match(*dncut, "dncut");
      }
      if (gamma.has_value()) check_lengths_match(*gamma, "gamma");

      /** Check input driver indexes */
      for (auto did : drv_id) {
        if (did >= int(drvs.get_size())) {
          color_log::error(operator_name(),
                           "driver id: " + std::to_string(did) +
                               " is out of range of the driver list: " + std::to_string(drvs.get_size()));
        }
        if (drvs.type(did) == DRIVER_TYPE::UNDEFINED) {
          color_log::error(operator_name(), "The driver type is undefined for id: " + std::to_string(did));
        }
      }

      for (size_t p = 0; p < number_of_pairs; p++) {
        HertzParams params = hertz_cp_detail::make_pair_params(
            p, *kt, kr.has_value() ? &*kr : nullptr, *mu, *damprate, fc.has_value() ? &*fc : nullptr,
            dncut.has_value() ? &*dncut : nullptr, gamma.has_value() ? &*gamma : nullptr);
        cp.register_driver(static_cast<int>(particle_groups[p]), drv_id[p], params);
      }
    }

    // Reduced Young modulus of each pair (group, driver), from the materials
    if (driver_young.has_value() || driver_poisson.has_value()) {
      if (!driver_young.has_value() || !driver_poisson.has_value()) {
        color_log::error(operator_name(), "driver_young and driver_poisson must be defined together.");
      }
      const int nd = cp.number_of_drivers_;
      const int ng = cp.number_of_groups_;
      if (static_cast<int>(driver_young->size()) != nd || static_cast<int>(driver_poisson->size()) != nd) {
        color_log::error(operator_name(), "driver_young and driver_poisson must have one value per driver (" +
                                              std::to_string(nd) + " drivers).");
      }
      for (int g = 0; g < ng; g++) {
        // The material of a group is kept in the parameters of the pair (group, group)
        const HertzParams& group_material = cp.multimat_cp_[cp.get_idx_multimat(g, g)];
        if (group_material.young_ <= 0.0) {
          color_log::error(operator_name(),
                           "The material of the group " + std::to_string(g) +
                               " is not defined: please use young and poisson in multimat_contact_params_hertz "
                               "before drivers_contact_params_hertz.");
        }
        for (int d = 0; d < nd; d++) {
          HertzParams& params = cp.drivers_cp_[cp.get_idx_drivers(g, d)];
          params.estar_ = compute_reduced_young_modulus(group_material.young_, group_material.poisson_,
                                                        (*driver_young)[d], (*driver_poisson)[d]);
          params.young_ = group_material.young_;
          params.poisson_ = group_material.poisson_;
        }
      }
    }

    cp.check_completeness(false, true);
    if (*verbosity) {
      cp.display(false, true);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(hertz_contact_params) {
  OperatorNodeFactory::instance()->register_factory("multimat_contact_params_hertz",
                                                    make_simple_operator<MultiMatHertzContactParams>);
  OperatorNodeFactory::instance()->register_factory("drivers_contact_params_hertz",
                                                    make_simple_operator<DriversHertzContactParams>);
}
}  // namespace exaDEM
