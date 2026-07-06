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

// onika
#include <onika/memory/allocator.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// exanb
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/particle_type_id.h>

// compute
#include <exanb/compute/compute_cell_particles.h>

// exaDEM
#include <exaDEM/color_log.hpp>

namespace exaDEM {
template <typename T>
using CudaMMVector = onika::memory::CudaMMVector<T>;

struct SetGroupFunctor {
  const uint32_t* lut;  // type_id -> group value
  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint32_t type, uint32_t& group) const { group = lut[type]; }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ComputeCellParticlesTraits<exaDEM::SetGroupFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT, field::_type, field::_group>>
class SetGroupOperator : public OperatorNode {
  using ComputeFields = FieldSet<field::_type, field::_group>;
  static constexpr ComputeFields compute_field_set{};

  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(ParticleTypeMap, particle_type_map, INPUT, REQUIRED);
  ADD_SLOT(std::vector<std::string>, type, INPUT, REQUIRED, DocString{"List of particle type names"});
  ADD_SLOT(std::vector<uint32_t>, group, INPUT, REQUIRED,
           DocString{"Group index associated to each type (same order as 'type')"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Assigns the group field to every particle according to its type.

        YAML examples:

          init_polyhedra:
            - set_group:
               type:  [ alpha3, Octahedron, Cube ]
               group: [      0,          1,    0 ]

          init_spheres:
            - set_group:
               type:  [ Sphere1, Sphere2, Sphere3 ]
               group: [       0,       1,       0 ]
        )EOF";
  }

  inline void execute() final {
    const auto& type_map = *particle_type_map;
    const auto& type_names = *type;
    const auto& group_values = *group;

    if (type_names.size() != group_values.size()) {
      color_log::error("set_group", "'type' and 'group' lists must have the same length.");
    }

    // Build type_id -> group lookup table in unified memory
    const size_t n_types = type_map.size();
    CudaMMVector<uint32_t> lut(n_types, 0);

    for (size_t i = 0; i < type_names.size(); i++) {
      const auto it = type_map.find(type_names[i]);
      if (it == type_map.end()) {
        color_log::error("set_group", "Type [" + type_names[i] + "] is not defined in particle_type_map.");
      }
      lut[it->second] = group_values[i];
    }

    SetGroupFunctor func = {lut.data()};
    compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context("set_group"));
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(set_group) {
  OperatorNodeFactory::instance()->register_factory("set_group", make_grid_variant_operator<SetGroupOperator>);
}
}  // namespace exaDEM
