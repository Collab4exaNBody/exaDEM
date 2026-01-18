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
#include <exaDEM/shapes.hpp>

namespace exaDEM {

struct field_manager {
  bool set_t = false;  // type
  bool set_d = false;  // density
  bool set_v = false;  // velocity
  bool set_rnd_v = false;
  bool set_r = false;  // radius
  bool set_q = false;  // quaternion
  bool set_rnd_q = false;
  bool set_i = true;  // inertia (should be to true)
  bool set_ang_v = false;
  bool set_rnd_ang_v = false;
};

template <typename GridT, class = AssertGridHasFields<GridT, field::_type>>
class ParticleType : public OperatorNode {
  ADD_SLOT(ParticleTypeMap, particle_type_map, INPUT_OUTPUT);
  ADD_SLOT(std::vector<std::string>, type, INPUT, REQUIRED);

  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
        This operator fills particle type. 
        )EOF";
  }

 public:
  inline void execute() final {
    auto& typeMap = *particle_type_map;
    auto& types = *type;

    for (const auto& type_name : types) {
      const auto type_id = typeMap.size();
      typeMap[type_name] = type_id;
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(particle_type) {
  OperatorNodeFactory::instance()->register_factory("particle_type", make_grid_variant_operator<ParticleType>);
}

}  // namespace exaDEM
