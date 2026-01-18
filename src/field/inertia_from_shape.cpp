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
#include <exaDEM/shapes.hpp>

namespace exaDEM {
template <typename GridT,
          class = AssertGridHasFields<GridT, field::_inertia, field::_radius, field::_mass, field::_homothety>>
class PolyhedraUpdateInertia : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});

 public:
  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
        This operator updates the inertia field.

        YAML example [no option]:

          - inertia_from_shape
        )EOF";
  }

  inline void execute() final {
    auto cells = grid->cells();
    const IJK dims = grid->dimension();
    auto& sphs = *shapes_collection;

#pragma omp parallel
    {
      GRID_OMP_FOR_BEGIN(dims, i, loc, schedule(dynamic)) {
        auto* __restrict__ m = cells[i][field::mass];
        auto* __restrict__ inertia = cells[i][field::inertia];
        auto* __restrict__ t = cells[i][field::type];
        auto* __restrict__ h = cells[i][field::homothety];
        const size_t n = cells[i].size();
#pragma omp simd
        for (size_t j = 0; j < n; j++) {
          inertia[j] = m[j] * sphs[t[j]]->get_Im(h[j]);
        }
      }
      GRID_OMP_FOR_END
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(inertia_from_shape) {
  OperatorNodeFactory::instance()->register_factory("inertia_from_shape",
                                                    make_grid_variant_operator<PolyhedraUpdateInertia>);
}
}  // namespace exaDEM
