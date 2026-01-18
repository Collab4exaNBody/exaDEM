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

namespace exaDEM {

template <typename GridT, class = AssertGridHasFields<GridT, field::_radius, field::_type>>
class SetMultipleRadius : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(std::vector<double>, radius, INPUT, REQUIRED, DocString{"Array of radius values"});
  ADD_SLOT(double, rcut_max, INPUT_OUTPUT, 0.0, DocString{"rcut_max"});

 public:
  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
        DEPRECIATED, please use set_fields.
        This operator applies various radius according to their material properties.
        )EOF";
  }

  inline void execute() final {
    for (auto& r : *radius) {
      *rcut_max = std::max(*rcut_max, 2 * r);
    }

    auto cells = grid->cells();
    const IJK dims = grid->dimension();
#pragma omp parallel
    {
      GRID_OMP_FOR_BEGIN(dims, i, loc, schedule(dynamic)) {
        // double* __restrict__ m = cells[i][field::mass];
        double* __restrict__ r = cells[i][field::radius];
        auto* __restrict__ myType = cells[i][field::type];
        const double* rad = (*radius).data();
        const size_t n = cells[i].size();
#pragma omp simd
        for (size_t j = 0; j < n; j++) {
          r[j] = rad[myType[j]];  // 4/3 * pi * r^3 * d
        }
      }
      GRID_OMP_FOR_END
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(set_radius_multiple_materials) {
  OperatorNodeFactory::instance()->register_factory("set_radius_multiple_materials",
                                                    make_grid_variant_operator<SetMultipleRadius>);
}

}  // namespace exaDEM
