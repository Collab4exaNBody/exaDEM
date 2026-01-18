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
#include <exanb/compute/reduce_cell_particles.h>
#include <mpi.h>
#include <exaDEM/color_log.hpp>

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT, field::_homothety>>
class CheckHomothety : public OperatorNode {
  //      using ReduceFields = FieldSet<field::_radius>;
  //      static constexpr ReduceFields reduce_field_set {};

  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(double, threshold, INPUT, 1e-14);

  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
        Check if the field homothety is well defined, i.e homothety > 0.0. 

        YAML example [no option]:

          - check_homothety
      )EOF";
  }

  inline std::string operator_name() {
    return "check_homothety";
  }

 public:

  inline void execute() final {
    MPI_Comm comm = *mpi;
    double th = *threshold;

    if (th < 0.0) {
      color_log::error(operator_name(), "Slot [threshold=" + std::to_string(th) +"] should be positive.");
    }

    auto& gridAccessor = *grid;
    auto cells = gridAccessor.cells();
    const IJK dims = gridAccessor.dimension();
    uint64_t n_particles = 0;
#   pragma omp parallel
    {
      GRID_OMP_FOR_BEGIN(dims, i, loc, schedule(dynamic) reduction(+: n_particles)){
        if(gridAccessor.is_ghost_cell(i)) {
          continue;
        }
        double* __restrict__ h = cells[i][field::homothety];
        const size_t n = cells[i].size();
#       pragma omp simd
        for (size_t j = 0; j < n; j++) {
          if (h[j] <= th) {
            n_particles++;
          } 
        }
      }
      GRID_OMP_FOR_END
    }

    MPI_Allreduce(MPI_IN_PLACE, &n_particles, 1, MPI_UINT64_T, MPI_SUM, comm);

    if (n_particles > 0) {
      color_log::error(operator_name(), std::to_string(n_particles) + " particles are not defined corretly.");
    }
  }  
};

// === register factories ===
ONIKA_AUTORUN_INIT(check_rcut) {
  OperatorNodeFactory::instance()->register_factory("check_homothety", make_grid_variant_operator<CheckHomothety>);
}
}  // namespace exaDEM
