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
using namespace exanb;

template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>>
class CheckRadius : public OperatorNode {
  //      using ReduceFields = FieldSet<field::_radius>;
  //      static constexpr ReduceFields reduce_field_set {};

  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(double, rcut_max, INPUT_OUTPUT, 0.0);

  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
        Check if  - rcut_max is different of 0
                  - rcut_max is < particle rcut
                  - 0.5 cell size  < particle rcut  
                 
        YAML example [no option]:

          - check_rcut
      )EOF";
  }

  inline std::string operator_name() {
    return "check_rcut";
  }

 public:
  void check_slots() {
    if (*rcut_max <= 0.0) {
      color_log::error(operator_name(), "rmax is not correctly defined (rcut max <= 0.0)");
    }
  }

  inline void execute() final {
    check_slots();
    MPI_Comm comm = *mpi;
    double rmax = *rcut_max;

    double half_cell_size =  0.5 * grid->cell_size();
    auto cells = grid->cells();
    const IJK dims = grid->dimension();
    double rcm = 0.0;
    uint64_t count_oversize_particles = 0;
    uint64_t count_unsuitable_particle_grid = 0;

#   pragma omp parallel
    {
      GRID_OMP_FOR_BEGIN(dims, i, loc,
                         schedule(dynamic) reduction(max : rcm)
                         reduction(+: count_oversize_particles,
                                   count_unsuitable_particle_grid) ){
        double* __restrict__ r = cells[i][field::radius];
        const size_t n = cells[i].size();
#       pragma omp simd
        for (size_t j = 0; j < n; j++) {
          if (r[j]>rmax) {
            count_oversize_particles++;
          }
          if (r[j]>half_cell_size) {
            count_unsuitable_particle_grid++;
          }
          rcm = std::max(rcm, r[j]);
        }
      }
      GRID_OMP_FOR_END
    }

    MPI_Allreduce(MPI_IN_PLACE, &rcm, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(MPI_IN_PLACE, &count_oversize_particles, 1, MPI_UINT64_T, MPI_SUM, comm);
    MPI_Allreduce(MPI_IN_PLACE, &count_unsuitable_particle_grid, 1, MPI_UINT64_T, MPI_SUM, comm);

    if (count_unsuitable_particle_grid>0) {
      color_log::error(operator_name(),
                       std::to_string(count_unsuitable_particle_grid)
                       + " particles are larger than 1/2 cell size: "
                       + std::to_string(half_cell_size));
    }

    if (count_oversize_particles>0) {
      color_log::error(operator_name(),
                       std::to_string(count_oversize_particles)
                       + " particles are larger than rmax: "
                       + std::to_string(rmax));
    }
  }
};  // namespace exaDEM

template <class GridT>
using CheckRadiusTmpl = CheckRadius<GridT>;

// === register factories ===
ONIKA_AUTORUN_INIT(check_rcut) {
  OperatorNodeFactory::instance()->register_factory("check_rcut", make_grid_variant_operator<CheckRadiusTmpl>);
}
}
