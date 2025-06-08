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

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>> class DEMRcutMax : public OperatorNode
  {
    //      using ReduceFields = FieldSet<field::_radius>;
    //      static constexpr ReduceFields reduce_field_set {};
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(double, rcut_max, INPUT_OUTPUT, 0.0);

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final { return R"EOF(Fill rcut_max with the maximum of the radii. )EOF"; }

    public:
    inline void execute() override final
    {
      double &rmax = *rcut_max;
      auto cells = grid->cells();
      const IJK dims = grid->dimension();
      double rcm = 0;
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN (dims, i, loc, schedule(dynamic) reduction(max : rcm))
        {
          double *__restrict__ r = cells[i][field::radius];
          const size_t n = cells[i].size();
#     pragma omp simd
          for (size_t j = 0; j < n; j++)
          {
            rcm = std::max(rcm, 2*r[j]); 
          }
        }
        GRID_OMP_FOR_END
      } 
      MPI_Allreduce(MPI_IN_PLACE, &rcm, 1, MPI_DOUBLE, MPI_MAX, *mpi);
      rmax = std::max(rmax, rcm);
      std::cout << rmax << std::endl;
    } // namespace exaDEM
  };

  template <class GridT> using DEMRcutMaxTmpl = DEMRcutMax<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(dem_rcut_max) 
  { 
    OperatorNodeFactory::instance()->register_factory("dem_rcut_max", make_grid_variant_operator<DEMRcutMaxTmpl>); 
  }
}
