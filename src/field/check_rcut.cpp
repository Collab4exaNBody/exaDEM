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

  template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>> class CheckRadius : public OperatorNode
  {
    //      using ReduceFields = FieldSet<field::_radius>;
    //      static constexpr ReduceFields reduce_field_set {};

    ADD_SLOT(GridT, grid, INPUT);
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(double, rcut_max, INPUT_OUTPUT, 0.0);

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final { return R"EOF(Check if the rcut_max is different of 0 or if the rcut_max is < particle rcut. )EOF"; }

    public:

    void check_slots()
    {
      if(*rcut_max <= 0.0) 
      {
        lout << "\033[1;31m[check_rcut, ERROR] rmax is not correctly defined (rcut max <= 0.0)\033[0m" << std::endl;
        std::exit(0);
      }
    }

    inline void execute() override final
    {
      check_slots();
      MPI_Comm comm = *mpi;
      double rmax = *rcut_max;

      auto cells = grid->cells();
      const IJK dims = grid->dimension();
      double rcm = 0.0;
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN (dims, i, loc, schedule(dynamic) reduction(max : rcm))
        {
          double *__restrict__ r = cells[i][field::radius];
          const size_t n = cells[i].size();
#     pragma omp simd
          for (size_t j = 0; j < n; j++)
          {
            rcm = std::max(rcm, r[j]); // 4/3 * pi * r^3 * d
          }
        }
        GRID_OMP_FOR_END
      }
 
      MPI_Allreduce(MPI_IN_PLACE, &rcm, 1, MPI_DOUBLE, MPI_MAX, comm);

      if ( rcm > rmax )
      {
        lout << "\033[1;31m[check_rcut, ERROR] At least one particle has a radius larger than the maximum radius cutoff\033[0m" << std::endl;       
        std::exit(0);
      }
    } // namespace exaDEM
  }
  ;

  template <class GridT> using CheckRadiusTmpl = CheckRadius<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(check_rcut) { OperatorNodeFactory::instance()->register_factory("check_rcut", make_grid_variant_operator<CheckRadiusTmpl>); }
}
