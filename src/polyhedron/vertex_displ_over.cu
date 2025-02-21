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
#include <exanb/core/grid.h>
#include <onika/math/basic_types.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>

#include <mpi.h>

#include <exanb/compute/reduce_cell_particles.h>
#include <exanb/mpi/particle_displ_over_async_request.h>
#include <exaDEM/shapes.hpp>
#include <exaDEM/backup_dem.h>
#include <exaDEM/traversal.hpp>

namespace exaDEM
{
  using namespace exanb;
  template <typename GridT> class VertexDisplacementOver : public OperatorNode
  {
    static constexpr FieldSet<field::_rx, field::_ry, field::_rz, field::_type, field::_orient> reduce_field_set{};

    // -----------------------------------------------
    // -----------------------------------------------
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(GridT, grid, INPUT);
    ADD_SLOT(double, threshold, INPUT, 0.0);
    ADD_SLOT(bool, async, INPUT, false);
    ADD_SLOT(shapes, shapes_collection, INPUT, DocString{"Collection of shapes"});
    ADD_SLOT(bool, result, OUTPUT);
    ADD_SLOT(DEMBackupData, backup_dem, INPUT);
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});
    ADD_SLOT(ParticleDisplOverAsyncRequest, particle_displ_comm, INPUT_OUTPUT);

  public:
    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
compute the distance between each particle in grid input and it's backup position in backup_dem input.
sets result output to true if at least one particle has moved further than threshold.
)EOF";
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute() override final
    {
      MPI_Comm comm = *mpi;
      const shapes &shps = *shapes_collection;

      // interest for auto here, is to be able to easily switch between single and double precision floats if needed.
      const double max_dist = *threshold;
      const double max_dist2 = max_dist * max_dist;

      auto [cell_ptr, cell_size] = traversal_real->info();

      particle_displ_comm->m_comm = *mpi;
      particle_displ_comm->m_request = MPI_REQUEST_NULL;
      particle_displ_comm->m_particles_over = 0;
      particle_displ_comm->m_all_particles_over = 0;
      particle_displ_comm->m_async_request = false;
      particle_displ_comm->m_request_started = false;

      ReduceMaxVertexDisplacementFunctor func = {backup_dem->m_data.data(), max_dist2, shps.data()};

      if (*async)
      {
        ldbg << "Async particle_displ_over => result set to false" << std::endl;
        particle_displ_comm->m_async_request = true;
        auto user_cb = onika::parallel::ParallelExecutionCallback{reduction_end_callback, &(*particle_displ_comm)};
        reduce_cell_particles(*grid, false, func, particle_displ_comm->m_particles_over, reduce_field_set, parallel_execution_context(), user_cb, cell_ptr, cell_size);
        particle_displ_comm->start_mpi_async_request();
        *result = false;
      }
      else
      {
        ldbg << "Nb part moved over " << max_dist << " (local) = " << particle_displ_comm->m_particles_over << std::endl;
        if (grid->number_of_cells() > 0)
        {
          auto user_cb = onika::parallel::ParallelExecutionCallback{};
          reduce_cell_particles(*grid, false, func, particle_displ_comm->m_particles_over, reduce_field_set, parallel_execution_context(), user_cb, cell_ptr, cell_size);
        }
        MPI_Allreduce(&(particle_displ_comm->m_particles_over), &(particle_displ_comm->m_all_particles_over), 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
        ldbg << "Nb part moved over " << max_dist << " (local/all) = " << particle_displ_comm->m_particles_over << " / " << particle_displ_comm->m_all_particles_over << std::endl;
        *result = (particle_displ_comm->m_all_particles_over > 0);
      }
    }

    static inline void reduction_end_callback(void *userData)
    {
      ::exanb::ldbg << "async CPU/GPU reduction done, start async MPI collective" << std::endl;
      auto *particle_displ_comm = (ParticleDisplOverAsyncRequest *)userData;
      assert(particle_displ_comm != nullptr);
      assert(particle_displ_comm->m_all_particles_over >= particle_displ_comm->m_particles_over);
      particle_displ_comm->start_mpi_async_request();
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("vertex_displ_over", make_grid_variant_operator<VertexDisplacementOver>); }

} // namespace exaDEM
