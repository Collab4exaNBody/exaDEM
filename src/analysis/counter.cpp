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
//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/compute/reduce_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/core/grid.h>

#include <memory>
#include <exaDEM/cell_list_wrapper.hpp>
#include <exaDEM/AnalysisManager.hpp>
#include <exaDEM/counter.hpp>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_rx, field::_ry, field::_rz, field::_type>> class ParticleCounterAnalysis : public OperatorNode
  {

    static constexpr FieldSet<field::_rx,field::_ry,field::_rz,field::_type> reduce_field_set{};
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(CellListWrapper, cell_list, INPUT_OUTPUT, DocString{"list of non empty cells within the current grid"});
    ADD_SLOT(double, dt, INPUT, REQUIRED);
    ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Iteration number"});
    ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
    ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory, usually defined into io_config."});
    ADD_SLOT(std::string, name, INPUT, "ParticleCounter.txt", DocString{"Filename. Default is: ParticleCounter.txt"});
    ADD_SLOT(std::vector<int>, types, INPUT, REQUIRED , DocString{"List of types"});


    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        The purpose of this operator is to count the number of particles per type in a particular region.
        )EOF";
    }

    inline void execute() override final
    {
      analysis::AnalysisFileManager manager = {};
      manager.set_path ((*dir_name) + "/ExaDEMAnalyses");
      manager.set_filename (*name);
      double t = (*dt) * (*timestep);
      manager.add_element("Time", t, "%3f");
      auto& list_of_types = (*types);

      if( list_of_types.size() == 0 ) lout << "[Analysis/counter] types is empty, this operator is skipped" << std::endl;

      auto [cell_ptr, cell_size] = cell_list->info();

      // iterate over types -- it could be optimized by computing all types in a single call of reduce_cell_particles.
      for(size_t i = 0 ; i < list_of_types.size() ; i++)
      {
        uint16_t type = list_of_types[i];
        ParticleRegionCSGShallowCopy prcsg; 
        // now, fill the radius field
        if (region.has_value())
        {
          if (!particle_regions.has_value())
          {
            fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
          }

          if (region->m_nb_operands == 0)
          {
            ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
            region->build_from_expression_string(particle_regions->data(), particle_regions->size());
          }
          prcsg =  *region;
        }
        ReduceParticleCounterTypeFunctor func = {prcsg, type};
        int count = 0;
        reduce_cell_particles(*grid, false, func, count, reduce_field_set, parallel_execution_context(), {}, cell_ptr, cell_size);
        uint64_t local(count), global(0);
        MPI_Reduce(&local, &global, 1, MPI_UINT64_T, MPI_SUM, 0, *mpi); 
        std::string var_name = "Type[" + std::to_string(type) + "]";
        manager.add_element(var_name, count, "%d");
      }
      manager.endl();
      manager.write();
    }
  };

  template <class GridT> using ParticleCounterAnalysisTmpl = ParticleCounterAnalysis<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("particle_counter", make_grid_variant_operator<ParticleCounterAnalysisTmpl>); }

} // namespace exaDEM
