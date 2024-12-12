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
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <memory>
#include <exaDEM/cell_list_wrapper.hpp>
#include <exaDEM/AnalysisManager.hpp>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT>> class CounterAnalysis : public OperatorNode
  {

    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(CellListWrapper, cell_list, INPUT_OUTPUT, DocString{"list of non empty cells within the current grid"});
    ADD_SLOT(double, dt, INPUT, REQUIRED);
    ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Iteration number"});
    ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
    ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory"});
    ADD_SLOT(std::string, name, INPUT, "counter_undefined.txt", DocString{"Filename. Default is: counter_undefined.txt"});
    ADD_SLOT(std::vector<int>, types, INPUT, REQUIRED , DocString{"List of types"});


    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator computes forces related to the gravity.
        )EOF";
    }

    inline void execute() override final
    {
      analysis::AnalysisFileManager manager = {};
      manager.set_path ((*dir_name) + "/Analysis");
      manager.set_filename (*name);
      double t = (*dt) * (*timestep);
      manager.add_element("Time", t, "%3f");

      // now, fill the radius field
      if (region.has_value())
      {
        ParticleRegionCSGShallowCopy prcsg = *region;
        UpdateRadiusPolyhedronFunctor func = {prcsg, onika::cuda::vector_data(r)};
        if (!particle_regions.has_value())
        {
          fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
        }

        if (region->m_nb_operands == 0)
        {
          ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
          region->build_from_expression_string(particle_regions->data(), particle_regions->size());
        }
        compute_cell_particles(*grid, false, func, compute_region_field_set, parallel_execution_context());
      }
      else
      {
        ParticleRegionCSGShallowCopy prcsg;
        UpdateRadiusPolyhedronFunctor func = {prcsg, onika::cuda::vector_data(r)};
        compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context());
      }

      manager.endl();
      manager.write();
    }
  };

  template <class GridT> using CounterAnalysisTmpl = CounterAnalysis<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("counter", make_grid_variant_operator<CounterAnalysisTmpl>); }

} // namespace exaDEM
