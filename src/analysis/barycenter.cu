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
#include <exanb/compute/reduce_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/core/grid.h>

#include <memory>
#include <exaDEM/color_log.hpp>
#include <exaDEM/traversal.h>
#include <exaDEM/AnalysisManager.hpp>
#include <exaDEM/barycenter.hpp>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_rx, field::_ry, field::_rz, field::_type>> class ParticleBarycenterAnalysis : public OperatorNode
  {

    static constexpr FieldSet<field::_rx,field::_ry,field::_rz,field::_type> reduce_field_set{};
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(GridT, grid, INPUT, REQUIRED);
    ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});
    ADD_SLOT(double, dt, INPUT, REQUIRED);
    ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Iteration number"});
    ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
    ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory, usually defined into io_config."});
    ADD_SLOT(std::string, name, INPUT, "ParticleBarycenter.txt", DocString{"Filename. Default is: ParticleBarycenter.txt"});
    ADD_SLOT(std::vector<int>, types, INPUT, REQUIRED , DocString{"List of types"});


    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        The purpose of this operator is to compute the barycenter of particles per type into a particular region.

        YAML example:

          - particle_barycenter:
             types: [0,1]
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

      if( list_of_types.size() == 0 ) lout << "[Analysis/barycenter] types is empty, this operator is skipped" << std::endl;

      const ReduceCellParticlesOptions rcpo = traversal_real->get_reduce_cell_particles_options();

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
        // Reduce over the subdomain
        ReduceParticleBarycenterTypeFunctor func = {prcsg, type};
        ParticleBarycenterTypeValue value = {0, {0.0,0.0,0.0}}; // int , Vec3d
        reduce_cell_particles(*grid, false, func, value, reduce_field_set, parallel_execution_context(), {}, rcpo);
        // Reduce over MPI processes
        double local[4] = {(double)(value.count),value.barycenter.x, value.barycenter.y, value.barycenter.z};
        double global[4] = {0.0, 0.0, 0.0, 0.0}; // count, x, y, z
        MPI_Reduce(&local, &global, 4, MPI_DOUBLE, MPI_SUM, 0, *mpi); 
        std::string var_name_rx = "Type[" + std::to_string(type) + "][rx]";
        std::string var_name_ry = "Type[" + std::to_string(type) + "][ry]";
        std::string var_name_rz = "Type[" + std::to_string(type) + "][rz]";
        // Compute means
        if( global[0] != 0.0) // There is at least one particle
        {
          for(int i = 0 ; i < 3 ; i++) 
          { 
            global[i+1] /= global[0]; // It compute the average values
          }
        }
        else 
        {
          color_log::warning("barycenter", "No particle into this region (particle_barycenter). Barycenter is set to {0,0,0}"); 
        }
        // Add data to the output file
        manager.add_element(var_name_rx, global[1], "%f");
        manager.add_element(var_name_ry, global[2], "%f");
        manager.add_element(var_name_rz, global[3], "%f");
      }
      manager.endl();
      manager.write();
    }
  };

  template <class GridT> using ParticleBarycenterAnalysisTmpl = ParticleBarycenterAnalysis<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(barycenter) { OperatorNodeFactory::instance()->register_factory("particle_barycenter", make_grid_variant_operator<ParticleBarycenterAnalysisTmpl>); }

} // namespace exaDEM
