/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements. See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership. The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
 */

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>

#include <exanb/analytics/particle_cell_projection.h>
#include <exanb/core/grid_particle_field_accessor.h>
#include <exanb/compute/field_combiners.h>

#include <exaDEM/color_log.hpp>
#include <mpi.h>
#include <regex>

namespace exaDEM
{
  using namespace exanb;

  template< class GridT >
    class QuantitiesCellProjection : public OperatorNode
  {    
    using StringList = std::vector<std::string>;
    ADD_SLOT( MPI_Comm    , mpi             , INPUT );
    ADD_SLOT( GridT          , grid              , INPUT , REQUIRED );
    ADD_SLOT( double         , splat_size        , INPUT , -1.0, DocString{"Overlap width centered on the particle to calculate its contribution to neighboring cells"} );
    ADD_SLOT( StringList     , fields            , INPUT , StringList({".*"}) , DocString{"List of regular expressions to select fields to project"} );
    ADD_SLOT( long           , grid_subdiv       , INPUT_OUTPUT , 1 );
    ADD_SLOT( GridCellValues , grid_cell_values  , INPUT_OUTPUT );

    public:

    // -----------------------------------------------
    inline void execute ()  override final
    {
      using namespace ParticleCellProjectionTools;

      if( grid->number_of_cells() == 0 ) return;


      if( *splat_size == -1 ) 
      {
        // default value ... 
        *splat_size = 0.5 * grid->cell_size() / (*grid_subdiv);
      }

      if( *splat_size <= 0 )
      {
        color_log::error("quantities_cell_projection", "splat_size sould be superior to 0");
      }
      int rank=0;
      MPI_Comm_rank(*mpi, &rank);

      VelocityNormCombiner vnorm = {};
      ParticleCountCombiner count = {};

      auto proj_fields = make_field_tuple_from_field_set( grid->field_set, count, vnorm );
      auto field_selector = [flist = *fields] ( const std::string& name ) -> bool { for(const auto& f:flist) if( std::regex_match(name,std::regex(f)) ) return true; return false; } ;
      project_particle_fields_to_grid( ldbg, *grid, *grid_cell_values, *grid_subdiv, *splat_size, field_selector, proj_fields );
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        Project particle quantities onto a regular grid.

        Example: 

          global:
            simulation_analyses_frequency: 1000

          analyses:
            - timestep_paraview_file: "ParaviewOutputFiles/quantities_%010d.vtk"
            - message: { mesg: "Write " , endl: false }
            - print_dump_file:
                rebind: { mesg: filename }
                body:
                - message: { endl: true }
            - resize_grid_cell_values
            - quantities_cell_projection:
               splat_size: 0.99
               grid_subdiv: 2
               fields: [count, vnorm, vx, vy, vz, stress]
            - write_grid_vtklegacy

       )EOF";
    }    

  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(quantities_cell_projection)
  {
    OperatorNodeFactory::instance()->register_factory("quantities_cell_projection", make_grid_variant_operator< QuantitiesCellProjection > );
  }

}
