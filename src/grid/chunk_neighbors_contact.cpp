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

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/amr/amr_grid.h>
#include <exanb/core/grid.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/amr/amr_grid_algorithm.h>
#include <exanb/core/particle_type_pair.h>

#include <onika/cuda/cuda_context.h>
#include <onika/memory/allocator.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_config.h>
#include <exanb/particle_neighbors/chunk_neighbors_scratch.h>
#include <exanb/particle_neighbors/chunk_neighbors_host_write_accessor.h>

#include <exanb/core/domain.h>
#include <exanb/core/xform.h>

#include <exanb/particle_neighbors/chunk_neighbors_execute.h>

namespace exaDEM
{
  using namespace exanb;

  template<class CellsT>
  struct ContactNeighborFilterFunc
  {
    CellsT cells;
    const double rcut_inc = 0.0;
    inline bool operator () (double d2, double rcut2,size_t cell_a,size_t p_a,size_t cell_b,size_t p_b) const
    {
      assert( cell_a!=cell_b || p_a!=p_b );
      const double r_a = cells[cell_a][field::radius][p_a];
      const double r_b = cells[cell_b][field::radius][p_b];
      const double rmax = r_a + r_b + rcut_inc;
      const double rmax2 = rmax * rmax;
      return d2 > 0.0 && d2 < rmax2 && d2 < rcut2 ;
    }
  };

  template<typename GridT>
  struct ChunkNeighborsContact : public OperatorNode
  {
#ifdef XSTAMP_CUDA_VERSION
    ADD_SLOT( onika::cuda::CudaContext , cuda_ctx , INPUT , OPTIONAL );
#endif

    ADD_SLOT( GridT               , grid            , INPUT );
    ADD_SLOT( AmrGrid             , amr             , INPUT );
    ADD_SLOT( AmrSubCellPairCache , amr_grid_pairs  , INPUT );
    ADD_SLOT( Domain              , domain          , INPUT );
    ADD_SLOT( double              , rcut_inc        , INPUT );  // value added to the search distance to update neighbor list less frequently
    ADD_SLOT( double              , nbh_dist_lab    , INPUT );  // value added to the search distance to update neighbor list less frequently
    ADD_SLOT( GridChunkNeighbors  , chunk_neighbors , INPUT_OUTPUT );

    ADD_SLOT( ChunkNeighborsConfig, config , INPUT, ChunkNeighborsConfig{} );
    ADD_SLOT( ChunkNeighborsScratchStorage, chunk_neighbors_scratch, PRIVATE );

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator builds the neighbor lists used to detect contacts between particles.
        )EOF";
		}

    inline void execute () override final
    {
      unsigned int cs = config->chunk_size;
      unsigned int cs_log2 = 0;
      while( cs > 1 )
      {
        assert( (cs&1)==0 );
        cs = cs >> 1;
        ++ cs_log2;
      }
      cs = 1<<cs_log2;
      //ldbg << "cs="<<cs<<", log2(cs)="<<cs_log2<<std::endl;
      if( cs != static_cast<size_t>(config->chunk_size) )
      {
        lerr<<"chunk_size is not a power of two"<<std::endl;
        std::abort();
      }

      //const bool gpu_enabled = parallel_execution_context()->has_gpu_context();

			bool gpu_enabled = ( global_cuda_ctx() != nullptr );
      if( gpu_enabled ) gpu_enabled = global_cuda_ctx()->has_devices(); 

      auto cells = grid->cells();
      ContactNeighborFilterFunc<decltype(cells)> nbh_filter { cells , *rcut_inc };
      static constexpr std::false_type no_z_order = {};

      if( ! domain->xform_is_identity() )
      {
        LinearXForm xform = { domain->xform() };
        chunk_neighbors_execute(ldbg,*chunk_neighbors,*grid,*amr,*amr_grid_pairs,*config,*chunk_neighbors_scratch,cs,cs_log2,*nbh_dist_lab, xform, gpu_enabled, no_z_order, nbh_filter );
      }
      else
      {
        NullXForm xform = { };
        chunk_neighbors_execute(ldbg,*chunk_neighbors,*grid,*amr,*amr_grid_pairs,*config,*chunk_neighbors_scratch,cs,cs_log2,*nbh_dist_lab, xform, gpu_enabled, no_z_order, nbh_filter );
      }
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("chunk_neighbors_contact", make_grid_variant_operator< ChunkNeighborsContact > );
  }

}

