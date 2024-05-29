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

#include <exanb/particle_neighbors/chunk_neighbors_execute_2.h>

//#include <exaDEM/interactions_PP.h>
//#include <exaDEM/neighbor_friction.h>
#include <../forcefield/include/exaDEM/interactions_PP.h>

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
  
  __global__ void setGPU(int* pa, 
  			int* cella,
  			int* pb,
  			int* cellb,
  			double* ftx,
  			double* fty,
  			double* ftz,
  			onika::memory::CudaMMVector<int> pa2, 
  			onika::memory::CudaMMVector<int> cella2,
  			onika::memory::CudaMMVector<int> pb2,
  			onika::memory::CudaMMVector<int> cellb2,
  			onika::memory::CudaMMVector<double> ftx2,
  			onika::memory::CudaMMVector<double> fty2,
  			onika::memory::CudaMMVector<double> ftz2,
  			int size)
  {
  	
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	if(idx < size)
  	{
  		
  		auto pa3 = onika::cuda::vector_data(pa2);
  		auto cella3 = onika::cuda::vector_data(cella2);
  		auto pb3 = onika::cuda::vector_data(pb2);
  		auto cellb3 = onika::cuda::vector_data(cellb2);
  		auto ftx3 = onika::cuda::vector_data(ftx2);
  		auto fty3 = onika::cuda::vector_data(fty2);
  		auto ftz3 = onika::cuda::vector_data(ftz2);
  		
  		pa[idx] = pa3[idx];
  		
  		cella[idx] = cella3[idx];
  		
  		pb[idx] = pb3[idx];
  		
  		cellb[idx] = cellb3[idx];
  		ftx[idx] = ftx3[idx];
  		fty[idx] = fty3[idx];
  		ftz[idx] = ftz3[idx];
  	
  	}
  }
  

  template<typename GridT>
  struct ChunkNeighborsContact : public OperatorNode
  {
  
    using intVector = onika::memory::CudaMMVector<int>;
      
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
    
    ADD_SLOT( Interactions_PP     , interactions_PP , INPUT_OUTPUT);//HOOKE_FORCE_GPU

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator builds the neighbor lists used to detect contacts between particles.
        )EOF";
		}

    inline void execute () override final
    {
    	printf("CHUNK START\n");
    	std::vector< std::vector< std::vector< std::pair<int,int>>>> cell_particles_nbh;//HOOKE_FORCE_GPU
    	std::vector< std::vector<int>> id_cell_particles_nbh;//HOOKE_FORCE_GPU
    	std::vector< std::vector<std::vector <int>>> id2_cell_particles_nbh;//HOOKE_FORCE_GPU
    	
    	//HOOKE_FORCE_GPU
    	Interactions_PP& interactions_new = *interactions_PP;
    	Interactions_PP interactions_old = interactions_new;
    	
    	interactions_old.maj_friction();
    	
    	interactions_new.reset();
    	//HOOKE_FORCE_GPU
    
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
        chunk_neighbors_execute2(ldbg,*chunk_neighbors,*grid,*amr,*amr_grid_pairs,*config,*chunk_neighbors_scratch,cs,cs_log2,*nbh_dist_lab, xform, gpu_enabled, no_z_order, cell_particles_nbh, id_cell_particles_nbh, id2_cell_particles_nbh, nbh_filter );
      }
      else
      {
        NullXForm xform = { };
        chunk_neighbors_execute2(ldbg,*chunk_neighbors,*grid,*amr,*amr_grid_pairs,*config,*chunk_neighbors_scratch,cs,cs_log2,*nbh_dist_lab, xform, gpu_enabled, no_z_order, cell_particles_nbh , id_cell_particles_nbh, id2_cell_particles_nbh, nbh_filter );
      }
      
      //HOOKE_FORCE_GPU
      for(int i= 0; i < cell_particles_nbh.size(); i++){
      	int cell= i;
      	for(int j= 0; j < cell_particles_nbh[i].size(); j++){
      		int particle= j;
      		auto ida = id_cell_particles_nbh[i][j];
      		std::vector< std::pair<int, int>> nbh= cell_particles_nbh[i][j];
      		std::vector<int> idb = id2_cell_particles_nbh[i][j];
      		if(nbh.size() > 0){
      			interactions_new.add_particle(particle, cell, nbh, ida, idb);
      		}
      	}
      }
     //interactions_new.test();
      interactions_new.quickSort();
      
      int p = interactions_new.print();
      
      interactions_new.set();
      
      int p2 = interactions_new.print();
      
      printf("OLD: %d NEW: %d\n", p, p2);
            
      interactions_new.init_friction(interactions_old);
      
      interactions_new.init_GPU();
      
      
	int size = interactions_new.nb_interactions;
	int blockSize = 128;
	int numBlocks;
	if(size % blockSize == 0){ numBlocks = size/blockSize;}
	else if(size / blockSize < 1){ numBlocks=1; blockSize = size;}
	else { numBlocks = int(size/blockSize)+1; }
      
      setGPU<<<numBlocks, blockSize>>>(interactions_new.pa_GPU2.data(), interactions_new.cella_GPU2.data(), interactions_new.pb_GPU2.data(), interactions_new.cellb_GPU2.data(), interactions_new.ftx_GPU2.data(), interactions_new.fty_GPU2.data(), interactions_new.ftz_GPU2.data(), interactions_new.pa_GPU, interactions_new.cella_GPU, interactions_new.pb_GPU, interactions_new.cellb_GPU, interactions_new.ftx_GPU, interactions_new.fty_GPU, interactions_new.ftz_GPU, size);
      cudaDeviceSynchronize();
      printf("CHUNK FINISH\n");
      			 
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("chunk_neighbors_contact", make_grid_variant_operator< ChunkNeighborsContact > );
  }

}

