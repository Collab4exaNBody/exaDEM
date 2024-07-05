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
#include <exanb/particle_neighbors/neighbor_filter_func.h>

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
  
  
  template< class GridT > ONIKA_HOST_DEVICE_FUNC bool nbh_filter_GPU(GridT* cells,
  						double rcut_inc,
  						double d2,
  						double rcut2,
  						size_t cell_a,
  						size_t p_a,
  						size_t cell_b,
  						size_t p_b)
  {
      const double r_a = cells[cell_a][field::radius][p_a];
      const double r_b = cells[cell_b][field::radius][p_b];
      const double rmax = r_a + r_b + rcut_inc;
      const double rmax2 = rmax * rmax;
      return d2 > 0.0 && d2 < rmax2 && d2 < rcut2 ;
  }
  
  
 template< class GridT > __global__ void neighbor_GPU(GridT* cells, 
 							int* cells2,
 							int* cells_size,
 							int* cells_start,
 							int* cells_tot,
 							Mat3d xform,
 							const double dist_lab,
 							double rcut_inc,
 							int* cells_b,
 							int* cells_b_size,
 							int* result,
 							int* result_start
  							)
  {
  	int cell = cells2[blockIdx.x];
  	int size = cells_size[blockIdx.x];
  	int start = cells_start[blockIdx.x];
  	int tot = cells_tot[blockIdx.x];
  	int res_start = result_start[blockIdx.x];
  	
  	if(threadIdx.x < size)
  	{
  		int p_a = threadIdx.x;
  		
  		for(int i = start; i < start + tot; i++)
  		{
  			int cell_b = cells_b[i];
  			
  			for(int p_b = 0; p_b < cells_b_size[i]; p_b++)
  			{
  				double rcut2 = dist_lab * dist_lab;
  				
  			 	const Vec3d dr = { cells[cell][field::rx][p_a] - cells[cell_b][field::rx][p_b] , cells[cell][field::ry][p_a] - cells[cell_b][field::ry][p_b] , cells[cell][field::rz][p_a] - cells[cell_b][field::ry][p_b] };
                    		double d2 = norm2(xform * dr );
  				if(nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell, p_a, cell_b, p_b)) { /*result[res_start + threadIdx.x * p_b] = 1;*/}
  				else{ /*result[res_start + threadIdx.x * p_b] = 0;*/}
  				
  				result_start[blockIdx.x]++;
  				
  			}
  		}
  	}
  	
  }
  
  template< class GridT > __global__ void kernel_GPU(GridT* cells,
  							int cell,
  							int size,
  							int* result,
  							int* cellb,
  							int* pb,
  							const double dist_lab,
  							Mat3d xform,
  							double rcut_inc)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	if(idx < size)
  	{
  		int p_a = (int)(threadIdx.x/size);
  		int nbh = threadIdx.x - (p_a * size);
  		
  		int cell_b = cellb[nbh];
  		int p_b = pb[nbh];
  		
  		double rcut2 = dist_lab * dist_lab;
  		
  		const Vec3d dr = { cells[cell][field::rx][p_a] - cells[cell_b][field::rx][p_b] , cells[cell][field::ry][p_a] - cells[cell_b][field::ry][p_b] , cells[cell][field::rz][p_a] - cells[cell_b][field::ry][p_b] };
                double d2 = norm2(xform * dr );
  		
  		if(nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell, p_a, cell_b, p_b)) { result[idx] = 1;}
  		else{ result[idx] = 1;}
  		
  	}
  }
 
  
  template< class GridT > __global__ void kernel_GPU2(GridT* cells,
  							int size,
  							int* cells_ids,
  							int* cells_total_size,
  							int* cells_total,
  							int* cells_incr,
  							int* cells_nb_nbh,
  							int* cells_start,
  							int* cells_256_div,
  							int* cells_nbh_ids,
  							int* cells_nbh_size,
  							const double dist_lab,
  							Mat3d xform,
  							double rcut_inc,
  							int* pa,
  							int* cella,
  							int* pb,
  							int* cellb,
  							int* nb_interactions,
  							int* interactions_blocks,
  							int* interactions_threads)
  {
  
  	int cell = cells_ids[blockIdx.x];
  	int total_size = cells_total_size[blockIdx.x];
  	int total = cells_total[blockIdx.x];
  	int incr = cells_incr[blockIdx.x];
  	int nb_nbh = cells_nb_nbh[blockIdx.x];
  	int start = cells_start[blockIdx.x];
  	int div = cells_256_div[blockIdx.x];
  	int interactions = 0;
  	
  	for(int i = 0; i < div; i++)
  	{
  		int idx = 256*i + threadIdx.x;
  		if(idx < total_size)
  		{
  			int p_a = (int)(idx / total);
  			int idx2 = idx - (p_a * total);
  			int cell_b;
  			int p_b;
  			bool b2 = true;
  			int incr2 = 0;
  			int incr3 = 0;
  			
  			for(int j = start; j < start + nb_nbh; j++)
  			{
  				incr2+= cells_nbh_size[j];
  				if(b2 && idx2 < incr2)
  				{
  					cell_b = cells_nbh_ids[j];
  					p_b = idx2 - incr3;
  					b2 = false;
  				}
  				
  				incr3+= cells_nbh_size[j];
  			}
  			
  			double rcut2 = dist_lab * dist_lab;
  		
  			const Vec3d dr = { cells[cell][field::rx][p_a] - cells[cell_b][field::rx][p_b] , cells[cell][field::ry][p_a] - cells[cell_b][field::ry][p_b] , cells[cell][field::rz][p_a] - cells[cell_b][field::rz][p_b] };
                	double d2 = norm2( xform * dr );
                
                	//printf("IDX : %d PA: %d, CELLA: %d CELLB: %d, PB: %d\n", idx, p_a, cell, cell_b, p_b);
  			if(nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell, p_a, cell_b, p_b)) 
  			{ 
  				//printf("CARLO\n");
  				pa[incr + interactions + div*threadIdx.x] = p_a;
  				cella[incr + interactions + div*threadIdx.x] = cell;
  				pb[incr + interactions + div*threadIdx.x] = p_b;
  				cellb[incr + interactions + div*threadIdx.x] = cell_b;
  				atomicAdd(&nb_interactions[0], 1);
  				interactions++;
  			}
  		}
  		
  		
  	}
  	
  	interactions_threads[ 256*blockIdx.x + threadIdx.x ] = interactions;
  	atomicAdd(&interactions_blocks[blockIdx.x], interactions);
  	 
  }
  		
  		
  __global__ void kernel_GPU3( int* cells_incr, int* incr_threads, int* interactions_threads, int* cells_256_div, int* pa, int* cella, int* pb, int* cellb, int* cella_final, int* pa_final, int* cellb_final, int* pb_final, int nb_interactions )
  {
  	
  	int incr = cells_incr[blockIdx.x];
  	int div = cells_256_div[blockIdx.x];
  	
  	for(int i = 0; i < interactions_threads[ 256*blockIdx.x + threadIdx.x ]; i++)
  	{
  		pa_final[ nb_interactions + incr_threads[256*blockIdx.x + threadIdx.x] + i ] = 0;//pa[incr + i + div*threadIdx.x];
  		//pb_final[ nb_interactions + incr_threads[256*blockIdx.x + threadIdx.x] ] = pb[incr + i + div*threadIdx.x];
  		//cella_final[ nb_interactions + incr_threads[256*blockIdx.x + threadIdx.x] ] = cella[incr + i + div*threadIdx.x];
  		//cellb_final[ nb_interactions + incr_threads[256*blockIdx.x + threadIdx.x] ] = cellb[incr + i + div*threadIdx.x];
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
    //ADD_SLOT( std::vector<int>, cells_non_empty, INPUT_OUTPUT); 
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
    	
    	
    	onika::memory::CudaMMVector<int> nb_particles_cells;
    	
    	
    	
    	onika::memory::CudaMMVector< onika::memory::CudaMMVector<int>> cell_particles_neighbors;
    	
    	onika::memory::CudaMMVector< onika::memory::CudaMMVector<int>> cell_particles_neighbors_size;
    	
    	//auto& cells_use = cells_non_empty;
    	
    	//cells_use.clear();
    	
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
      	//printf("?????????????????\n");
        LinearXForm xform = { domain->xform() };
        chunk_neighbors_execute2(ldbg,*chunk_neighbors,*grid,*amr,*amr_grid_pairs,*config,*chunk_neighbors_scratch,cs,cs_log2,*nbh_dist_lab, xform, gpu_enabled, no_z_order, cell_particles_nbh, id_cell_particles_nbh, id2_cell_particles_nbh, nb_particles_cells, cell_particles_neighbors, cell_particles_neighbors_size, nbh_filter );
      }
      else
      {
        NullXForm xform = { };
        //chunk_neighbors_execute2(ldbg,*chunk_neighbors,*grid,*amr,*amr_grid_pairs,*config,*chunk_neighbors_scratch,cs,cs_log2,*nbh_dist_lab, xform, gpu_enabled, no_z_order, cell_particles_nbh , id_cell_particles_nbh, id2_cell_particles_nbh, nbh_filter );
      }
      
      //HOOKE_FORCE_GPU
      
      //interactions_new.add_cells(cell_particles_nbh);
      
      for(int i= 0; i < cell_particles_nbh.size(); i++){
      	int cell= i;
      	if(cell_particles_nbh[i].size() > 0)
      	{ 
      		interactions_new.add_cell(cell, cell_particles_nbh[i].size(), cell_particles_neighbors[i], cell_particles_neighbors_size[i]);
      	}
      	/*for(int j= 0; j < cell_particles_nbh[i].size(); j++){
      		int particle= j;
      		auto ida = id_cell_particles_nbh[i][j];
      		std::vector< std::pair<int, int>>& nbh= cell_particles_nbh[i][j];
      		std::vector<int>& idb = id2_cell_particles_nbh[i][j];
      		if(nbh.size() > 0){
      			interactions_new.add_particle(particle, cell, nbh, ida, idb);
      		}
      	}*/
      }
      
      //getchar();
      
      
      int incr_cells = 500;
      
      int interactions_zzz = 0;
      
      onika::memory::CudaMMVector<int> cella_final;
      onika::memory::CudaMMVector<int> pa_final;
      onika::memory::CudaMMVector<int> cellb_final;
      onika::memory::CudaMMVector<int> pb_final;
      
      for(int i = 0; i < interactions_new.cells_GPU; i+= incr_cells)
      {
	
	int size = 0;
	int nb_cells = incr_cells;
	
	onika::memory::CudaMMVector<int> cells_ids;
	cells_ids.resize(nb_cells);
	
	onika::memory::CudaMMVector<int> cells_total_size;
	cells_total_size.resize(nb_cells);
	
	onika::memory::CudaMMVector<int> cells_total;
	cells_total.resize(nb_cells);
	
	onika::memory::CudaMMVector<int> cells_incr;
	cells_incr.resize(nb_cells);
	
	onika::memory::CudaMMVector<int> cells_nb_nbh;
	cells_nb_nbh.resize(nb_cells);
	
	onika::memory::CudaMMVector<int> cells_start;
	cells_start.resize(nb_cells);
	
	onika::memory::CudaMMVector<int> cells_256_div;
	cells_256_div.resize(nb_cells);
	
	onika::memory::CudaMMVector<int> cells_nbh_ids;
	
	onika::memory::CudaMMVector<int> cells_nbh_size;
	
	onika::memory::CudaMMVector<int> interactions_blocks;
	interactions_blocks.resize(nb_cells);
	
	onika::memory::CudaMMVector<int> interactions_threads;
	interactions_threads.resize( 256*nb_cells );
	
	int incr = 0;
	
	int z = 0;
	int start2 = 0;
      	for(int j = i; j < i + incr_cells; j++)
      	{
      		if(j < interactions_new.cells_GPU)
      		{
      		
      		cells_start[z] = start2;
      		cells_ids[z] = interactions_new.cells_GPU_ids[j];
      		int size_cell = interactions_new.cells_GPU_size[j];
      		int total_cell = interactions_new.cells_GPU_total[j];
      		int total = size_cell * total_cell;
      		cells_total_size[z] = total;
      		cells_total[z] = total_cell;
      		cells_nb_nbh[z] = interactions_new.cells_GPU_nb_nbh[j];
      		
      		int start = interactions_new.cells_GPU_start[j];
      		int nb = interactions_new.cells_GPU_nb_nbh[j];
      		
      		start2+= nb;
      		
      		int div = (int)(total/256);
      		
      		cells_256_div[z] = div + 1;
        		
      		
      		for(int t = start; t < start+nb; t++)
      		{
      			cells_nbh_ids.push_back(interactions_new.cells_GPU_nbh[t]);
      			cells_nbh_size.push_back(interactions_new.cells_GPU_nbh_size[t]);
      		}
      		
      		size+= total;
      		
      		z++;
      		
      		}
      	}
      	
      	printf("SIZE: %d\n", size);

      	onika::memory::CudaMMVector<int> pa;
      	onika::memory::CudaMMVector<int> cella;
      	onika::memory::CudaMMVector<int> pb;
      	onika::memory::CudaMMVector<int> cellb;
 
      	pa.resize(size);
      	cella.resize(size);
      	pb.resize(size);
      	cellb.resize(size);

      	int blockSize = 256;
	int numBlocks = nb_cells;
	
	onika::memory::CudaMMVector<int> nb_interactions;
	nb_interactions.resize(1);

      	kernel_GPU2<<<numBlocks, blockSize>>>( cells, size, cells_ids.data(), cells_total_size.data(), cells_total.data(), cells_incr.data(), cells_nb_nbh.data(), cells_start.data(), cells_256_div.data(), cells_nbh_ids.data(), cells_nbh_size.data(), *nbh_dist_lab, domain->xform(), *rcut_inc,  pa.data(), cella.data(), pb.data(), cellb.data(), nb_interactions.data(), interactions_blocks.data(), interactions_threads.data() );
      	
      	cudaDeviceSynchronize();
      	
      	/*onika::memory::CudaMMVector<int> incr_blocks;
      	incr_blocks.resize(nb_cells);
      	incr_blocks[0] = 0;
      	
      	for(int j = 1; j < nb_cells; j++)
      	{
      		incr_blocks[j] = incr_blocks[j - 1] + interactions_blocks[j - 1];
      	}*/
      	
      	onika::memory::CudaMMVector<int> incr_threads;
      	incr_threads.resize(256 * nb_cells);
      	incr_threads[0] = 0;
      	
      	for(int j = 1; j < 256 * nb_cells; j++)
      	{
      		incr_threads[j] = incr_threads[j - 1] + interactions_threads[j - 1];
      	}
      	
      	cella_final.resize(interactions_zzz + nb_interactions[0]);
      	pa_final.resize(interactions_zzz + nb_interactions[0]);
      	cellb_final.resize(interactions_zzz + nb_interactions[0]);
      	pb_final.resize(interactions_zzz + nb_interactions[0]);
      	
      	kernel_GPU3<<<numBlocks, blockSize>>>( cells_incr.data(), incr_threads.data(), interactions_threads.data(), cells_256_div.data(), pa.data(), cella.data(), pb.data(), cellb.data(), cella_final.data(), pa_final.data(), cellb_final.data(), pb_final.data(), interactions_zzz );
      
      	printf("APRÈS CUDA %d\n", i);
      	
      	printf("NB_INTERACTIONS : %d\n", nb_interactions[0]);

	//getchar();
	interactions_zzz+= nb_interactions[0];
      	
      }
      
      //printf("ICICICICICICICICICICICICICI\n");
    
      //interactions_new.quickSort();
            
      //interactions_new.init_friction(interactions_old);
      
      //interactions_new.init_GPU();
      
      printf("TOTAL INTERACTIONS : %d\n", interactions_zzz);
      
      getchar();

      printf("CHUNK FINISH\n");
      			 
    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory("chunk_neighbors_contact", make_grid_variant_operator< ChunkNeighborsContact > );
  }

}

