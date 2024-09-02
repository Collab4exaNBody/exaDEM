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

  
  template< class GridT > __global__ void kernel_GPU2(GridT* cells,
  							int size,
  							int* cells_ids,
  							int* cells_total_size,
  							int* cells_total,
  							int* cells_incr,
  							int* cells_nb_nbh,
  							int* cells_start,
  							int* cells_512_div,
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
  	int div = cells_512_div[blockIdx.x];
  	int interactions = 0;
  	int total_interactions = 0;
  	int total_interactions2 = 0;
  	
  	//__shared__ int shared_threads[512];
  	__shared__ int shared_threads[512];
	
	shared_threads[threadIdx.x] = 0;
	__syncthreads();
  	
  	for(int i = 0; i < div; i++)
  	{
  		//int idx = 512*i + threadIdx.x
  		int idx = 512*i + threadIdx.x;
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
                	
  			if(nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell, p_a, cell_b, p_b)) 
  			{ 
  		
  				atomicAdd(&nb_interactions[0], 1);
  				atomicAdd(&interactions_blocks[blockIdx.x], 1);
  				shared_threads[threadIdx.x] = 1;
  			}
  			
  			__syncthreads();
  			
  			int index = 0;
  			
  			if(shared_threads[threadIdx.x] == 1)
  			{
  				
  				for(int i = 0; i < threadIdx.x; i++)
  				{
  					if(shared_threads[i] == 1) index++;
  				}
  				
  				pa[ incr + total_interactions + index ] = p_a;
  				cella[ incr + total_interactions + index ] = cell;
  				pb[ incr + total_interactions + index ] = p_b;
  				cellb[ incr + total_interactions + index ] = cell_b;
  				
  			}
  			
  			__syncthreads();
  			
  			total_interactions= interactions_blocks[blockIdx.x];
  			
  			shared_threads[threadIdx.x] = 0;
  			
  			__syncthreads();
  			
  		}
  		
  		
  	}  	
  	 
  }
  
    template< class GridT > __global__ void kernel_GPU6(GridT* cells,
  							int size,
  							int* cells_ids,
  							int* cells_total_size,
  							int* cells_total,
  							int* cells_incr,
  							int* cells_nb_nbh,
  							int* cells_start,
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
  							int nb_blocks,
  							int nb_cells)
  {
  	
  	__shared__ int shared_index_cell[1];
  	__shared__ int shared_index_cell2[1];
  	__shared__ int shared_threads[512];
  	
  	int this_block = blockIdx.x * 512;
  	
	int div_cells_blocks = (nb_cells / 512) + 1;
	
	int start_thread = threadIdx.x * div_cells_blocks;
	
	int end_thread = (threadIdx.x + 1) * div_cells_blocks;
	
	int last_total = 0;
	
	bool find = false;
	
	if(start_thread > 0)
	{
		last_total = cells_incr[start_thread - 1];
	}
	
	
	for(int i = start_thread; i < end_thread; i++)
	{
		if(find == false && i < nb_cells)
		{
			int total = cells_incr[i];
		
			if( this_block >= last_total && this_block < total)
			{
				find = true;
				shared_index_cell[0] = i; 
			}
			
			last_total = total;
		}	
	}
	
	__syncthreads();
	
	int index_cell = shared_index_cell[0];
	
	int this_block2;
	
	if(index_cell > 0)
	{
		this_block2 = this_block - cells_incr[index_cell - 1];
	}
	else
	{
		this_block2 = this_block;
	}
	
	int last_total2 = 0;
	
	if(threadIdx.x > 0)
	{
		last_total2 = (512) * threadIdx.x; 
	}
	
	if( this_block2 >= last_total2 && this_block2 < last_total2 + 512 )
	{
		shared_index_cell2[0] = threadIdx.x;
	}
	
  	shared_threads[threadIdx.x] = 0;
  	
  	__syncthreads();
  	
  	int index_cell2 = shared_index_cell2[0];
  	
  	if(blockIdx.x < 30)
  	if(threadIdx.x == 0) printf("BLOCKIDX : %d, THIS_BLOCK: %d, THIS_BLOCK2: %d, INDEX1; %d, INDEX2: %d\n", blockIdx.x, this_block, this_block2, index_cell, index_cell2);
  
  	int cell = cells_ids[index_cell];
  	 int total_size = cells_total_size[index_cell];
  	int total = cells_total[index_cell];
  	int nb_nbh = cells_nb_nbh[index_cell];
  	int start = cells_start[index_cell];
  	//for(int i = 0; i < div; i++)
  	//{
  		//int idx = 512*i + threadIdx.x
  		int idx = 512*/*i*/index_cell2 + threadIdx.x;
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
                	
  			if(nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell, p_a, cell_b, p_b)) 
  			{ 
  		
  				atomicAdd(&nb_interactions[0], 1);
  				atomicAdd(&interactions_blocks[nb_blocks], 1);
  				shared_threads[threadIdx.x] = 1;
  			}
  			
  			__syncthreads();
  			
  			int index = 0;
  			
  			if(shared_threads[threadIdx.x] == 1)
  			{
  				
  				for(int i = 0; i < threadIdx.x; i++)
  				{
  					if(shared_threads[i] == 1) index++;
  				}
  				
  				pa[ /*incr*/blockIdx.x*20 + index ] = p_a;
  				cella[ /*incr*/blockIdx.x*20 + index ] = cell;
  				pb[ /*incr*/blockIdx.x*20 + index ] = p_b;
  				cellb[ /*incr*/blockIdx.x*20 + index ] = cell_b;
  				
  			}
  			
  			//__syncthreads();
  			
  			//total_interactions= interactions_blocks[blockIdx.x];
  			
  			//shared_threads[threadIdx.x] = 0;
  			
  			//__syncthreads();
  			
  		}
  		
  		
  	//}  	
  	 
  }
  
  __global__ void kernel_GPU4(int* cells_incr,int* pa, int* cella, int* pb, int* cellb, int* interactions_blocks, int* pa_final, int* cella_final, int* pb_final, int* cellb_final)
  {
  	int incr = 0;
  	
  	for(int i = 0; i < blockIdx.x ; i++)
  	{
  		incr+= interactions_blocks[i];
  	}
  	
  	int incr2 = cells_incr[blockIdx.x];
  	
  	int nb_int = interactions_blocks[blockIdx.x];
  	
  	//int div = (int)(nb_int/512) + 1;
  	int div = (int)(nb_int/512) + 1;
  	
  	for(int i = 0; i < div; i++)
  	{
  		//int idx = 512*i + threadIdx.x;
  		int idx = 512*i + threadIdx.x;
  		
  		if(idx < nb_int)
  		{
  			pa_final[incr + idx] = pa[incr2 + idx];
  			cella_final[incr + idx] = cella[incr2 + idx];
  			pb_final[incr + idx] = pb[incr2 + idx];
  			cellb_final[incr + idx] = cellb[incr2 + idx];
  		} 
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
      	for(int j= 0; j < cell_particles_nbh[i].size(); j++){
      		int particle= j;
      		auto ida = id_cell_particles_nbh[i][j];
      		std::vector< std::pair<int, int>>& nbh= cell_particles_nbh[i][j];
      		std::vector<int>& idb = id2_cell_particles_nbh[i][j];
      		if(nbh.size() > 0){
      			interactions_new.add_particle(particle, cell, nbh, ida, idb);
      		}
      	}
      }
      
      //getchar();
      
      
      int incr_cells = interactions_new.cells_GPU;//*/10000;
      
      int interactions_zzz = 0;
      
      //interactions_new.cells_GPU = incr_cells;
      
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
	
	onika::memory::CudaMMVector<int> cells_512_div;
	cells_512_div.resize(nb_cells);
	
	onika::memory::CudaMMVector<int> cells_nbh_ids;
	
	onika::memory::CudaMMVector<int> cells_nbh_size;
	
	onika::memory::CudaMMVector<int> interactions_blocks;
	//interactions_blocks.resize(nb_cells);
	
	onika::memory::CudaMMVector<int> interactions_threads;
	//interactions_threads.resize( 512*nb_cells );
	interactions_threads.resize( 512*nb_cells);
	
	int nb_blocks = 0;
	
	int incr = 0;
	
	int div2 = 0;
	
	int z = 0;
	int start2 = 0;
	int total1 = 0;
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
      		//cells_incr[z] = size;
      		cells_incr[z] = total1 + total;
      		
      		int start = interactions_new.cells_GPU_start[j];
      		int nb = interactions_new.cells_GPU_nb_nbh[j];
      		
      		total1+= total;
      		
      		start2+= nb;
      		
      		
      		//int div = (int)(total/512);
      		int div = (int)(total/512);
      		
      		div2+= div;
      		
      		//cells_512_div[z] = div + 1;
        		
      		
      		for(int t = start; t < start+nb; t++)
      		{
      			cells_nbh_ids.push_back(interactions_new.cells_GPU_nbh[t]);
      			cells_nbh_size.push_back(interactions_new.cells_GPU_nbh_size[t]);
      		}
      		
      		/*if(total < 100)
      		{
      			//printf("INDEX CELL: %d\n", j);
      			size+= (int)(total);
      		}
      		else if(j > 18000)
      		{
      			//printf("TOTAL 12M: %d\n", total);
      			size+= (int)(total);
      		}
      		else
      		{
      			size+= (int)(total/10);
      		}*/
      		
      		size+= 20*(div + 1);	
      		
      		z++;
      		
      		nb_blocks+= div+1;
      		
      		}
      	}
      	
      	interactions_blocks.resize(nb_blocks);
      	
      	printf("SIZE: %d\n", size);
      	
      	printf("DIV2: %d\n", div2);
      
      	onika::memory::CudaMMVector<int> pa;
      	onika::memory::CudaMMVector<int> cella;
      	onika::memory::CudaMMVector<int> pb;
      	onika::memory::CudaMMVector<int> cellb;
 
      	pa.resize(size);
      	cella.resize(size);
      	pb.resize(size);
      	cellb.resize(size);
      	
     	
      	//int blockSize = 512;
      	int blockSize = 512;
      	int numBlocks = nb_cells;
      	
	onika::memory::CudaMMVector<int> nb_interactions;
	nb_interactions.resize(1);

      	//kernel_GPU2<<<numBlocks, blockSize>>>( cells, size, cells_ids.data(), cells_total_size.data(), cells_total.data(), cells_incr.data(), cells_nb_nbh.data(), cells_start.data(), cells_512_div.data(), cells_nbh_ids.data(), cells_nbh_size.data(), *nbh_dist_lab, domain->xform(), *rcut_inc,  pa.data(), cella.data(), pb.data(), cellb.data(), nb_interactions.data(), interactions_blocks.data(), interactions_threads.data() );
      	
      	kernel_GPU6<<<nb_blocks , blockSize>>>( cells, size, cells_ids.data(), cells_total_size.data(), cells_total.data(),  cells_incr.data(), cells_nb_nbh.data(), cells_start.data(), cells_nbh_ids.data(), cells_nbh_size.data(), *nbh_dist_lab, domain->xform(), *rcut_inc,  pa.data(), cella.data(), pb.data(), cellb.data(), nb_interactions.data(), interactions_blocks.data(), nb_blocks, nb_cells );
      	
      	cudaDeviceSynchronize();
      	
      	for(int i = 0;  i < 1000; i++)
      {
      	printf("INTERACTIONS%d A(%d, %d) B(%d, %d)\n", i, cella[i], pa[i], cellb[i], pb[i]);
      }
      
      printf("INTERACTIONS: %d\n", nb_interactions[0]);
      
      getchar();

      	cella_final.resize(interactions_zzz + nb_interactions[0]);
      	pa_final.resize(interactions_zzz + nb_interactions[0]);
      	cellb_final.resize(interactions_zzz + nb_interactions[0]);
      	pb_final.resize(interactions_zzz + nb_interactions[0]);
      	
      	kernel_GPU4<<<numBlocks, blockSize>>>( cells_incr.data(), pa.data(), cella.data(), pb.data(), cellb.data(), interactions_blocks.data(), pa_final.data(), cella_final.data(), pb_final.data(), cellb_final.data() );

      	cudaDeviceSynchronize();

	interactions_zzz+= nb_interactions[0];
      	
      }
      
      for(int i = 1000000;  i < 1001000; i++)
      {
      	printf("INTERACTIONS%d A(%d, %d) B(%d, %d)\n", i, cella_final[i], pa_final[i], cellb_final[i], pb_final[i]);
      }

      //interactions_new.quickSort();
            
      //interactions_new.init_friction(interactions_old);
      
      interactions_new.init_GPU();
      
      if(interactions_new.nb_interactions == cella_final.size())
      {
      	printf("ON AVANCE\n");
      	
      	for(int i = 0; i < /*interactions_new.nb_interactions*/cella_final.size(); i++)
      	{
      		if(interactions_new.cella_GPU[i] != cella_final[i] ||  interactions_new.pa_GPU[i] != pa_final[i] || interactions_new.cellb_GPU[i] != cellb_final[i] || interactions_new.pb_GPU[i] != pb_final[i])
      		{
      			printf("ERREUR: %d\n", i);
      		}
      	}
      	
      }
      
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

