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

#include <exaDEM/interaction/interaction.hpp>

#include <exanb/particle_neighbors/chunk_neighbors_execute.h>

namespace exaDEM
{
  using namespace exanb;

  template <class CellsT> struct ContactNeighborFilterFunc
  {
    CellsT cells;
    const double rcut_inc = 0.0;
    inline bool operator()(double d2, double rcut2, size_t cell_a, size_t p_a, size_t cell_b, size_t p_b) const
    {
      assert(cell_a != cell_b || p_a != p_b);
      const double r_a = cells[cell_a][field::radius][p_a];
      const double r_b = cells[cell_b][field::radius][p_b];
      const double rmax = r_a + r_b + rcut_inc;
      const double rmax2 = rmax * rmax;
      return d2 > 0.0 && d2 < rmax2 && d2 < rcut2;
    }
  };
  
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
  
  template< class GridT > __global__ void kernelUN(GridT* cells,
  							int* cell_id,
  							int* nb_total,
  							int* nb_particles_nbh,
  							int* incr_cell,
  							int* incr_cell2,
  							int* cell_nb_nbh,
  							int* start_cell,
  							int* cell_div,
  							int* cells_nbh_ids,
  							int* cells_nbh_particles,
  							const double dist_lab,
  							Mat3d xform,
  							double rcut_inc,
  							exaDEM::Interaction* interactions,
  							int* nb_nbh2,
  							int* interactions_blocks )
  {
  	
  	int cell = cell_id[blockIdx.x];
  	int total_size = nb_total[blockIdx.x];
  	int total = nb_particles_nbh[blockIdx.x];
  	int incr = incr_cell[blockIdx.x];
  	int incr4 = incr_cell2[blockIdx.x];
  	int nb_nbh = cell_nb_nbh[blockIdx.x];
  	int start = start_cell[blockIdx.x];
  	int div = cell_div[blockIdx.x];
  	
  	//__shared__ int shared_threads[256];
  	__shared__ int shared_threads[256];

	shared_threads[threadIdx.x] = -1;
	__syncthreads();
	
	int nbh_pa;
  	
  	for(int i = 0; i < div; i++)
  	{
  		//int idx = 256*i + threadIdx.x
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
  				incr2+= cells_nbh_particles[j];
  				if(b2 && idx2 < incr2)
  				{
  					cell_b = cells_nbh_ids[j];
  					p_b = idx2 - incr3;
  					b2 = false;
  				}
  				
  				incr3+= cells_nbh_particles[j];
  			
  			}
  			
  			double rcut2 = dist_lab * dist_lab;
  		
  			const Vec3d dr = { cells[cell][field::rx][p_a] - cells[cell_b][field::rx][p_b] , cells[cell][field::ry][p_a] - cells[cell_b][field::ry][p_b] , cells[cell][field::rz][p_a] - cells[cell_b][field::rz][p_b] };
                	double d2 = norm2( xform * dr );
                	
  			if(nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell, p_a, cell_b, p_b)) 
  			{ 
  				shared_threads[threadIdx.x] = p_a;
  				
  				nbh_pa = nb_nbh2[incr + p_a];
  			}
  			
  			__syncthreads();
  			
  			int index = 0;
  			
  			if(shared_threads[threadIdx.x] != -1)
  			{
  				
  				for(int i = 0; i < threadIdx.x; i++)
  				{
  					if(shared_threads[i] == p_a) index++;
  				}
  				
  				exaDEM::Interaction item;
  				item.moment = {0,0,0};
  				item.friction = {0,0,0};
  				item.cell_i = cell;
  				item.type = 0;
  				item.id_i = cells[cell][field::id][p_a];
  				item.p_i = p_a;
  				item.id_j = cells[cell_b][field::id][p_b];
  				item.p_j = p_b;
  				item.cell_j = cell_b;
  				
  				interactions[incr4 + 15*p_a + nbh_pa + index] = item;
  				
  				atomicAdd(&nb_nbh2[incr + p_a], 1);
  				atomicAdd(&interactions_blocks[blockIdx.x], 1);
  				
  			}
  			
  			__syncthreads();
  			
  			shared_threads[threadIdx.x] = -1;
  			
  			__syncthreads();
  			
  		}
  		
  		
  	}  	
  	 
  }

  template <typename GridT> struct ChunkNeighborsContact : public OperatorNode
  {
#   ifdef XSTAMP_CUDA_VERSION
      ADD_SLOT(onika::cuda::CudaContext, cuda_ctx, INPUT, OPTIONAL);
#   endif

    ADD_SLOT(GridT, grid, INPUT);
    ADD_SLOT(AmrGrid, amr, INPUT);
    ADD_SLOT(AmrSubCellPairCache, amr_grid_pairs, INPUT);
    ADD_SLOT(Domain, domain, INPUT);
    ADD_SLOT(double, rcut_inc, INPUT);     // value added to the search distance to update neighbor list less frequently
    ADD_SLOT(double, nbh_dist_lab, INPUT); // value added to the search distance to update neighbor list less frequently
    ADD_SLOT(GridChunkNeighbors, chunk_neighbors, INPUT_OUTPUT);

    ADD_SLOT(ChunkNeighborsConfig, config, INPUT, ChunkNeighborsConfig{});
    ADD_SLOT(ChunkNeighborsScratchStorage, chunk_neighbors_scratch, PRIVATE);

    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator builds the neighbor lists used to detect contacts between particles.
        )EOF";
    }

    inline void execute() override final
    {
      unsigned int cs = config->chunk_size;
      unsigned int cs_log2 = 0;
      while (cs > 1)
      {
        assert((cs & 1) == 0);
        cs = cs >> 1;
        ++cs_log2;
      }
      cs = 1 << cs_log2;
      // ldbg << "cs="<<cs<<", log2(cs)="<<cs_log2<<std::endl;
      if (cs != static_cast<size_t>(config->chunk_size))
      {
        lerr << "chunk_size is not a power of two" << std::endl;
        std::abort();
      }

      // const bool gpu_enabled = parallel_execution_context()->has_gpu_context();

      bool gpu_enabled = (global_cuda_ctx() != nullptr);
      if (gpu_enabled)
        gpu_enabled = global_cuda_ctx()->has_devices();

      auto cells = grid->cells();
      ContactNeighborFilterFunc<decltype(cells)> nbh_filter{cells, *rcut_inc};
      static constexpr std::false_type no_z_order = {};

      if (!domain->xform_is_identity())
      {
        LinearXForm xform = {domain->xform()};
        chunk_neighbors_execute(ldbg, *chunk_neighbors, *grid, *amr, *amr_grid_pairs, *config, *chunk_neighbors_scratch, cs, cs_log2, *nbh_dist_lab, xform, gpu_enabled, no_z_order, nbh_filter);
      }
      else
      {
        NullXForm xform = {};
        chunk_neighbors_execute(ldbg, *chunk_neighbors, *grid, *amr, *amr_grid_pairs, *config, *chunk_neighbors_scratch, cs, cs_log2, *nbh_dist_lab, xform, gpu_enabled, no_z_order, nbh_filter);
      }
       
      onika::memory::CudaMMVector<int> nb_particles_cell;
      onika::memory::CudaMMVector<onika::memory::CudaMMVector<int>> cell_particles_neighbors;
      onika::memory::CudaMMVector<onika::memory::CudaMMVector<int>> cell_particles_neighbors_size;
      
      	auto& g = *grid;
      	//auto cells = g.cells();
      	IJK dims = g.dimension();
      	
      	auto& amr_gpu = *amr;
      	const size_t* sub_grid_start = amr_gpu.sub_grid_start().data();
      	const uint32_t* sub_grid_cells = amr_gpu.sub_grid_cells().data();
      	
      	const double max_dist = *nbh_dist_lab;
      	const double max_dist2 = max_dist*max_dist;
      	
      	auto& amr_grid_pairs_gpu = *amr_grid_pairs;
      	const unsigned int loc_max_gap = amr_grid_pairs_gpu.cell_layers();
      	//const unsigned int nbh_cell_side = loc_max_gap+1;
      	const unsigned int n_nbh_cell = amr_grid_pairs_gpu.nb_nbh_cells();
      	//assert( nbh_cell_side*nbh_cell_side*nbh_cell_side == n_nbh_cell );
      	
      	const size_t n_cells = g.number_of_cells();
      	
      	nb_particles_cell.resize( n_cells );
      	cell_particles_neighbors.resize( n_cells );
      	cell_particles_neighbors_size.resize( n_cells );
      	
      	//chunk_neighbors_gpu.clear();
      	//chunk_neighbors_gpu.set_number_of_cells( n_cells );
      	//chunk_neighbors_gpu.set_chunk_size( cs );
      	//chunk_neighbors_gpu.realloc_stream_pool( config_gpu.stream_prealloc_factor );
      	
      	
      	//ldbg << "cell max gap = "<<loc_max_gap<<", cslog2="<<cs_log2<<", n_nbh_cell="<<n_nbh_cell<<", pre-alloc="<<chunk_neighbors_gpu.m_fixed_stream_pool.m_capacity <<std::endl;
      	
      	unsigned int max_threads = omp_get_max_threads();
      	//if( max_threads > chunk_neighbors_scratch_gpu.thread.size() )
      	//{
      	//	chunk_neighbors_scratch_gpu.thread.resize( max_threads );
      	//}
      	
      	//GridChunkNeighborsHostWriteAccessor chunk_nbh( chunk_neighbors_gpu );
      	
#	pragma omp parallel
	{
		int tid = omp_get_thread_num();
		assert( tid>=0 && size_t(tid)<max_threads );
		//auto& cell_a_particle_nbh = chunk_neighbors_scratch_gpu.thread[tid].cell_a_particle_nbh;
		
		GRID_OMP_FOR_BEGIN(dims,cell_a,loc_a, schedule(dynamic))
		{
			size_t n_particles_a = cells[cell_a].size();
			
			nb_particles_cell[cell_a] = n_particles_a;
			
			//cell_a_particle_nbh.resize( n_particles_a );
			
			//for(size_t i=0;i<n_particles_a;i++)
			//{
			//	cell_a_particle_nbh[i].clear();
			//}
			
			ssize_t sgstart_a = sub_grid_start[cell_a];
			ssize_t sgsize_a = sub_grid_start[cell_a+1] - sgstart_a;
			ssize_t n_sub_cells_a = sgsize_a+1;
			ssize_t sgside_a = icbrt64( n_sub_cells_a );
			assert( sgside_a <= static_cast<ssize_t>(GRID_CHUNK_NBH_MAX_AMR_RES) );
			
          		ssize_t bstarti = std::max( loc_a.i-loc_max_gap , 0l ); 
          		ssize_t bendi = std::min( loc_a.i+loc_max_gap , dims.i-1 );
          		ssize_t bstartj = std::max( loc_a.j-loc_max_gap , 0l );
          		ssize_t bendj = std::min( loc_a.j+loc_max_gap , dims.j-1 );
          		ssize_t bstartk = std::max( loc_a.k-loc_max_gap , 0l );
         		ssize_t bendk = std::min( loc_a.k+loc_max_gap , dims.k-1 );
         		
          		for(ssize_t loc_bk=bstartk;loc_bk<=bendk;loc_bk++)
          		for(ssize_t loc_bj=bstartj;loc_bj<=bendj;loc_bj++)
          		for(ssize_t loc_bi=bstarti;loc_bi<=bendi;loc_bi++)
          		{
            			IJK loc_b { loc_bi, loc_bj, loc_bk };
            			ssize_t cell_b = grid_ijk_to_index( dims, loc_b );
            			size_t n_particles_b = cells[cell_b].size();
            			
            			if( n_particles_b > 0)
            			{
            				cell_particles_neighbors[cell_a].push_back(cell_b);
            				cell_particles_neighbors_size[cell_a].push_back(n_particles_b);
            			}
            		}         		
			
		}
		GRID_OMP_FOR_END
	}

	//PARTIE CONSTRUCTION LISTES GPU
	
	//IDENTIFIANT DES CELLULES NON VIDES
	onika::memory::CudaMMVector<int> cell_id;
	for(int i=0; i < nb_particles_cell.size(); i++)
	{
		if( nb_particles_cell[i] > 0) cell_id.push_back(i);
	}
	
	
	onika::memory::CudaMMVector<int> nb_particles; //NOMBRE DE PARTICULES
	nb_particles.resize(cell_id.size());
	onika::memory::CudaMMVector<int> cell_nb_nbh; //NOMBRE DE CELLULES VOISINES
	cell_nb_nbh.resize(cell_id.size());
	onika::memory::CudaMMVector<int> nb_particles_nbh; //NOMBRE TOTAL DE PARTICULES VOISINES
	nb_particles_nbh.resize(cell_id.size());
	onika::memory::CudaMMVector<int> nb_total; //NOMBRE DE PARTICULES * NOMBRE DE PARTICULES VOISINES
	nb_total.resize(cell_id.size());
	
	for(int i=0; i < cell_id.size(); i++)
	{
		int index = cell_id[i];
		nb_particles[i] = nb_particles_cell[index];
		cell_nb_nbh[i] = cell_particles_neighbors[index].size();
		
		int sum = 0;
		for(auto nb : cell_particles_neighbors_size[index] )
		{
			sum+= nb;
		}
		nb_particles_nbh[i] = sum;
		nb_total[i] = nb_particles[i] * nb_particles_nbh[i];
	}
	
	for(int i = 0; i < cell_id.size(); i++)
	{
		printf("CELLULE_%i : %d SIZE: %d NB_NBH: %d\n", i, cell_id[i], nb_particles_cell[cell_id[i]], cell_nb_nbh[i]);
	}
	
	
	onika::memory::CudaMMVector<int> start_cell; //START CELL
	start_cell.resize(cell_id.size());
	onika::memory::CudaMMVector<int> incr_cell; //INCR
	incr_cell.resize(cell_id.size());
	onika::memory::CudaMMVector<int> incr_cell2;
	incr_cell2.resize(cell_id.size());
	
	int total = 0;
	int total2 = 0;
	int total3 = 0;
	for(int i = 0; i < cell_id.size(); i++)
	{
		start_cell[i] = total;
		incr_cell[i] = total2;
		incr_cell2[i] = total3;
		
		total+= cell_nb_nbh[i];
		total2+= nb_particles[i];
		total3+= nb_particles[i] * 15; 
	}
	
	onika::memory::CudaMMVector<int> cell_nbh_ids; //IDENTIFIANTS VOISINS
	cell_nbh_ids.resize(total + cell_nb_nbh[cell_nb_nbh.size() - 1]);
	onika::memory::CudaMMVector<int> cell_nbh_particles; //PARTICULES VOISINES
	cell_nbh_particles.resize( cell_nbh_ids.size() );
	onika::memory::CudaMMVector<int> cell_div; //DIV
	cell_div.resize(cell_id.size());
	int nb_threads = 256;
	for(int i = 0; i < cell_id.size(); i++)
	{
		int index = cell_id[i];
		int start = start_cell[i];
		int div = (int)(nb_total[i]/nb_threads) + 1;
		for(int j = 0; j < cell_nb_nbh[i]; j++)
		{
			cell_nbh_ids[start+j] = cell_particles_neighbors[index][j];
			cell_nbh_particles[start+j] = cell_particles_neighbors_size[index][j];
		}
		
		cell_div[i] = div;
	}

	
	int numBlocks = cell_id.size();
	
	// TABLEAUX DE RESULTATS
	//onika::memory::CudaMMVector<exaDEM::Interaction> interactions;
	//onika::memory::CudaMMVector<int> nb_nbh;
	//onika::memory::CudaMMVector<int> interactions_blocks;
	onika::memory::CudaMMVector<exaDEM::Interaction> interactions;
	onika::memory::CudaMMVector<int> nb_nbh;
	onika::memory::CudaMMVector<int> interactions_blocks;
	
	//PREPARATION DES TABLEAUX
	nb_nbh.resize(total2);
	interactions.resize(total3);
	interactions_blocks.resize(cell_id.size());
	
	kernelUN<<<numBlocks, nb_threads>>>(cells, cell_id.data(), nb_total.data(), nb_particles_nbh.data(), incr_cell.data(), incr_cell2.data(), cell_nb_nbh.data(), start_cell.data(), cell_div.data(), cell_nbh_ids.data(), cell_nbh_particles.data(), *nbh_dist_lab, domain->xform(), *rcut_inc, interactions.data(), nb_nbh.data(), interactions_blocks.data());
	
	cudaDeviceSynchronize();
	
	int incrr = incr_cell[15];
	
	for(int i = 0; i < nb_particles[15]; i++)
	{
		printf("PARTICULE: %d VOISINS: %d\n", i, nb_nbh[incrr + i]);
	}
	
	getchar();	      
      
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("chunk_neighbors_contact", make_grid_variant_operator<ChunkNeighborsContact>); }

} // namespace exaDEM
