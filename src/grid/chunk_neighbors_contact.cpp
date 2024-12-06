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

#include <exaDEM/interaction/interactionSOA.hpp>

#include <exanb/particle_neighbors/chunk_neighbors_execute.h>

#include <cub/cub.cuh>

#define THREADS_PER_BLOCK 256

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
  							int* nb_particles,
  							int* cell_neighbors_ids,
  							int* cell_neighbors_size,
  							int* cell_start,
  							int* cell_end,
  							int* cell_particle_start,
  							const double dist_lab,
  							Mat3d xform,
  							double rcut_inc,
  							uint64_t* id_i,
  							uint64_t* id_j,
  							uint32_t* cell_i,
  							uint32_t* cell_j,
  							uint16_t* p_i,
  							uint16_t* p_j,
  							int* nb_nbh,
  							int* id,
  							int* total_interactions )
  {
  	
  	int cell = cell_id[blockIdx.x];
	int nb = nb_particles[blockIdx.x];
	int start = cell_start[blockIdx.x];
	int end = cell_end[blockIdx.x];
	int incr = cell_particle_start[blockIdx.x];
	
	if(threadIdx.x < nb)
	{
		int p_a = threadIdx.x;
		int nb_interactions = 0;
		
		for(int i = start; i < end; i++)
		{
			int cell_b = cell_neighbors_ids[i];
			
			for(int j = 0; j < cell_neighbors_size[i]; j++)
			{
				int p_b = j;
	  			
	  			double rcut2 = dist_lab * dist_lab;
  		
  				const Vec3d dr = { cells[cell][field::rx][p_a] - cells[cell_b][field::rx][p_b] , cells[cell][field::ry][p_a] - cells[cell_b][field::ry][p_b] , cells[cell][field::rz][p_a] - cells[cell_b][field::rz][p_b] };
                		double d2 = norm2( xform * dr );
                	
  				if(nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell, p_a, cell_b, p_b)) 
  				{ 
  					id_i[incr*6 + p_a*6 + nb_interactions] = cells[cell][field::id][p_a];
  					id_j[incr*6 + p_a*6 + nb_interactions] = cells[cell_b][field::id][p_b];
  					cell_i[incr*6 + p_a*6 + nb_interactions] = cell;
  					cell_j[incr*6 + p_a*6 + nb_interactions] = cell_b;
  					p_i[incr*6 + p_a*6 + nb_interactions] = p_a;
  					p_j[incr*6 + p_a*6 + nb_interactions] = p_b;
  					
  					nb_interactions++;
  					
  					atomicAdd(&total_interactions[0], 1);
  				}				
			}
		}
		
		nb_nbh[incr + p_a] = nb_interactions;
		id[incr + p_a] = cells[cell][field::id][p_a];
	}
   }
   
  __global__ void kernelDEUX(int* nb_nbh, int* nb_nbh_incr, uint64_t* id_i, uint64_t* id_j, uint32_t* cell_i, uint32_t* cell_j, uint16_t* p_i, uint16_t* p_j, uint64_t* id_i_final, uint64_t* id_j_final, uint32_t* cell_i_final, uint32_t* cell_j_final, uint16_t* p_i_final, uint16_t* p_j_final, int n )
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < n)
  	{
  		int nb = nb_nbh[idx];
  		int incr = nb_nbh_incr[idx];
  		
  		for(int i = 0; i < nb; i++)
  		{
  			id_i_final[incr + i] = id_i[idx*6 + i];
  			id_j_final[incr + i] = id_j[idx*6 + i];
  			cell_i_final[incr + i] = cell_i[idx*6 + i];
  			cell_j_final[incr + i] = cell_j[idx*6 + i];
  			p_i_final[incr + i] = p_i[idx*6 + i];
  			p_j_final[incr + i] = p_j[idx*6 + i];
  		}
  	}
  }
  
  __global__ void setParticles1(int* ids, int* particles_start, int* particles_end, int n)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < n && idx > 0)
  	{
  		if(ids[idx] == ids[idx - 1])
  		{
  			ids[idx] = 0;
  		}
  	}
  }
  
  __global__ void setParticles2(int* particles_start_final, int* particles_end_final, int* ids, int* particles_start, int* particles_end, int n)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < n)
  	{
  		int id = ids[idx];
  		if(id > 0)
  		{
  			particles_start_final[id - 1] = particles_start[idx];
  			particles_end_final[id -1] = particles_end[idx];
  		}
  	}
  }
  
  __global__ void update_friction_moment(uint64_t* id_i, uint64_t* id_j, int* start_old, int* end_old, uint64_t* id_j_old, double* ftx, double* fty, double* ftz, double* ftx_old, double* fty_old, double* ftz_old, double* momx, double* momy, double* momz, double* momx_old, double* momy_old, double* momz_old, int n)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < n)
  	{
  		int id = id_i[idx];
  		int id2 = id_j[idx];
  		int start = start_old[id];
  		int end = end_old[id];
  		
  		for(int i = start; i < end; ++i)
  		{
  			if(id2 == id_j_old[i])
  			{
  				ftx[idx] = ftx_old[i];
  				fty[idx] = fty_old[i];
  				ftz[idx] = ftz_old[i];
  				momx[idx] = momx_old[i];
  				momy[idx] = momy_old[i];
  				momz[idx] = momz_old[i];
  			}
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
    ADD_SLOT(InteractionSOA, interaction_neighbors, INPUT_OUTPUT);

    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator builds the neighbor lists used to detect contacts between particles.
        )EOF";
    }

    inline void execute() override final
    {
      printf("CHUNK_BEGIN\n");
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
      
      auto& int_neighbors = *interaction_neighbors;
      
      auto number_of_particles = grid->number_of_particles();
       
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
	onika::memory::CudaMMVector<int> cell_neighbors_ids;
	onika::memory::CudaMMVector<int> cell_neighbors_size;
	
	for(int i=0; i < cell_id.size(); i++)
	{
		int index = cell_id[i];
		nb_particles[i] = nb_particles_cell[index];
		cell_nb_nbh[i] = cell_particles_neighbors[index].size();
		for(auto id: cell_particles_neighbors[index])
		{
			cell_neighbors_ids.push_back(id);	
		}
		for(auto size: cell_particles_neighbors_size[index])
		{
			cell_neighbors_size.push_back(size);
		}
	
	}
	
	//CELL START
	onika::memory::CudaMMVector<int> cell_start;
	cell_start.resize(cell_nb_nbh.size());

	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, cell_nb_nbh.data(), cell_start.data(), cell_start.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, cell_nb_nbh.data(), cell_start.data(), cell_start.size());
	
	cudaFree(d_temp_storage);
	
	//CELL END
	onika::memory::CudaMMVector<int> cell_end;
	cell_end.resize(cell_nb_nbh.size());
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, cell_nb_nbh.data(), cell_end.data(), cell_end.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, cell_nb_nbh.data(), cell_end.data(), cell_end.size());
		
	cudaFree(d_temp_storage);
	
	//CELL PARTICLES START
	onika::memory::CudaMMVector<int> nb_particles_start;
	nb_particles_start.resize(nb_particles.size());
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_particles.data(), nb_particles_start.data(), nb_particles.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_particles.data(), nb_particles_start.data(), nb_particles.size());
	
	cudaFree(d_temp_storage);	
	
	int numBlocks = cell_id.size();

	onika::memory::CudaMMVector<uint64_t> id_i;
	onika::memory::CudaMMVector<uint64_t> id_j;
	onika::memory::CudaMMVector<uint32_t> cell_i;
	onika::memory::CudaMMVector<uint32_t> cell_j;
	onika::memory::CudaMMVector<uint16_t> p_i;
	onika::memory::CudaMMVector<uint16_t> p_j;
	
	onika::memory::CudaMMVector<int> nb_nbh;
	onika::memory::CudaMMVector<int> id_particle;
	
	onika::memory::CudaMMVector<int> total_interactions;
	
	total_interactions.resize(1);

	int total = 0;
	for(int i = 0; i < nb_particles.size(); i++)
	{
		total+= nb_particles[i];
	}
	
	nb_nbh.resize(total);
	id_particle.resize(total);
	
	total = total*6;
	
	id_i.resize(total);
	id_j.resize(total);
	cell_i.resize(total);
	cell_j.resize(total);
	p_i.resize(total);
	p_j.resize(total);
	
	kernelUN<<<numBlocks, 32>>>(cells, cell_id.data(), nb_particles.data(), cell_neighbors_ids.data(), cell_neighbors_size.data(), cell_start.data(), cell_end.data(), nb_particles_start.data(), *nbh_dist_lab, domain->xform(), *rcut_inc, id_i.data(), id_j.data(), cell_i.data(), cell_j.data(), p_i.data(), p_j.data(), nb_nbh.data(), id_particle.data(), total_interactions.data());
	
	cudaDeviceSynchronize();
	
	/*for(int i = 0; i < 100; i++)
	{
		printf("CELL: %d ID_I: %d ID_J: %d\n", cell_i[i], id_i[i], id_j[i]);
	}*/
	
	//printf("\n\n\n\n\n\n");
	
	onika::memory::CudaMMVector<int> nb_nbh_incr;
	nb_nbh_incr.resize(nb_nbh.size());
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), nb_nbh.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), nb_nbh.size());
	
	cudaFree(d_temp_storage);
	
	onika::memory::CudaMMVector<uint64_t> id_i_final;
	onika::memory::CudaMMVector<uint64_t> id_j_final;
	onika::memory::CudaMMVector<uint32_t> cell_i_final;
	onika::memory::CudaMMVector<uint32_t> cell_j_final;
	onika::memory::CudaMMVector<uint16_t> p_i_final;
	onika::memory::CudaMMVector<uint16_t> p_j_final;
	
	id_i_final.resize(total_interactions[0]);
	id_j_final.resize(total_interactions[0]);
	cell_i_final.resize(total_interactions[0]);
	cell_j_final.resize(total_interactions[0]);
	p_i_final.resize(total_interactions[0]);
	p_j_final.resize(total_interactions[0]);
	
	numBlocks = (nb_nbh.size() + 256 - 1) / 256;
	
	kernelDEUX<<<numBlocks, 256>>>(nb_nbh.data(), nb_nbh_incr.data(), id_i.data(), id_j.data(), cell_i.data(), cell_j.data(), p_i.data(), p_j.data(), id_i_final.data(), id_j_final.data(), cell_i_final.data(), cell_j_final.data(), p_i_final.data(), p_j_final.data(), nb_nbh.size());
	
	cudaDeviceSynchronize();
	
	onika::memory::CudaMMVector<int> nb_nbh_end;
	nb_nbh_end.resize(nb_nbh.size());
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_end.data(), nb_nbh.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_end.data(), nb_nbh.size());
	
	cudaFree(d_temp_storage);
	
	onika::memory::CudaMMVector<int> id_particle_sorted;
	id_particle_sorted.resize(id_particle.size());
	
	onika::memory::CudaMMVector<int> particles_start_sorted;
	onika::memory::CudaMMVector<int> particles_end_sorted;
	
	particles_start_sorted.resize(id_particle.size());
	particles_end_sorted.resize(id_particle.size());
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, id_particle.data(), id_particle_sorted.data(), nb_nbh_incr.data(), particles_start_sorted.data(), id_particle.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, id_particle.data(), id_particle_sorted.data(), nb_nbh_incr.data(), particles_start_sorted.data(), id_particle.size());
	
	cudaDeviceSynchronize();
	
	cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, id_particle.data(), id_particle_sorted.data(), nb_nbh_end.data(), particles_end_sorted.data(), id_particle.size());
	
	/*for(int i = 0; i < 100; i++)
	{
		printf("CELL: %d ID_I: %d ID_J: %d\n", cell_i_final[i], id_i_final[i], id_j_final[i]);
	}*/
	
	setParticles1<<<numBlocks, 256>>>(id_particle_sorted.data(), particles_start_sorted.data(), particles_end_sorted.data(), id_particle.size());
	
	cudaDeviceSynchronize();
	
	onika::memory::CudaMMVector<int> particles_start_final;
        onika::memory::CudaMMVector<int> particles_end_final;
      
        particles_start_final.resize(number_of_particles);
        particles_end_final.resize(number_of_particles);
	
	setParticles2<<<numBlocks, 256>>>(particles_start_final.data(), particles_end_final.data(), id_particle_sorted.data(), particles_start_sorted.data(), particles_end_sorted.data(), id_particle.size());
	
	cudaDeviceSynchronize();
	
	//for(int i = 0; i < 100; i++)
	//{
	//	printf("ID: %d START: %d END: %d\n", id_particle_sorted[i], particles_start_sorted[i], particles_end_sorted[i]);
	//}
	
	onika::memory::CudaMMVector<double> ftx;
	onika::memory::CudaMMVector<double> fty;
	onika::memory::CudaMMVector<double> ftz;
	onika::memory::CudaMMVector<double> momx;
	onika::memory::CudaMMVector<double> momy;
	onika::memory::CudaMMVector<double> momz;
	
	ftx.resize(id_i_final.size());
	fty.resize(id_i_final.size());
	ftz.resize(id_j_final.size());
	momx.resize(id_i_final.size());
	momy.resize(id_i_final.size());
	momz.resize(id_i_final.size());
	
	auto& start_old = int_neighbors.particles_start;
	auto& end_old = int_neighbors.particles_end;
	auto& id_j_old = int_neighbors.id_j;
	auto& ftx_old = int_neighbors.ft_x;
	auto& fty_old = int_neighbors.ft_y;
	auto& ftz_old = int_neighbors.ft_z;
	auto& momx_old = int_neighbors.mom_x;
	auto& momy_old = int_neighbors.mom_y;
	auto& momz_old = int_neighbors.mom_z;
	auto& id_i_SOA = int_neighbors.id_i;
	auto& cell_i_SOA = int_neighbors.cell_i;
	auto& cell_j_SOA = int_neighbors.cell_j;
	auto& p_i_SOA = int_neighbors.p_i;
	auto& p_j_SOA = int_neighbors.p_j;
	
	//getchar();
	
	/*if(int_neighbors.iterator)
	{
		numBlocks = (id_i_final.size() + 256 - 1) / 256;
		
		//update_friction_moment<<<numBlocks, 256>>>(id_i_final.data() , id_j_final.data(), start_old.data(), end_old.data(), id_j_old.data(), ftx.data(), fty.data(), ftz.data(), ftx_old.data(), fty_old.data(), ftz_old.data(), momx.data(), momy.data(), momz.data(), momx_old.data(), momy_old.data(), momz_old.data(), id_i_final.size());
		
		 //cudaDeviceSynchronize();
		 
		 /*start_old.clear();
		 end_old.clear();
		 id_j_old.clear();
		 ftx_old.clear();
		 fty_old.clear();
		 ftz_old.clear();
		 momx_old.clear();
		 momy_old.clear();
		 momz_old.clear();
		 id_i_SOA.clear();
		 cell_i_SOA.clear();
		 cell_j_SOA.clear();
		 p_i_SOA.clear();
		 p_j_SOA.clear();*/
		 
		 //for(int i = 0; i < 100; i++)
		 //{
		 //	printf("FX: %f FY: %f FZ: %f MX: %f MY: %f MZ: %f\n", ftx[i], fty[i], ftz[i], momx[i], momy[i], momz[i]);
		 //}
		 
		 //getchar();
	/*}
	else
	{
		int_neighbors.iterator = true;
	}*/
	/*
		 start_old = particles_start_sorted;;
		 end_old = particles_end_sorted;;
		 id_j_old = id_j_final;;
		 ftx_old = ftx;;
		 fty_old = fty;;
		 ftz_old = ftz;;
		 momx_old = momx;;
		 momy_old = momy;;
		 momz_old = momz;;
		 id_i_SOA = id_i_final;;
		 cell_i_SOA = cell_i_final;;
		 cell_j_SOA = cell_j_final;;
		 p_i_SOA = p_i_final;
		 p_j_SOA = p_j_final;*/	
	
	printf("CHUNK_END\n");	 
	
	//getchar();  
      
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("chunk_neighbors_contact", make_grid_variant_operator<ChunkNeighborsContact>); }

} // namespace exaDEM
