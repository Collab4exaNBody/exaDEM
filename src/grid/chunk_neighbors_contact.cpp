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

#include <exanb/particle_neighbors/chunk_neighbors_execute.h>

#include <cub/cub.cuh>

#include <exaDEM/classifier/interactionSOA.hpp>

#include <exaDEM/drivers.h>
#include <exaDEM/shape_detection_driver.hpp>

#include <exaDEM/traversal.hpp>

namespace exaDEM
{
  using namespace exanb;
  
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
  
  template< class GridT > bool filter( GridT* cells, double rcut_inc, double d2, double rcut2, int cell_a, int p_a, int cell_b, int p_b)
  {
      const double r_a = cells[cell_a][field::radius][p_a];
      const double r_b = cells[cell_b][field::radius][p_b];
      const double rmax = r_a + r_b + rcut_inc;
      const double rmax2 = rmax * rmax;
      return d2 > 0.0 && d2 < rmax2 && d2 < rcut2 ;  	
  }
  
  bool is_in(int index, size_t* cells, size_t size)
  {
  	//bool b = true;
  	int i = 0;

  	while( i < size)
  	{
  		if(index == cells[i]) return true;
  		
  		//if(index > cells[i]) b = false;
  		
  		i++;
  	}

  	return false;
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
  							int* celli,
  							int* pi,
  							int* total_interactions,
  							int* total_interactions_driver,
  							int nombre_voisins_potentiels,
  							Cylinder driver,
  							int* interaction_driver )
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
		
		const double rVerletMax = cells[cell][field::radius][p_a] + rcut_inc;
		
		const Vec3d r = {cells[cell][field::rx][p_a], cells[cell][field::ry][p_a], cells[cell][field::rz][p_a]};
		
		if (driver.filter(rVerletMax, r))
		{
			atomicAdd(&total_interactions_driver[0], 1);
			interaction_driver[incr + p_a] = 1;
			
		}
		else
		{
			interaction_driver[incr + p_a] = 0;
		}
		
		for(int i = start; i < end; i++)
		{
			int cell_b = cell_neighbors_ids[i];
			
			for(int j = 0; j < cell_neighbors_size[i]; j++)
			{
				int p_b = j;
	  			
	  			double rcut2 = dist_lab * dist_lab;
  		
  				const Vec3d dr = { cells[cell][field::rx][p_a] - cells[cell_b][field::rx][p_b] , cells[cell][field::ry][p_a] - cells[cell_b][field::ry][p_b] , cells[cell][field::rz][p_a] - cells[cell_b][field::rz][p_b] };
                		double d2 = norm2( xform * dr );
                	
  				if(nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell, p_a, cell_b, p_b) && (cells[cell][field::id][p_a] < cells[cell_b][field::id][p_b]) && (cell!=cell_b || p_a!=p_b)) 
  				{ 
  					id_i[incr*nombre_voisins_potentiels + p_a*nombre_voisins_potentiels + nb_interactions] = cells[cell][field::id][p_a];
  					id_j[incr*nombre_voisins_potentiels + p_a*nombre_voisins_potentiels + nb_interactions] = cells[cell_b][field::id][p_b];
  					cell_i[incr*nombre_voisins_potentiels + p_a*nombre_voisins_potentiels + nb_interactions] = cell;
  					cell_j[incr*nombre_voisins_potentiels + p_a*nombre_voisins_potentiels + nb_interactions] = cell_b;
  					p_i[incr*nombre_voisins_potentiels + p_a*nombre_voisins_potentiels + nb_interactions] = p_a;
  					p_j[incr*nombre_voisins_potentiels + p_a*nombre_voisins_potentiels + nb_interactions] = p_b;
  					
  					nb_interactions++;
  					
  					atomicAdd(&total_interactions[0], 1);
  				}				
			}
		}
		
		nb_nbh[incr + p_a] = nb_interactions;
		id[incr + p_a] = cells[cell][field::id][p_a];
		celli[incr + p_a] = cell;
		pi[incr + p_a] = p_a;
	}
   }
   
  __global__ void kernelDEUX(int* nb_nbh, int* nb_nbh_incr, uint64_t* id_i, uint64_t* id_j, uint32_t* cell_i, uint32_t* cell_j, uint16_t* p_i, uint16_t* p_j, uint64_t* id_i_final, uint64_t* id_j_final, uint32_t* cell_i_final, uint32_t* cell_j_final, uint16_t* p_i_final, uint16_t* p_j_final, double* ftx_final, double* fty_final, double* ftz_final, double* momx_final, double* momy_final, double* momz_final, int n, int nombre_voisins_potentiels, int* id, int* cell, int* p, uint64_t* id_i_driver, uint64_t* id_j_driver, uint32_t* cell_i_driver, uint16_t* p_i_driver, int* interaction_driver, int* driver_incr, double* ftx_driver, double* fty_driver, double* ftz_driver, double* momx_driver, double* momy_driver, double* momz_driver)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < n)
  	{
  		int nb = nb_nbh[idx];
  		int incr = nb_nbh_incr[idx];
  		
  		int incr_driver = driver_incr[idx];
  		
  		if(interaction_driver[idx] == 1)
  		{
  			id_i_driver[incr_driver] = id[idx];
  			id_j_driver[incr_driver] = 0;
  			cell_i_driver[incr_driver] = cell[idx];
  			p_i_driver[incr_driver] = p[idx];
  			ftx_driver[incr_driver] = 0;
  			fty_driver[incr_driver] = 0;
  			ftz_driver[incr_driver] = 0;
  			momx_driver[incr_driver] = 0;
  			momy_driver[incr_driver] = 0;
  			momz_driver[incr_driver] = 0;
  		}
  		
  		for(int i = 0; i < nb; i++)
  		{
  			id_i_final[incr + i] = id_i[idx*nombre_voisins_potentiels + i];
  			id_j_final[incr + i] = id_j[idx*nombre_voisins_potentiels + i];
  			cell_i_final[incr + i] = cell_i[idx*nombre_voisins_potentiels + i];
  			cell_j_final[incr + i] = cell_j[idx*nombre_voisins_potentiels + i];
  			p_i_final[incr + i] = p_i[idx*nombre_voisins_potentiels + i];
  			p_j_final[incr + i] = p_j[idx*nombre_voisins_potentiels + i];
  			ftx_final[incr + i] = 0;
  			fty_final[incr + i] = 0;
  			ftz_final[incr + i] = 0;
  			momx_final[incr + i] = 0;
  			momy_final[incr + i] = 0;
  			momz_final[incr + i] = 0;
  		}
  	}
  }

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
    
    ADD_SLOT(InteractionSOA, interaction_type0, INPUT_OUTPUT);
    ADD_SLOT(InteractionSOA, interaction_type4, INPUT_OUTPUT);
    
    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, DocString{"List of Drivers"});
    
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});

    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator builds the neighbor lists used to detect contacts between particles.
        )EOF";
    }

    inline void execute() override final
    {
    
      printf("CHUNK_START\n");
      
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

      /*if (!domain->xform_is_identity())
      {
        LinearXForm xform = {domain->xform()};
        chunk_neighbors_execute(ldbg, *chunk_neighbors, *grid, *amr, *amr_grid_pairs, *config, *chunk_neighbors_scratch, cs, cs_log2, *nbh_dist_lab, xform, gpu_enabled, no_z_order, nbh_filter);
      }
      else
      {
        NullXForm xform = {};
        chunk_neighbors_execute(ldbg, *chunk_neighbors, *grid, *amr, *amr_grid_pairs, *config, *chunk_neighbors_scratch, cs, cs_log2, *nbh_dist_lab, xform, gpu_enabled, no_z_order, nbh_filter);
      }*/
      
      
      //NOUVELLE MÉTHODE
 
      auto number_of_particles = grid->number_of_particles();
       
      onika::memory::CudaMMVector<int> nb_particles_cell;
      //onika::memory::CudaMMVector<onika::memory::CudaMMVector<int>> cell_particles_neighbors;
      onika::memory::CudaMMVector<int> cell_particles_neighbors;
      //onika::memory::CudaMMVector<onika::memory::CudaMMVector<int>> cell_particles_neighbors_size;
      onika::memory::CudaMMVector<int> cell_particles_neighbors_size;
      
      onika::memory::CudaMMVector<int> cell_particles_number_of_neighbors_cells;
      
      	auto& g = *grid;

      	IJK dims = g.dimension();
      	
      	auto& amr_gpu = *amr;
      	const size_t* sub_grid_start = amr_gpu.sub_grid_start().data();
      	const uint32_t* sub_grid_cells = amr_gpu.sub_grid_cells().data();
      	
      	const double max_dist = *nbh_dist_lab;
      	const double max_dist2 = max_dist*max_dist;
      	
      	auto& amr_grid_pairs_gpu = *amr_grid_pairs;
      	const unsigned int loc_max_gap = amr_grid_pairs_gpu.cell_layers();

      	const unsigned int n_nbh_cell = amr_grid_pairs_gpu.nb_nbh_cells();

      	
      	const size_t n_cells = g.number_of_cells();

      	nb_particles_cell.resize( n_cells );
      	cell_particles_neighbors.resize( n_cells*27 );
      	cell_particles_neighbors_size.resize( n_cells*27 );
      	
      	cell_particles_number_of_neighbors_cells.resize(n_cells);
      	
      	/*#pragma omp parallel for
      	for(int i = 0; i < n_cells; ++i)
      	{
      		cell_particles_neighbors[i].resize(27);
      		cell_particles_neighbors_size[i].resize(27);
      	}*/
 
      	unsigned int max_threads = omp_get_max_threads();

#	pragma omp parallel
	{
		int tid = omp_get_thread_num();
		assert( tid>=0 && size_t(tid)<max_threads );
		
		GRID_OMP_FOR_BEGIN(dims,cell_a,loc_a, schedule(dynamic))
		{
			size_t n_particles_a = cells[cell_a].size();
			
			nb_particles_cell[cell_a] = n_particles_a;

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
         		
         		int nb = 0;
         		
          		for(ssize_t loc_bk=bstartk;loc_bk<=bendk;loc_bk++)
          		for(ssize_t loc_bj=bstartj;loc_bj<=bendj;loc_bj++)
          		for(ssize_t loc_bi=bstarti;loc_bi<=bendi;loc_bi++)
          		{
            			IJK loc_b { loc_bi, loc_bj, loc_bk };
            			ssize_t cell_b = grid_ijk_to_index( dims, loc_b );
            			size_t n_particles_b = cells[cell_b].size();
            			
            			if( n_particles_b > 0)
            			{
            				//cell_particles_neighbors[cell_a][nb] = cell_b;
            				cell_particles_neighbors[cell_a*27 + nb] = cell_b;
            				//cell_particles_neighbors_size[cell_a][nb] = n_particles_b;
            				cell_particles_neighbors_size[cell_a*27 + nb] = n_particles_b;
            				nb++;
            			}
            		}
            		
            		cell_particles_number_of_neighbors_cells[cell_a] = nb;         		
			
		}
		GRID_OMP_FOR_END
	}

	onika::memory::CudaMMVector<int> cell_id;
	
	auto [cell_ptr, cell_size] = traversal_real->info();

	for(int i=0; i < nb_particles_cell.size(); i++)
	{
		if( nb_particles_cell[i] > 0 && is_in(i, cell_ptr, cell_size) ) cell_id.push_back(i);
	}
	
	
	onika::memory::CudaMMVector<int> nb_particles; //NOMBRE DE PARTICULES CELLULE_I
	nb_particles.resize(cell_id.size());
	
	onika::memory::CudaMMVector<int> cell_nb_nbh; //NOMBRE DE CELLULES VOISINES
	cell_nb_nbh.resize(cell_id.size());
	
	onika::memory::CudaMMVector<int> cell_neighbors_ids;
	
	onika::memory::CudaMMVector<int> cell_neighbors_size;
	
	for(int i=0; i < cell_id.size(); i++)
	{
		int index = cell_id[i];
		
		nb_particles[i] = nb_particles_cell[index];

		cell_nb_nbh[i] = cell_particles_number_of_neighbors_cells[index];

		for(int j = 0; j < cell_nb_nbh[i]; j++)
		{
			//cell_neighbors_ids.push_back(cell_particles_neighbors[index][j]);
			cell_neighbors_ids.push_back(cell_particles_neighbors[index*27 + j]);
			//cell_neighbors_size.push_back(cell_particles_neighbors_size[index][j]);
			cell_neighbors_size.push_back(cell_particles_neighbors_size[index*27 + j]);	
		}
	}
	
	/*
	for(int i = 0; i < cell_id.size(); i++)
	{
		int cell_a = cell_id[i];
		
		for(int j = 0; j < nb_particles[i]; j++)
		{
			
			int p_a = j;
			
			for(int z = 0; z < cell_particles_neighbors[cell_a].size(); z++)
			{
				int cell_b = cell_particles_neighbors[cell_a][z];
				
				for(int k = 0; k < cell_particles_neighbors_size[cell_a][z]; k++)
				{
					
					int p_b = k;
					
	  				double rcut2 = *nbh_dist_lab * *nbh_dist_lab;
  					const Vec3d dr = { cells[cell_a][field::rx][p_a] - cells[cell_b][field::rx][p_b] , cells[cell_a][field::ry][p_a] - cells[cell_b][field::ry][p_b] , cells[cell_a][field::rz][p_a] - cells[cell_b][field::rz][p_b] };
                			double d2 = norm2( domain->xform() * dr );
                			
                			if( filter(cells, *rcut_inc, d2, rcut2, cell_a, p_a, cell_b, p_b) && (cells[cell_a][field::id][p_a] < cells[cell_b][field::id][p_b]) && (cell_a!=cell_b || p_a!=p_b))  totAAL++;						
				}
			
			}
		}
	}*/
	
	//CELL START
	onika::memory::CudaMMVector<int> cell_start;
	cell_start.resize(cell_id.size());

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
	onika::memory::CudaMMVector<int> cell_particle;
	onika::memory::CudaMMVector<int> p_particle;
	
	onika::memory::CudaMMVector<int> total_interactions;
	onika::memory::CudaMMVector<int> total_interactions_driver;
	
	onika::memory::CudaMMVector<int> interaction_driver;
	
	total_interactions.resize(1);
	total_interactions_driver.resize(1);

	int total = 0;
	for(int i = 0; i < nb_particles.size(); i++)
	{
		total+= nb_particles[i];
	}
	
	nb_nbh.resize(total);
	id_particle.resize(total);
	cell_particle.resize(total);
	p_particle.resize(total);
	interaction_driver.resize(total);
	
	int nombre_voisins_potentiels = 6;
	
	total = total* nombre_voisins_potentiels;
	
	id_i.resize(total);
	id_j.resize(total);
	cell_i.resize(total);
	cell_j.resize(total);
	p_i.resize(total);
	p_j.resize(total);
	
	auto &drvs = *drivers;
	Cylinder &driver = std::get<Cylinder>(drvs.data(0));
	
	
	
	kernelUN<<<numBlocks, 256>>>(cells, cell_id.data(), nb_particles.data(), cell_neighbors_ids.data(), cell_neighbors_size.data(), cell_start.data(), cell_end.data(), nb_particles_start.data(), *nbh_dist_lab, domain->xform(), *rcut_inc, id_i.data(), id_j.data(), cell_i.data(), cell_j.data(), p_i.data(), p_j.data(), nb_nbh.data(), id_particle.data(), cell_particle.data(), p_particle.data(), total_interactions.data(), total_interactions_driver.data(), nombre_voisins_potentiels, driver, interaction_driver.data());
	
	cudaDeviceSynchronize();

	onika::memory::CudaMMVector<int> nb_nbh_incr;
	nb_nbh_incr.resize(nb_nbh.size());
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), nb_nbh.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), nb_nbh.size());
	
	cudaFree(d_temp_storage);
	
	onika::memory::CudaMMVector<int> driver_incr;
	driver_incr.resize(interaction_driver.size());
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, interaction_driver.data(), driver_incr.data(), interaction_driver.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, interaction_driver.data(), driver_incr.data(), interaction_driver.size());
	
	cudaFree(d_temp_storage); 
	
	auto& type0 = *interaction_type0;
	auto& type4 = *interaction_type4;
	
	type0.clear();
	type4.clear();
	
	type0.type = 0;
	type4.type = 4;
	
	type0.resize(total_interactions[0]);
	type4.resize(total_interactions_driver[0]);
	
	onika::memory::CudaMMVector<uint64_t> &id_i_final = type0.id_i;
	onika::memory::CudaMMVector<uint64_t> &id_j_final = type0.id_j;
	onika::memory::CudaMMVector<uint32_t> &cell_i_final = type0.cell_i;
	onika::memory::CudaMMVector<uint32_t> &cell_j_final = type0.cell_j;
	onika::memory::CudaMMVector<uint16_t> &p_i_final = type0.p_i;
	onika::memory::CudaMMVector<uint16_t> &p_j_final = type0.p_j;
	onika::memory::CudaMMVector<double> &ftx_final = type0.ft_x;
	onika::memory::CudaMMVector<double> &fty_final = type0.ft_y;
	onika::memory::CudaMMVector<double> &ftz_final = type0.ft_z;
	onika::memory::CudaMMVector<double> &momx_final = type0.mom_x;
	onika::memory::CudaMMVector<double> &momy_final = type0.mom_y;
	onika::memory::CudaMMVector<double> &momz_final = type0.mom_z;
	
	onika::memory::CudaMMVector<uint64_t> &id_i_driver = type4.id_i;
	onika::memory::CudaMMVector<uint64_t> &id_j_driver = type4.id_j;
	onika::memory::CudaMMVector<uint32_t> &cell_i_driver = type4.cell_i;
	onika::memory::CudaMMVector<uint16_t> &p_i_driver = type4.p_i;
	onika::memory::CudaMMVector<double> &ftx_driver = type4.ft_x;
	onika::memory::CudaMMVector<double> &fty_driver = type4.ft_y;
	onika::memory::CudaMMVector<double> &ftz_driver = type4.ft_z;
	onika::memory::CudaMMVector<double> &momx_driver = type4.mom_x;
	onika::memory::CudaMMVector<double> &momy_driver = type4.mom_y;
	onika::memory::CudaMMVector<double> &momz_driver = type4.mom_z;

	numBlocks = (nb_nbh.size() + 256 - 1) / 256;
	
	kernelDEUX<<<numBlocks, 256>>>(nb_nbh.data(), nb_nbh_incr.data(), id_i.data(), id_j.data(), cell_i.data(), cell_j.data(), p_i.data(), p_j.data(), id_i_final.data(), id_j_final.data(), cell_i_final.data(), cell_j_final.data(), p_i_final.data(), p_j_final.data(), ftx_final.data(), fty_final.data(), ftz_final.data(), momx_final.data(), momy_final.data(), momz_final.data(), nb_nbh.size(), nombre_voisins_potentiels, id_particle.data(),  cell_particle.data(), p_particle.data(), id_i_driver.data(), id_j_driver.data(), cell_i_driver.data(), p_i_driver.data(), interaction_driver.data(), driver_incr.data(), ftx_driver.data(), fty_driver.data(), ftz_driver.data(), momx_driver.data(), momy_driver.data(), momz_driver.data());

	printf("CHUNK_END\n");	
		
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("chunk_neighbors_contact", make_grid_variant_operator<ChunkNeighborsContact>); }

} // namespace exaDEM
