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

#include <exaDEM/classifier/classifier.hpp>

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
  	int i = 0;

  	while( i < size)
  	{
  		if(index == cells[i]) return true;
  		i++;
  	}

  	return false;
  }
  
  template< class GridT > __global__ void kernelUN(GridT* cells,
  							int* ghost_cell,
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
  							int nombre_voisins_potentiels,
  							Cylinder driver,
  							int* interaction_driver)
  {
  	
	int nb = nb_particles[blockIdx.x];
	
	if(threadIdx.x < nb)
	{
		int cell = cell_id[blockIdx.x];
		int start = cell_start[blockIdx.x];
		int end = cell_end[blockIdx.x];
		int incr = cell_particle_start[blockIdx.x];
		
		int p_a = threadIdx.x;
				
		int incr2 = incr*nombre_voisins_potentiels + p_a*nombre_voisins_potentiels;
		
		int nb_interactions = 0;
		
		const double rVerletMax = cells[cell][field::radius][p_a] + rcut_inc;
		
		auto &rx_a = cells[cell][field::rx][p_a];
		auto &ry_a = cells[cell][field::ry][p_a];
		auto &rz_a = cells[cell][field::rz][p_a];
		auto &id_a = cells[cell][field::id][p_a];
		
		const Vec3d r = { rx_a , ry_a , rz_a };
		
		if (driver.filter(rVerletMax, r))
		{
			interaction_driver[incr + p_a] = 1;
			id[incr + p_a] = id_a;
			pi[incr + p_a] = p_a;
			celli[incr + p_a] = cell;
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
				
				auto &id_b = cells[cell_b][field::id][p_b];
	  			
	  			double rcut2 = dist_lab * dist_lab;
  		
  				const Vec3d dr = { rx_a - cells[cell_b][field::rx][p_b] , ry_a - cells[cell_b][field::ry][p_b] , rz_a - cells[cell_b][field::rz][p_b] };
                		double d2 = norm2( xform * dr );
                	
  				if(nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell, p_a, cell_b, p_b)) 
  				{
  					if( id_a < id_b)
  					{
  						id_i[incr2 + nb_interactions] = id_a;
  						id_j[incr2 + nb_interactions] = id_b;
  						cell_i[incr2 + nb_interactions] = cell;
  						cell_j[incr2 + nb_interactions] = cell_b;
  						p_i[incr2 + nb_interactions] = p_a;
  						p_j[incr2 + nb_interactions] = p_b;
  						
  						nb_nbh[incr2 + nb_interactions] = 1;
  					
  						nb_interactions++;
  					}
  					else if( ghost_cell[i]==1 )
  					{
   						id_i[incr2 + nb_interactions] = id_a;
  						id_j[incr2 + nb_interactions] = id_b;
  						cell_i[incr2 + nb_interactions] = cell;
  						cell_j[incr2 + nb_interactions] = cell_b;
  						p_i[incr2 + nb_interactions] = p_a;
  						p_j[incr2 + nb_interactions] = p_b;
  						
  						nb_nbh[incr2 + nb_interactions] = 1;
  					
  						nb_interactions++;
  						
  					}
  				}				
			}
		}
	}
   }
   
  __global__ void kernelDEUX(int* nb_nbh, int* nb_nbh_incr, uint64_t* id_i, uint64_t* id_j, uint32_t* cell_i, uint32_t* cell_j, uint16_t* p_i, uint16_t* p_j, uint64_t* id_i_final, uint64_t* id_j_final, uint32_t* cell_i_final, uint32_t* cell_j_final, uint16_t* p_i_final, uint16_t* p_j_final, double* ftx_final, double* fty_final, double* ftz_final, double* momx_final, double* momy_final, double* momz_final, int size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		int incr = nb_nbh_incr[idx];
  		if(nb_nbh[idx] == 1)
  		{
  			id_i_final[incr] = id_i[idx];
  			id_j_final[incr] = id_j[idx];
  			cell_i_final[incr] = cell_i[idx];
  			cell_j_final[incr] = cell_j[idx];
  			p_i_final[incr] = p_i[idx];
  			p_j_final[incr] = p_j[idx];
  			ftx_final[incr] = 0;
  			fty_final[incr] = 0;
  			ftz_final[incr] = 0;
  			momx_final[incr] = 0;
  			momy_final[incr] = 0;
  			momz_final[incr] = 0;
  		}
  	}
  }
  
  __global__ void kernelTROIS(int* interaction_driver, int* driver_incr, int* id, int* cell, int* p, uint64_t* id_i_driver, uint32_t* cell_i_driver, uint16_t* p_i_driver, int size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		int incr = driver_incr[idx];
  		
  		if(interaction_driver[idx] == 1)
  		{
  			id_i_driver[incr] = id[idx];
  			cell_i_driver[incr] = cell[idx];
  			p_i_driver[incr] = p[idx];
  		}
  	}
  }
  
  __global__ void filtre_un( double* ft_x,
  			double* ft_y,
  			double* ft_z,
  			double* mom_x,
  			double* mom_y,
  			double* mom_z,
  			size_t size,
  			int* filtre)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		if(ft_x[idx]!=0 || ft_y[idx]!=0 || ft_z[idx]!=0 || mom_x[idx]!=0 || mom_y[idx]!=0 || mom_z[idx]!=0)
  		{
  			filtre[idx] = 1;
  		}
  	}
  }
  
  __global__ void filtre_deux( uint64_t* id_i,
  				uint64_t* id_j,
  				uint16_t* sub_j,
  				uint64_t* id_i_res,
  				uint64_t* id_j_res,
  				double* ft_x,
  				double* ft_y,
  				double* ft_z,
  				double* ft_x_res,
  				double* ft_y_res,
  				double* ft_z_res,
  				double* mom_x,
  				double* mom_y,
  				double* mom_z,
  				double* mom_x_res,
  				double* mom_y_res,
  				double* mom_z_res,
  				int* filtre_incr,
  				int* indices,
  				int size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;

  	if(idx < size)
  	{
  		if(ft_x[idx]!=0 || ft_y[idx]!=0 || ft_z[idx]!=0 || mom_x[idx]!=0 || mom_y[idx]!=0 || mom_z[idx]!=0)
  		{
  			int &incr = filtre_incr[idx];
  			
  			id_i_res[incr] = id_i[idx];
  			id_j_res[incr] = id_j[idx];
  			ft_x_res[incr] = ft_x[idx];
  			ft_y_res[incr] = ft_y[idx];
  			ft_z_res[incr] = ft_z[idx];
  			mom_x_res[incr] = mom_x[idx];
  			mom_y_res[incr] = mom_y[idx];
  			mom_z_res[incr] = mom_z[idx];
  			
  			indices[incr] = incr;
  		}
  	}
  }
  
  __global__ void generateKeys( uint64_t* keys, 
  				const uint64_t* id_i, 
  				const uint64_t* id_j,
  				int min,
  				int max,
  				int type,
  				int size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		int range = max - min + 1;
  		if(type == 0)keys[idx] = (id_i[idx] - min) * range + (id_j[idx] - min);
  		if(type == 4)keys[idx] = id_i[idx];
  	}
  }
  
  
  void exclusive_sum( int* d_in, int* d_out, int size )
  {
  	void* d_temp_storage = nullptr;
  	size_t temp_storage_bytes = 0;
  	
  	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, size);
  	
  	cudaMalloc(&d_temp_storage, temp_storage_bytes);
  	
  	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, size);
  	
  	cudaFree(d_temp_storage);
  }
  
	void sortWithIndices(uint64_t* id_in, int *indices_in, uint64_t* id_out, int* indices_out, int size) {
	    // Allocate temporary storage
	    void *d_temp_storage = nullptr;
	    size_t temp_storage_bytes = 0;

	    // Step 1: Determine temporary storage size
	    cub::DeviceRadixSort::SortPairs(
		d_temp_storage, temp_storage_bytes, 
		id_in, id_out, 
		indices_in, indices_out, 
		size);

	    // Step 2: Allocate temporary storage
	    cudaMalloc(&d_temp_storage, temp_storage_bytes);

	    // Step 3: Perform the radix sort (key-value pair sorting)
	    cub::DeviceRadixSort::SortPairs(
		d_temp_storage, temp_storage_bytes, 
		id_in, id_out, 
		indices_in, indices_out, 
		size);

	    // Free temporary storage and double-buffered arrays
	    cudaFree(d_temp_storage);
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
    
    //ADD_SLOT(InteractionSOA, interaction_type0, INPUT_OUTPUT);
    //ADD_SLOT(InteractionSOA, interaction_type4, INPUT_OUTPUT);
    
    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, DocString{"List of Drivers"});
    
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});
    
    ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
    ADD_SLOT(OldClassifiers, ic_olds, INPUT_OUTPUT);

    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator builds the neighbor lists used to detect contacts between particles.
        )EOF";
    }

    inline void execute() override final
    {
    
      //printf("CHUNK_START\n");
      
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
      
      //UNCLASSIFY
      //printf("UNCLASSIFY\n");
      
      if (ic.has_value())
      {
      
      auto &olds = *ic_olds;
      
      if(olds.use)
      {
      	olds.waves.resize(13);
      	
      	olds.use = false;
      }
      
      auto& c = *ic;
      
      for(int type = 0; type < 13; type++)
      {
      
      auto [data, size] = c.get_info(type);
      
      if(size > 0)
      {
      
      //printf("PASSAGE_%d\n", type);
      
      InteractionWrapper<InteractionSOA> interactions(data);
      
      int blockSize = 256;
      int numBlocks = ( size + blockSize - 1 ) / blockSize;
      
      
      //onika::memory::CudaMMVector<int> blocks;
      //blocks.resize(numBlocks);
      
      //onika::memory::CudaMMVector<int> blocks_incr;
      //blocks_incr.resize(numBlocks);
      
      //onika::memory::CudaMMVector<int> total;
      //total.resize(1);
      
      onika::memory::CudaMMVector<int> filtre;
      filtre.resize(size);
      
      filtre_un<<<numBlocks, blockSize>>>( interactions.ft_x, interactions.ft_y, interactions.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, size, filtre.data());
      
      onika::memory::CudaMMVector<int> filtre_incr;
      filtre_incr.resize(size);
      
      exclusive_sum( filtre.data(), filtre_incr.data(), size );
      
      int total = filtre_incr[filtre_incr.size() - 1] + filtre[filtre.size() - 1];
      
      onika::memory::CudaMMVector<uint64_t> id_i_res;
      id_i_res.resize(total);
      
      onika::memory::CudaMMVector<uint64_t> id_j_res;
      id_j_res.resize(total);
      
      onika::memory::CudaMMVector<uint16_t> sub_j_res;
      sub_j_res.resize(total);
	
      onika::memory::CudaMMVector<int> indices;
      indices.resize(total);
      
       onika::memory::CudaMMVector<uint64_t> keys;
       keys.resize(total);
      
      auto &o = olds.waves[type];
      
      o.set( total );
      
      OldClassifierWrapper old(o);
      
      filtre_deux<<<numBlocks, blockSize>>>( interactions.id_i, interactions.id_j, interactions.sub_j, id_i_res.data(), id_j_res.data(), interactions.ft_x, interactions.ft_y, interactions.ft_z, old.ft_x, old.ft_y, old.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, old.mom_x, old.mom_y, old.mom_z, filtre_incr.data(), indices.data(), size);
      
       cudaDeviceSynchronize();
       
       int min = 0;
       int max = grid->number_of_particles() - 1;
       
       numBlocks = ( total + blockSize - 1 ) / blockSize;

       generateKeys<<<numBlocks, blockSize>>>( keys.data(), id_i_res.data(), id_j_res.data(), min, max, type, total);
       
       sortWithIndices( keys.data(), indices.data(), old.keys, old.indices, total);
       
      }
      }      
      
      }
      //printf("UNCLASSIFY END\n");
      
      
      //NOUVELLE MÉTHODE
      //printf("CHUNK\n");
 
      auto number_of_particles = grid->number_of_particles();
       
      onika::memory::CudaMMVector<int> nb_particles_cell;

      onika::memory::CudaMMVector<int> cell_particles_neighbors;

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
            				cell_particles_neighbors[cell_a*27 + nb] = cell_b;
            				cell_particles_neighbors_size[cell_a*27 + nb] = n_particles_b;
            				nb++;
            			}
            		}
            		
            		cell_particles_number_of_neighbors_cells[cell_a] = nb;         		
			
		}
		GRID_OMP_FOR_END
	}

	onika::memory::CudaMMVector<int> cell_id;
	onika::memory::CudaMMVector<int> incr_cell_id;
	
	auto [cell_ptr, cell_size] = traversal_real->info();
	
	int incr_cell = 0;

	for(int i=0; i < nb_particles_cell.size(); i++)
	{
		if( nb_particles_cell[i] > 0 && is_in(i, cell_ptr, cell_size) ){ cell_id.push_back(i); incr_cell_id.push_back(incr_cell); incr_cell+= cell_particles_number_of_neighbors_cells[i]; }
	}

	onika::memory::CudaMMVector<int> nb_particles; //NOMBRE DE PARTICULES CELLULE_I
	nb_particles.resize(cell_id.size());
	
	onika::memory::CudaMMVector<int> cell_nb_nbh; //NOMBRE DE CELLULES VOISINES
	cell_nb_nbh.resize(cell_id.size());
	
	onika::memory::CudaMMVector<int> cell_neighbors_ids;
	cell_neighbors_ids.resize(incr_cell);
	
	onika::memory::CudaMMVector<int> cell_neighbors_size;
	cell_neighbors_size.resize(incr_cell);
	
	onika::memory::CudaMMVector<int> ghost_cell;
	ghost_cell.resize(incr_cell);
	
	#pragma omp parallel for
	for(int i=0; i < cell_id.size(); i++)
	{
		int index = cell_id[i];
		
		nb_particles[i] = nb_particles_cell[index];

		cell_nb_nbh[i] = cell_particles_number_of_neighbors_cells[index];
		
		int incr = incr_cell_id[i];

		for(int j = 0; j < cell_nb_nbh[i]; j++)
		{
			cell_neighbors_ids[incr + j] = cell_particles_neighbors[index*27 + j];
			
			ghost_cell[incr + j] = g.is_ghost_cell( cell_particles_neighbors[index*27 + j]);
			
			cell_neighbors_size[incr + j] = cell_particles_neighbors_size[index*27 + j];	
		}
	}
	
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

	onika::memory::CudaMMVector<int> interaction_driver;
	
	int total = 0;
	for(int i = 0; i < nb_particles.size(); i++)
	{
		total+= nb_particles[i];
	}

	id_particle.resize(total);
	cell_particle.resize(total);
	p_particle.resize(total);
	interaction_driver.resize(total);
	
	int nombre_voisins_potentiels = 20;
	
	total = total* nombre_voisins_potentiels;
	
	nb_nbh.resize(total);
	id_i.resize(total);
	id_j.resize(total);
	cell_i.resize(total);
	cell_j.resize(total);
	p_i.resize(total);
	p_j.resize(total);
	
	auto &drvs = *drivers;
	Cylinder &driver = std::get<Cylinder>(drvs.data(0));
	
	kernelUN<<<numBlocks, 256>>>(cells, ghost_cell.data(), cell_id.data(), nb_particles.data(), cell_neighbors_ids.data(), cell_neighbors_size.data(), cell_start.data(), cell_end.data(), nb_particles_start.data(), *nbh_dist_lab, domain->xform(), *rcut_inc, id_i.data(), id_j.data(), cell_i.data(), cell_j.data(), p_i.data(), p_j.data(), nb_nbh.data(), id_particle.data(), cell_particle.data(), p_particle.data(), nombre_voisins_potentiels, driver, interaction_driver.data());
	
	cudaDeviceSynchronize();

	onika::memory::CudaMMVector<int> nb_nbh_incr;
	nb_nbh_incr.resize(nb_nbh.size());
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), nb_nbh.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), nb_nbh.size());
	
	cudaFree(d_temp_storage);
	
	int total_interactions = nb_nbh_incr[nb_nbh.size() - 1] + nb_nbh[nb_nbh.size() - 1];
	
	onika::memory::CudaMMVector<int> driver_incr;
	driver_incr.resize(interaction_driver.size());
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, interaction_driver.data(), driver_incr.data(), interaction_driver.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, interaction_driver.data(), driver_incr.data(), interaction_driver.size());
	
	cudaFree(d_temp_storage);
	
	int total_interactions_driver = driver_incr[interaction_driver.size() - 1] + interaction_driver[interaction_driver.size() - 1];
	
	//printf("CHUNK END\n");
	
	//CLASSIFIER
	
	//printf("CLASSIFIER\n");
	
	if (!ic.has_value())
        	ic->initialize();
	
	auto& type0 = ic->get_wave(0);
	auto& type4 = ic->get_wave(4);
	
	type0.clear();
	type4.clear();
	
	type0.type = 0;
	type4.type = 4;
	
	type0.resize(total_interactions);
	type4.resize(total_interactions_driver);
	
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
	
	kernelDEUX<<<numBlocks, 256>>>(nb_nbh.data(), nb_nbh_incr.data(), id_i.data(), id_j.data(), cell_i.data(), cell_j.data(), p_i.data(), p_j.data(), id_i_final.data(), id_j_final.data(), cell_i_final.data(), cell_j_final.data(), p_i_final.data(), p_j_final.data(), ftx_final.data(), fty_final.data(), ftz_final.data(), momx_final.data(), momy_final.data(), momz_final.data(), nb_nbh.size());
	
	numBlocks = (interaction_driver.size() + 256 - 1) / 256;
	
	kernelTROIS<<<numBlocks, 256>>>(interaction_driver.data(), driver_incr.data(), id_particle.data(), cell_particle.data(), p_particle.data(), id_i_driver.data(), cell_i_driver.data(), p_i_driver.data(), interaction_driver.size());

	//printf("CLASSIFIER END\n");	
		
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("chunk_neighbors_contact", make_grid_variant_operator<ChunkNeighborsContact>); }

} // namespace exaDEM
