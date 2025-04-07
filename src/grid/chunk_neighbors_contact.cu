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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>

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

#include <exaDEM/traversal.h>

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

  template< class GridT > __global__ void kernelDriver(GridT* cells,
  							int* cell_id,
  							int* cell_particle_start,
  							double rcut_inc,
  							Cylinder driver,
  							int* interaction_driver,
  							int* p,
  							int* cell)
  {
  	int cell_a = cell_id[blockIdx.x];
  	int num_particles_a = cells[cell_a].size();
  	
  	if(threadIdx.x < num_particles_a)
  	{
  		int p_a = threadIdx.x;
  		int incr = cell_particle_start[blockIdx.x];
  		
  		const double rVerletMax = cells[cell_a][field::radius][p_a] + rcut_inc;
  		
  		auto rx_a = cells[cell_a][field::rx][p_a];
  		auto ry_a = cells[cell_a][field::ry][p_a];
  		auto rz_a = cells[cell_a][field::rz][p_a];
  		auto id_a = cells[cell_a][field::id][p_a];
  		
  		const Vec3d r = {rx_a, ry_a, rz_a};
  		
  		if(driver.filter(rVerletMax, r))
  		{
  			interaction_driver[incr + p_a] = 1;
  			p[incr + p_a] = p_a;
  			cell[incr + p_a] = cell_a;
  		}
  	}
  }
  
  template< class GridT > __global__ void kernelTROIS(GridT* cells, /*int* cell_id, int* cell_particle_start,*/ int* interaction_driver, int* driver_incr, uint64_t* id_i_driver, uint32_t* cell_i_driver, uint16_t* p_i_driver, int* p, int* cell, int size )
  {
  	//int cell_a = cell_id[blockIdx.x];
  	//int num_particles_a = cells[cell_a].size();
  	
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	//if(threadIdx.x < num_particles_a)
  	if(idx < size)
  	{
  		//int p_a = threadIdx.x;
  		//int incr = cell_particle_start[blockIdx.x];
  		
  		if(interaction_driver[idx]==1)
  		{
  			int incr = driver_incr[idx];
  			
  			int cell_a = cell[idx];
  			int p_a = p[idx];
  			
  			id_i_driver[incr] = cells[cell_a][field::id][p_a];
  			cell_i_driver[incr] = cell_a;
  			p_i_driver[incr] = p_a;
  		}  		
  	}
  }
  

  
  template< class GridT > __global__ void kernelUN(GridT* cells,
  							int* ghost_cell,
  							int* cell_id,
  							int* cell_neighbors_ids,
  							int* cell_start,
  							const double dist_lab,
  							Mat3d xform,
  							double rcut_inc,
  							int* nb_nbh,
  							int* p_i,
  							int* p_j,
  							int* cell_i,
  							int* cell_j,
  							int* cell_nb_nbh)
  {
  	
	//int nb = nb_particles[blockIdx.x];
	int num_cells_nbh = cell_nb_nbh[blockIdx.x];
	
	double rcut2 = dist_lab * dist_lab;
	
	int nb_interactions = 0;
	
	int global_x = blockIdx.x * blockDim.x + threadIdx.x;
	int global_y = blockIdx.y * blockDim.y + threadIdx.y;
	
	int grid_width = gridDim.x * blockDim.x;
	int gid = global_y * grid_width + global_x;
	
	for( int cellb = 0; cellb < num_cells_nbh; cellb++ )
	{
		int cell_a = cell_id[blockIdx.x];
		int start = cell_start[blockIdx.x];
		
		int cell_b = cell_neighbors_ids[start + cellb];
		
		int num_particles_a = cells[cell_a].size();
		int num_particles_b = cells[cell_b].size();
		
		for(int p_b = threadIdx.y; p_b < num_particles_b; p_b+= blockDim.y)
		{
			for(int p_a = threadIdx.x; p_a < num_particles_a; p_a+= blockDim.x)
			{
				const Vec3d dr = { cells[cell_a][field::rx][p_a] - cells[cell_b][field::rx][p_b] , cells[cell_a][field::ry][p_a] - cells[cell_b][field::ry][p_b] , cells[cell_a][field::rz][p_a] - cells[cell_b][field::rz][p_b] };
                		double d2 = norm2( xform * dr );
                		
                		if(nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell_a, p_a, cell_b, p_b)) 
                		{
                			if( cells[cell_a][field::id][p_a] < cells[cell_b][field::id][p_b] )
                			{
                				nb_nbh[gid * 30 + nb_interactions] = 1;
                				p_i[gid * 30 + nb_interactions] = p_a;
                				p_j[gid * 30 + nb_interactions] = p_b;
                				cell_i[gid * 30 + nb_interactions] = cell_a;
                				cell_j[gid * 30 + nb_interactions] = cell_b;
                				
                				nb_interactions++;
                			}
                			else if( ghost_cell[start + cellb] == 1 )
                			{
                				nb_nbh[gid * 30 + nb_interactions] = 1;
                				p_i[gid * 30 + nb_interactions] = p_a;
                				p_j[gid * 30 + nb_interactions] = p_b;
                				cell_i[gid * 30 + nb_interactions] = cell_a;
                				cell_j[gid * 30 + nb_interactions] = cell_b;
                				
                				nb_interactions++;
                			}
                		}
			}
		}
	}
}
	
  template < class GridT > __global__ void kernelDEUX(GridT* cells, int* nb_nbh, int* nb_nbh_incr, uint64_t* id_i_final, uint64_t* id_j_final, uint32_t* cell_i_final, uint32_t* cell_j_final, uint16_t* p_i_final, uint16_t* p_j_final, double* ftx_final, double* fty_final, double* ftz_final, double* momx_final, double* momy_final, double* momz_final, int* p_i, int* p_j, int* cell_i, int* cell_j, int size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		int incr = nb_nbh_incr[idx];
  		if(nb_nbh[idx] == 1)
  		{
  			int cell_a = cell_i[idx];
  			int cell_b = cell_j[idx];
  			int p_a = p_i[idx];
  			int p_b = p_j[idx];
  			
  			id_i_final[incr] = cells[cell_a][field::id][p_a];
  			id_j_final[incr] = cells[cell_b][field::id][p_b];
  			cell_i_final[incr] = cell_a;
  			cell_j_final[incr] = cell_b;
  			p_i_final[incr] = p_a;
  			p_j_final[incr] = p_b;
  			/*ftx_final[incr] = 0;
  			fty_final[incr] = 0;
  			ftz_final[incr] = 0;
  			momx_final[incr] = 0;
  			momy_final[incr] = 0;
  			momz_final[incr] = 0;*/
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
      /*
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
      
      }*/
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
      	const unsigned int nbh_cell_side = loc_max_gap+1;

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
	
	/*int max3 = 0;
	
	for(auto max: nb_particles_cell)
	{
		if(max > max3) max3 = max;
	}
	
	printf("MAX: %d\n", max3);*/
	
	onika::memory::CudaMMVector<int> cell_id;
	onika::memory::CudaMMVector<int> incr_cell_id;
	
	auto [cell_ptr, cell_size] = traversal_real->info();
	
	//printf("NUM_CELLS: %d\n", cell_size);
	
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
	
	//CELL PARTICLES START		

	onika::memory::CudaMMVector<int> nb_particles_start;
	nb_particles_start.resize(nb_particles.size());
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_particles.data(), nb_particles_start.data(), nb_particles.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_particles.data(), nb_particles_start.data(), nb_particles.size());
	
	cudaFree(d_temp_storage);	
	
	int numCells = cell_id.size();
	
	int numBlocks = numCells;
	
	onika::memory::CudaMMVector<int> nb_nbh;
	//int* nb_nbh;
	onika::memory::CudaMMVector<int> p_i;
	//int* p_i;
	onika::memory::CudaMMVector<int> p_j;
	//int* p_j;
	onika::memory::CudaMMVector<int> cell_i;
	//int* cell_i;
	onika::memory::CudaMMVector<int> cell_j;
	//int* cell_j;
	
	onika::memory::CudaMMVector<int> p_particle;
	//int* p_particle;
	onika::memory::CudaMMVector<int> cell_particle;
	//int* cell_particle;

	onika::memory::CudaMMVector<int> interaction_driver;
	//int* interaction_driver;

	int total = 0;
	for(int i = 0; i < nb_particles.size(); i++)
	{
		total+= nb_particles[i];
	}
	
	interaction_driver.resize(total);
	//cudaMalloc(&interaction_driver, total * sizeof(int) );
	p_particle.resize(total);
	//cudaMalloc(&p_particle, total * sizeof(int) );
	//cudaMalloc(&cell_particle, total * sizeof(int) );
	cell_particle.resize(total);
	
	int nombre_voisins_potentiels = 30;
	
	constexpr int block_x = 32;
        constexpr int block_y = 32;
        
        int total_1 = total;
	
	total = numBlocks * block_x * block_y * 30;
	
	nb_nbh.resize(total);
	//cudaMalloc(&nb_nbh, total * sizeof(int) );
	p_i.resize(total);
	//cudaMalloc(&p_i, total * sizeof(int) );
	p_j.resize(total);
	//cudaMalloc(&p_j, total * sizeof(int) );
	cell_i.resize(total);
	//cudaMalloc(&cell_i, total * sizeof(int) );
	cell_j.resize(total);
	//cudaMalloc(&cell_j, total * sizeof(int) ); 

	auto &drvs = *drivers;
	Cylinder &driver = drvs.get_typed_driver<Cylinder>(0);
	
        dim3 BlockSize(block_x, block_y, 1);

        //printf("CELLS: %d\n", cell_id.size());
        
        /*for(int i = 0; i < cell_nb_nbh.size(); i++)
        {
        	printf("cell_nb_nbh :%d\n", cell_nb_nbh[i]);
        }*/
        
        kernelDriver<<<cell_id.size(), 1024>>>( cells, cell_id.data(), nb_particles_start.data(), *rcut_inc, driver, interaction_driver.data(), p_particle.data(), cell_particle.data() );
        
        //int cells_max = 0;

    
	  kernelUN<<<cell_id.size(), BlockSize>>>( cells, ghost_cell.data(), cell_id.data(), cell_neighbors_ids.data(), cell_start.data(), *nbh_dist_lab, domain->xform(), *rcut_inc, nb_nbh.data(), p_i.data(), p_j.data(), cell_i.data(), cell_j.data(), cell_nb_nbh.data() );
	  
	  cudaDeviceSynchronize();

		//kernelUN_2<<<numBlocks, 1024>>>(cells, dims, ghost_cell.data(), cell_id.data(), nb_particles_start.data(), loc_max_gap, *nbh_dist_lab, domain->xform(), *rcut_inc, id_i.data(), id_j.data(), cell_i.data(), cell_j.data(), p_i.data(), p_j.data(), nb_nbh.data(), id_particle.data(), cell_particle.data(), p_particle.data(), nombre_voisins_potentiels, driver, interaction_driver.data());
	
	//cudaDeviceSynchronize();

	onika::memory::CudaMMVector<int> nb_nbh_incr;
	//int* nb_nbh_incr;
	nb_nbh_incr.resize(nb_nbh.size());
	//cudaMalloc(&nb_nbh_incr, total * sizeof(int) );
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), total);
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), total);
	
	cudaFree(d_temp_storage);
	
	/*int nb_nbh_incr_final;
	cudaMemcpy( &nb_nbh_incr_final, nb_nbh_incr + total - 1, sizeof(int), cudaMemcpyDeviceToHost);
	int nb_nbh_final;
	cudaMemcpy( &nb_nbh_final, nb_nbh + total - 1, sizeof(int), cudaMemcpyDeviceToHost);*/
	
	//int total_interactions = nb_nbh_incr_final + nb_nbh_final;
	
	int total_interactions = nb_nbh_incr[nb_nbh.size() - 1] + nb_nbh[nb_nbh.size() - 1];
	
	onika::memory::CudaMMVector<int> driver_incr;
	//int* driver_incr;
	driver_incr.resize(interaction_driver.size());
	//cudaMalloc(&driver_incr, total_1 * sizeof(int) );
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, interaction_driver.data(), driver_incr.data(), total_1);
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, interaction_driver.data(), driver_incr.data(), total_1);
	
	cudaFree(d_temp_storage);
	
	/*int driver_incr_final;
	cudaMemcpy( &driver_incr_final, driver_incr + total_1 - 1, sizeof(int), cudaMemcpyDeviceToHost);
	int interaction_driver_final;
	cudaMemcpy( &interaction_driver_final, interaction_driver + total_1 - 1, sizeof(int), cudaMemcpyDeviceToHost);*/
	
	int total_interactions_driver = driver_incr[interaction_driver.size() - 1] + interaction_driver[interaction_driver.size() - 1];
	
	//int total_interactions_driver = driver_incr_final + interaction_driver_final;
	
	//printf("CHUNK END\n");
	
	//CLASSIFIER
	
	//printf("CLASSIFIER\n");
	
	//if (!ic.has_value())
        //	ic->initialize();

	//auto& type0 = ic->get_wave(0);
	//auto& type4 = ic->get_wave(4);
	
	//type0.clear();
	/*cudaFree(type0.ft_x);
	cudaFree(type0.ft_y);
	cudaFree(type0.ft_z);
	cudaFree(type0.mom_x);
	cudaFree(type0.mom_y);
	cudaFree(type0.mom_z);
	cudaFree(type0.id_i);
	cudaFree(type0.id_j);
	cudaFree(type0.cell_i);
	cudaFree(type0.cell_j);
	cudaFree(type0.p_i);
	cudaFree(type0.p_j);*/
	//type4.clear();
	/*cudaFree(type4.ft_x);
	cudaFree(type4.ft_y);
	cudaFree(type4.ft_z);
	cudaFree(type4.mom_x);
	cudaFree(type4.mom_y);
	cudaFree(type4.mom_z);
	cudaFree(type4.id_i);
	cudaFree(type4.id_j);
	cudaFree(type4.cell_i);
	//cudaFree(type0.cell_j);
	cudaFree(type0.p_i);
	//cudaFree(type0.p_j);*/
	
	/*type0.type = 0;
	type0.size_soa = total;
	type4.type = 4;
	type4.size_soa = total_1;
	
	//type0.resize(total_interactions);
	cudaMalloc(&type0.ft_x, total * sizeof(double));
	cudaMalloc(&type0.ft_y, total * sizeof(double));
	cudaMalloc(&type0.ft_z, total * sizeof(double));
	cudaMalloc(&type0.mom_x, total * sizeof(double));
	cudaMalloc(&type0.mom_y, total * sizeof(double));
	cudaMalloc(&type0.mom_z, total * sizeof(double));
	cudaMalloc(&type0.id_i, total * sizeof(uint64_t));
	cudaMalloc(&type0.id_j, total * sizeof(uint64_t));
	cudaMalloc(&type0.cell_i, total * sizeof(uint32_t));
	cudaMalloc(&type0.cell_j, total * sizeof(uint32_t));
	cudaMalloc(&type0.p_i, total * sizeof(uint16_t));
	cudaMalloc(&type0.p_j, total * sizeof(uint16_t));
	//type4.resize(total_interactions_driver);
	cudaMalloc(&type4.ft_x, total * sizeof(double));
	cudaMalloc(&type4.ft_y, total * sizeof(double));
	cudaMalloc(&type4.ft_z, total * sizeof(double));
	cudaMalloc(&type4.mom_x, total * sizeof(double));
	cudaMalloc(&type4.mom_y, total * sizeof(double));
	cudaMalloc(&type4.mom_z, total * sizeof(double));
	cudaMalloc(&type4.id_i, total * sizeof(uint64_t));
	cudaMalloc(&type4.id_j, total * sizeof(uint64_t));
	cudaMalloc(&type4.cell_i, total * sizeof(uint32_t));
	//cudaFree(type0.cell_j);
	cudaMalloc(&type4.p_i, total * sizeof(uint16_t));
	//cudaFree(type0.p_j);*/
	
	
	onika::memory::CudaMMVector<uint64_t> id_i_final;// = type0.id_i;
	//uint64_t* id_i_final;// = type0.id_i; 
	onika::memory::CudaMMVector<uint64_t> id_j_final;// = type0.id_j;
	//uint64_t* id_j_final;// = type0.id_j;
	onika::memory::CudaMMVector<uint32_t> cell_i_final;// = type0.cell_i;
	//uint32_t* cell_i_final;// = type0.cell_i;
	onika::memory::CudaMMVector<uint32_t> cell_j_final;// = type0.cell_j;
	//uint32_t* cell_j_final;// = type0.cell_j;
	onika::memory::CudaMMVector<uint16_t> p_i_final;// = type0.p_i;
	//uint16_t* p_i_final;// = type0.p_i;
	onika::memory::CudaMMVector<uint16_t> p_j_final;// = type0.p_j;
	//uint16_t* p_j_final;// = type0.p_j;
	onika::memory::CudaMMVector<double> ftx_final;// = type0.ft_x;
	//double* ftx_final;// = type0.ft_x;
	onika::memory::CudaMMVector<double> fty_final;// = type0.ft_y;
	//double* fty_final;// = type0.ft_y;
	onika::memory::CudaMMVector<double> ftz_final;// = type0.ft_z;
	//double* ftz_final;// = type0.ft_z;
	onika::memory::CudaMMVector<double> momx_final;// = type0.mom_x;
	//double* momx_final;// = type0.mom_x;
	onika::memory::CudaMMVector<double> momy_final;// = type0.mom_y;
	//double* momy_final;// = type0.mom_y;
	onika::memory::CudaMMVector<double> momz_final;// = type0.mom_z;
	//double* momz_final;// = type0.mom_z;
	
	//cudaMalloc(&ftx_final, total * sizeof(double));
	ftx_final.resize(total);
	//cudaMalloc(&fty_final, total * sizeof(double));
	fty_final.resize(total);
	//cudaMalloc(&ftz_final, total * sizeof(double));
	ftz_final.resize(total);
	//cudaMalloc(&momx_final, total * sizeof(double));
	momx_final.resize(total);
	//cudaMalloc(&momy_final, total * sizeof(double));
	momy_final.resize(total);
	//cudaMalloc(&momz_final, total * sizeof(double));
	momz_final.resize(total);
	//cudaMalloc(&id_i_final, total * sizeof(uint64_t));
	id_i_final.resize(total);
	//cudaMalloc(&id_j_final, total * sizeof(uint64_t));
	id_j_final.resize(total);
	//cudaMalloc(&cell_i_final, total * sizeof(uint32_t));
	cell_i_final.resize(total);
	//cudaMalloc(&cell_j_final, total * sizeof(uint32_t));
	cell_j_final.resize(total);
	//cudaMalloc(&p_i_final, total * sizeof(uint16_t));
	p_i_final.resize(total);
	//cudaMalloc(&p_j_final, total * sizeof(uint16_t));
	p_j_final.resize(total);	
	
	onika::memory::CudaMMVector<uint64_t> id_i_driver;// = type4.id_i;
	//uint64_t* id_i_driver;// = type4.id_i;
	//onika::memory::CudaMMVector<uint64_t> &id_j_driver = type4.id_j;
	//uint64_t* id_j_driver = type4.id_j;
	onika::memory::CudaMMVector<uint32_t> cell_i_driver;// = type4.cell_i;
	//uint32_t* cell_i_driver;// = type4.cell_i;
	onika::memory::CudaMMVector<uint16_t> p_i_driver;// = type4.p_i;
	//uint16_t* p_i_driver;// = type4.p_i;
	//onika::memory::CudaMMVector<double> &ftx_driver = type4.ft_x;
	//double* ftx_driver = type4.ft_x;
	//onika::memory::CudaMMVector<double> &fty_driver = type4.ft_y;
	//double* fty_driver = type4.ft_y;
	//onika::memory::CudaMMVector<double> &ftz_driver = type4.ft_z;
	//double* ftz_driver = type4.ft_z;
	//onika::memory::CudaMMVector<double> &momx_driver = type4.mom_x;
	//double* momx_driver = type4.mom_x;
	//onika::memory::CudaMMVector<double> &momy_driver = type4.mom_y;
	//double* momy_driver = type4.mom_y;
	//onika::memory::CudaMMVector<double> &momz_driver = type4.mom_z;
	//double* momz_driver = type4.mom_z;
	
	//cudaMalloc(&id_i_driver, total_1 * sizeof(uint64_t));
	id_i_driver.resize(total_1);
	//cudaMalloc(&cell_i_driver, total_1 * sizeof(uint32_t));
	cell_i_driver.resize(total_1);
	//cudaMalloc(&p_i_driver, total_1 * sizeof(uint16_t));
	p_i_driver.resize(total_1);

	numBlocks = (total + 256 - 1) / 256;
	
	kernelDEUX<<<numBlocks, 256>>>(cells, nb_nbh.data(), nb_nbh_incr.data(), id_i_final.data(), id_j_final.data(), cell_i_final.data(), cell_j_final.data(), p_i_final.data(), p_j_final.data(), ftx_final.data(), fty_final.data(), ftz_final.data(), momx_final.data(), momy_final.data(), momz_final.data(), p_i.data(), p_j.data(), cell_i.data(), cell_j.data(), total);
	
	cudaDeviceSynchronize();
	
	numBlocks = (total_1 + 256 - 1) / 256;
	
	kernelTROIS<<<numBlocks, 256>>>( cells, /*cell_id.data(), nb_particles_start.data(),*/ interaction_driver.data(), driver_incr.data(), id_i_driver.data(), cell_i_driver.data(), p_i_driver.data(), p_particle.data(), cell_particle.data(), total_1 );
	
	cudaDeviceSynchronize();

	//printf("CLASSIFIER END\n");*/	
	

	//cudaFree(interaction_driver);
	//cudaFree(nb_nbh);

	
	//cudaFree(nb_nbh_incr);
	//cudaFree(driver_incr);
	
	//cudaFree(p_i);
	//cudaFree(p_j);
	//cudaFree(cell_i);
	//cudaFree(cell_j);

	//cudaFree(p_particle);
	//cudaFree(cell_particle);
	
	/*cudaFree(ftx_final);
	cudaFree(fty_final);
	cudaFree(ftz_final);
	cudaFree(momx_final);
	cudaFree(momy_final);
	cudaFree(momz_final);
	cudaFree(id_i_final);
	cudaFree(id_j_final);
	cudaFree(cell_i_final);
	cudaFree(cell_j_final);
	cudaFree(p_i_final);
	cudaFree(p_j_final);	
	
	cudaFree(id_i_driver);
	cudaFree(cell_i_driver);
	cudaFree(p_i_driver);*/

	
	//printf("CHUNK_END\n");
	
	//auto &olds = *ic_olds;
	
    }
  };
  

  // === register factories ===
  ONIKA_AUTORUN_INIT(chunk_neighbors_contact) { OperatorNodeFactory::instance()->register_factory("chunk_neighbors_contact", make_grid_variant_operator<ChunkNeighborsContact>); }

} // namespace exaDEM
