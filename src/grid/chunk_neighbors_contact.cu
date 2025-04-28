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
  
  //kernelUN<<<cellsa.size(), BlockSize>>>( cells, dims, cellsa.data(), cellsb.data(), ghost_cells.data(), *nbh_dist_lab, domain->xform(), *rcut_inc, nb_nbh.data(), res.data(), m_origin, m_offset, m_cell_size );

template< class GridT > __global__ void kernelUN(GridT* cells,
			      IJK dims,
                              int* cellsa,
                              int* cellsb,
                              int* ghost_cell,
                              const double dist_lab, 
                              Mat3d xform, 
                              double rcut_inc,
                              int* nb_nbh,
                              int* res,
                              Vec3d origin,
                              IJK offset,
                             double cell_size)
{
	using BlockReduce = cub::BlockReduce<int, 32, cub::BLOCK_REDUCE_RAKING, 32>;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	
	int cell_a = cellsa[blockIdx.x];
	int num_particles_a = cells[cell_a].size();
	auto* __restrict__ rx_a = cells[cell_a][field::rx];
	auto* __restrict__ ry_a = cells[cell_a][field::ry];
	auto* __restrict__ rz_a = cells[cell_a][field::rz];
	auto* __restrict__ id_a = cells[cell_a][field::id];
	auto* __restrict__ rad_a = cells[cell_a][field::radius];
	
	int cell_b = cellsb[blockIdx.x];
	int num_particles_b = cells[cell_b].size();
	auto* __restrict__ rx_b = cells[cell_b][field::rx];
	auto* __restrict__ ry_b = cells[cell_b][field::ry];
	auto* __restrict__ rz_b = cells[cell_b][field::rz];
	auto* __restrict__ id_b = cells[cell_b][field::id];
	auto* __restrict__ rad_b = cells[cell_b][field::radius];
	IJK loc_b = grid_index_to_ijk( dims, cell_b );
	AABB cellb_AABB_ = AABB{ (origin+((offset+loc_b)*cell_size)), (origin+((offset+loc_b+1)*cell_size))};
	AABB cellb_AABB = enlarge( cellb_AABB_, rcut_inc + 0.5);
	
	double rcut2 = dist_lab * dist_lab;
	int nb_interactions = 0;
	
	bool is_ghost = ghost_cell[blockIdx.x];
	
	for(int p_a = threadIdx.y; p_a < num_particles_a; p_a+= blockDim.y)
	{
		const Vec3d pos_a = { rx_a[p_a] , ry_a[p_a] , rz_a[p_a] };
		
		int ida = id_a[p_a];
		
		if( is_inside( cellb_AABB , pos_a ) )
		{
			for(int p_b = threadIdx.x; p_b < num_particles_b; p_b+= blockDim.x)
			{
				const Vec3d pos_b = { rx_b[p_b], ry_b[p_b], rz_b[p_b] };
				const Vec3d dr = pos_a - pos_b;
				
				double d2 = norm2( xform * dr );
				
				if( nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell_a, p_a, cell_b, p_b))
				{
					if(ida < id_b[p_b] || is_ghost)
					{
						nb_interactions++;
					}
				}
			}	
		}
	}
	
	//atomicAdd(&res[0], nb_interactions);
	
	int aggregate = BlockReduce(temp_storage).Sum(nb_interactions);
	__syncthreads();
	if(threadIdx.x == 0 && threadIdx.y == 0) nb_nbh[blockIdx.x] = aggregate;
}

template < class GridT > __global__ void kernelDriver( GridT* cells,
							IJK dims,
							int* cells_a,
							double rcut_inc,
							Cylinder driver,
							int* interaction_driver)
{
	using BlockReduce = cub::BlockReduce<int, 32, cub::BLOCK_REDUCE_RAKING, 32>;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	
	int cell_a = cells_a[blockIdx.x];
	auto* __restrict__ rx_a = cells[cell_a][field::rx];
	auto* __restrict__ ry_a = cells[cell_a][field::ry];
	auto* __restrict__ rz_a = cells[cell_a][field::rz];
	auto* __restrict__ rad_a = cells[cell_a][field::radius];
	
	int local_id = threadIdx.y * blockDim.x + threadIdx.x;
	int total_threads = blockDim.x * blockDim.y;
	
	int nb_interactions = 0;
	
	for(int p_a = local_id; p_a < cells[cell_a].size(); p_a+= total_threads)
	{
		const double rVerletMax = rad_a[p_a] + rcut_inc;
		
		const Vec3d r = {rx_a[p_a], ry_a[p_a], rz_a[p_a]};
		
		if(driver.filter(rVerletMax, r))
		{
			nb_interactions++;
		}
	}
	
	int aggregate = BlockReduce(temp_storage).Sum(nb_interactions);
	__syncthreads();
	if(threadIdx.x == 0 && threadIdx.y == 0) interaction_driver[blockIdx.x] = aggregate;
	
}

template< class GridT > __global__ void kernelDEUX(GridT* cells,
			      IJK dims,
                              int* cellsa,
                              int* cellsb,
                              int* ghost_cell,
                              const double dist_lab, 
                              Mat3d xform, 
                              double rcut_inc,
                              int* nb_nbh_incr,
                              int* res,
                              Vec3d origin,
                              IJK offset,
                             double cell_size,
                             uint64_t* id_i,
                             uint64_t* id_j,
                             uint32_t* cell_i,
                             uint32_t* cell_j,
                             uint16_t* p_i,
                             uint16_t* p_j,
                             uint64_t* keys,
                             int* indices,
                             int min,
                             int max)
{
	using BlockScan = cub::BlockScan<int, 32, cub::BLOCK_SCAN_RAKING, 32>;
	 __shared__ typename BlockScan::TempStorage temp_storage;
	
	int cell_a = cellsa[blockIdx.x];
	int num_particles_a = cells[cell_a].size();
	auto* __restrict__ rx_a = cells[cell_a][field::rx];
	auto* __restrict__ ry_a = cells[cell_a][field::ry];
	auto* __restrict__ rz_a = cells[cell_a][field::rz];
	auto* __restrict__ id_a = cells[cell_a][field::id];
	auto* __restrict__ rad_a = cells[cell_a][field::radius];
	
	int cell_b = cellsb[blockIdx.x];
	int num_particles_b = cells[cell_b].size();
	auto* __restrict__ rx_b = cells[cell_b][field::rx];
	auto* __restrict__ ry_b = cells[cell_b][field::ry];
	auto* __restrict__ rz_b = cells[cell_b][field::rz];
	auto* __restrict__ id_b = cells[cell_b][field::id];
	auto* __restrict__ rad_b = cells[cell_b][field::radius];
	IJK loc_b = grid_index_to_ijk( dims, cell_b );
	AABB cellb_AABB_ = AABB{ (origin+((offset+loc_b)*cell_size)), (origin+((offset+loc_b+1)*cell_size))};
	AABB cellb_AABB = enlarge( cellb_AABB_, rcut_inc + 0.5);
	
	double rcut2 = dist_lab * dist_lab;
	int nb_interactions = 0;
	int prefix = 0;
	
	int range = max - min + 1;
	
	bool is_ghost = ghost_cell[blockIdx.x];
	
	for(int p_a = threadIdx.y; p_a < num_particles_a; p_a+= blockDim.y)
	{
		const Vec3d pos_a = { rx_a[p_a] , ry_a[p_a] , rz_a[p_a] };
		
		int ida = id_a[p_a];
		
		if( is_inside( cellb_AABB , pos_a ) )
		{
			for(int p_b = threadIdx.x; p_b < num_particles_b; p_b+= blockDim.x)
			{
				const Vec3d pos_b = { rx_b[p_b], ry_b[p_b], rz_b[p_b] };
				const Vec3d dr = pos_a - pos_b;
				
				double d2 = norm2( xform * dr );
				
				if( nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell_a, p_a, cell_b, p_b))
				{
					if(ida < id_b[p_b] || is_ghost)
					{
						nb_interactions++;
					}
				}
			}	
		}
	}

        BlockScan(temp_storage).ExclusiveSum( nb_interactions , prefix );
        __syncthreads();
        prefix+= nb_nbh_incr[blockIdx.x];
        
        int nb2 = 0;
    
	for(int p_a = threadIdx.y; p_a < num_particles_a; p_a+= blockDim.y)
	{
		const Vec3d pos_a = { rx_a[p_a] , ry_a[p_a] , rz_a[p_a] };
		
		int ida = id_a[p_a];
		
		if( is_inside( cellb_AABB , pos_a ) )
		{
			for(int p_b = threadIdx.x; p_b < num_particles_b; p_b+= blockDim.x)
			{
				const Vec3d pos_b = { rx_b[p_b], ry_b[p_b], rz_b[p_b] };
				const Vec3d dr = pos_a - pos_b;
				
				double d2 = norm2( xform * dr );
				
				if( nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell_a, p_a, cell_b, p_b))
				{
					if(ida < id_b[p_b] || is_ghost)
					{
						id_i[prefix + nb2] = ida;
						id_j[prefix + nb2] = id_b[p_b];
						cell_i[prefix + nb2] = cell_a;
						cell_j[prefix + nb2] = cell_b;
						p_i[prefix + nb2] = p_a;
						p_j[prefix + nb2] = p_b;
						keys[prefix + nb2] = (ida - min) * range + (id_b[p_b] - min);
						indices[prefix + nb2] = prefix + nb2;
						
						nb2++;
					}
				}
			}	
		}
	}   
}

template < class GridT > __global__ void kernelTROIS( GridT* cells,
							IJK dims,
							int* cells_a,
							double rcut_inc,
							Cylinder driver,
							int* interaction_driver_incr,
							uint64_t* id_driver,
							uint32_t* cell_driver,
							uint16_t* p_driver,
							int* indices,
							int min,
							int max)
{
	using BlockScan = cub::BlockScan<int, 32, cub::BLOCK_SCAN_RAKING, 32>;
	__shared__ typename BlockScan::TempStorage temp_storage;
	
	int cell_a = cells_a[blockIdx.x];
	auto* __restrict__ rx_a = cells[cell_a][field::rx];
	auto* __restrict__ ry_a = cells[cell_a][field::ry];
	auto* __restrict__ rz_a = cells[cell_a][field::rz];
	auto* __restrict__ rad_a = cells[cell_a][field::radius];
	auto* __restrict__ id_a = cells[cell_a][field::id];
	
	int local_id = threadIdx.y * blockDim.x + threadIdx.x;
	int total_threads = blockDim.x * blockDim.y;
	
	int nb_interactions = 0;
	int prefix = 0;
	
	int range = max - min + 1;
	
	for(int p_a = local_id; p_a < cells[cell_a].size(); p_a+= total_threads)
	{
		const double rVerletMax = rad_a[p_a] + rcut_inc;
		
		const Vec3d r = {rx_a[p_a], ry_a[p_a], rz_a[p_a]};
		
		if(driver.filter(rVerletMax, r))
		{
			nb_interactions++;
		}
	}
	
	BlockScan(temp_storage).ExclusiveSum( nb_interactions, prefix );
	__syncthreads();
	prefix+= interaction_driver_incr[blockIdx.x];
	
	int nb2 = 0;
	
	for(int p_a = local_id; p_a < cells[cell_a].size(); p_a+= total_threads)
	{
		const double rVerletMax = rad_a[p_a] + rcut_inc;
		
		const Vec3d r = {rx_a[p_a], ry_a[p_a], rz_a[p_a]};
		
		if(driver.filter(rVerletMax, r))
		{
			id_driver[prefix + nb2] = id_a[p_a];
			cell_driver[prefix + nb2] = cell_a;
			p_driver[prefix + nb2] = p_a;
			indices[prefix + nb2] = prefix + nb2;
			
			nb2++;
		}
	}	
	
}

   __global__ void filterUN( double* ft_x,
  			double* ft_y,
  			double* ft_z,
  			double* mom_x,
  			double* mom_y,
  			double* mom_z,
  			size_t size,
  			int* filter)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		if(ft_x[idx]!=0 || ft_y[idx]!=0 || ft_z[idx]!=0 || mom_x[idx]!=0 || mom_y[idx]!=0 || mom_z[idx]!=0)
  		{
  			filter[idx] = 1;
  		}
  	}
  }
  
  //filterDEUX<<<numBlocks, BlockSize>>>( interactions.id_i, interactions_id_j, interactions.ft_x, interactions.ft_y, interactions.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, keys, indices, filter_0_incr.data(), min, max, type, size);
  
  __global__ void filterDEUX( uint64_t* id_i,
  				uint64_t* id_j,
  				double* ft_x,
  				double* ft_y,
  				double* ft_z,
  				double* mom_x,
  				double* mom_y,
  				double* mom_z,
  				uint64_t* keys,
  				int* indices,
  				int* filter_incr,
  				int min,
  				int max,
  				int type,
  				size_t size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		if(ft_x[idx]!=0 || ft_y[idx]!=0 || ft_z[idx]!=0 || mom_x[idx]!=0 || mom_y[idx]!=0 || mom_z[idx]!=0)
  		{
  			int incr = filter_incr[idx];
  			
  			int range = max - min + 1;
  			
  			if(type == 0) keys[incr] = (id_i[idx] - min) * range + (id_j[idx] - min);
  			if(type == 4) keys[incr] = id_i[idx];
  			
  			indices[incr] = idx;
  		}
  	}
  }
  
  void sortWithIndices( uint64_t* keys, int* indices, uint64_t* keys_sorted, int* indices_sorted, int size)
  {
  	void *d_temp_storage = nullptr;
  	size_t temp_storage_bytes = 0;
  	
  	cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, keys, keys_sorted, indices, indices_sorted, size);
  	
  	cudaMalloc(&d_temp_storage, temp_storage_bytes);
  	
  	cub::DeviceRadixSort::SortPairs( d_temp_storage, temp_storage_bytes, keys, keys_sorted, indices, indices_sorted, size);
  	
  	cudaFree(d_temp_storage);
  }
  
  __global__ void find_common_elements(const uint64_t* keys, const uint64_t* keys_old, size_t size1, size_t size2, double* ftx, double* fty, double* ftz, double* ftx_old, double* fty_old, double* ftz_old, double* momx, double* momy, double* momz, double* momx_old, double* momy_old, double* momz_old, int* indices, int* indices_old)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if (idx < size1)
  	{
  		uint64_t key = keys[idx];
  		int low = 0, high = size2 - 1;
  		while(low <= high)
  		{
  			int mid = low + (high - low) / 2;
  			if( keys_old[mid] == key)
  			{
  				ftx[indices[idx]] = ftx_old[indices_old[mid]];
  				fty[indices[idx]] = fty_old[indices_old[mid]];
  				ftz[indices[idx]] = ftz_old[indices_old[mid]];
  				
  				momx[indices[idx]] = momx_old[indices_old[mid]];
  				momy[indices[idx]] = momy_old[indices_old[mid]];
  				momz[indices[idx]] = momz_old[indices_old[mid]];
  				return;
  			}
  			else if( keys_old[mid] < key)
  			{
  				low = mid + 1;
  			}
  			else
  			{
  				high = mid - 1;
  			}
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
      
      auto& c = *ic;
      
      uint64_t* keys_0;
      int* indices_0;
      size_t size0 = 0;
      
      uint64_t* keys_4;
      int* indices_4;
      size_t size4 = 0;
      
      //printf("UNCLASSIFY\n");
      
      /*if (ic.has_value())
      {
	   auto [data, size] = c.get_info(0);
	   
	   if(size > 0)
	   {
	   	size0 = size;
	   
	   	InteractionWrapper<InteractionSOA> interactions(data);
	   	
	   	int BlockSize = 256;
	   	int numBlocks = (size + BlockSize - 1) / BlockSize;
	   	
	   	//int* filter_0;
	   	onika::memory::CudaMMVector<int> filter_0;
	   	//cudaMalloc(&filter_0, size * sizeof(int));
	   	filter_0.resize(size);
	   	
	   	filterUN<<<numBlocks, BlockSize>>>( interactions.ft_x, interactions.ft_y, interactions.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, size, filter_0.data());
	   	
	   	onika::memory::CudaMMVector<int> filter_0_incr;
	   	filter_0_incr.resize(size);
	   	
       		void* d_temp_storage = nullptr;
		size_t temp_storage_bytes = 0;
	
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, filter_0.data(), filter_0_incr.data(), size);
	
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, filter_0.data(), filter_0_incr.data(), size);
	
		cudaFree(d_temp_storage);
		
		int filter_0_total = filter_0[size - 1] + filter_0_incr[size - 1];

		uint64_t* keys;
		cudaMalloc(&keys, filter_0_total * sizeof(uint64_t));
		
		int* indices;
		cudaMalloc(&indices, filter_0_total * sizeof(int));
		
		int min = 0;
		int max = grid->number_of_particles() - 1;

		filterDEUX<<<numBlocks, BlockSize>>>( interactions.id_i, interactions.id_j, interactions.ft_x, interactions.ft_y, interactions.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, keys, indices, filter_0_incr.data(), min, max, 0, size);
		
		cudaDeviceSynchronize();
		
		cudaMalloc(&keys_0, filter_0_total * sizeof(uint64_t));
		cudaMalloc(&indices_0, filter_0_total * sizeof(int));
		
		sortWithIndices( keys, indices, keys_0, indices_0, filter_0_total);
		
		cudaFree(keys);
		cudaFree(indices);
		
	   }
	   
	   auto [data2, size2] = c.get_info(4);
	   
	   if(size2 > 0)
	   {
	   	size4 = size2;
	   	
	   	InteractionWrapper<InteractionSOA> interactions(data2);
	   	
	   	int BlockSize = 256;
	   	int numBlocks = (size2 + BlockSize - 1) / BlockSize;
	   	
	   	//int* filter_0;
	   	onika::memory::CudaMMVector<int> filter_4;
	   	//cudaMalloc(&filter_0, size * sizeof(int));
	   	filter_4.resize(size2);
	   	
	   	filterUN<<<numBlocks, BlockSize>>>( interactions.ft_x, interactions.ft_y, interactions.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, size2, filter_4.data());
	   	
	   	onika::memory::CudaMMVector<int> filter_4_incr;
	   	filter_4_incr.resize(size2);
	   	
       		void* d_temp_storage = nullptr;
		size_t temp_storage_bytes = 0;
	
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, filter_4.data(), filter_4_incr.data(), size2);
	
		cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
		cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, filter_4.data(), filter_4_incr.data(), size2);
	
		cudaFree(d_temp_storage);
		
		int filter_4_total = filter_4[size2 - 1] + filter_4_incr[size2 - 1];
		
		//cudaFree(filter_4);
		
		uint64_t* keys;
		cudaMalloc(&keys, filter_4_total * sizeof(uint64_t));
		
		int* indices;
		cudaMalloc(&indices, filter_4_total * sizeof(int));
		
		int min = 0;
		int max = grid->number_of_particles() - 1;

		filterDEUX<<<numBlocks, BlockSize>>>( interactions.id_i, interactions.id_j, interactions.ft_x, interactions.ft_y, interactions.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, keys, indices, filter_4_incr.data(), min, max, 4, size2);
		
		cudaDeviceSynchronize();
		
		cudaMalloc(&keys_4, filter_4_total * sizeof(uint64_t));
		cudaMalloc(&indices_4, filter_4_total * sizeof(int));
		
		sortWithIndices( keys, indices, keys_4, indices_4, filter_4_total);
		
		cudaFree(keys);
		cudaFree(indices);
		
	   }	    
      }*/
      
      //printf("UNCLASSIFY_END\n");
      
 	onika::memory::CudaMMVector<int> cellsb_ids;
 	
 	onika::memory::CudaMMVector<int> number_of_cells_neighbors;
 	
 	auto& g = *grid;
 	
 	IJK dims = g.dimension();
 	
 	auto& amr2 = *amr;
 	
 	const size_t* sub_grid_start = amr2.sub_grid_start().data();
 	const uint32_t* sub_grid_cells = amr2.sub_grid_cells().data();
 	
 	const size_t n_cells = g.number_of_cells();
 	
 	auto& amr_grid_pairs2 = *amr_grid_pairs;
 	const unsigned int loc_max_gap = amr_grid_pairs2.cell_layers();
 	
 	cellsb_ids.resize( n_cells * 27 );
 	number_of_cells_neighbors.resize( n_cells );
 	
 	unsigned int max_threads = omp_get_max_threads();
 	
#	pragma omp parallel
	{
		int tid = omp_get_thread_num();
		assert( tid>=0 && size_t(tid)<max_threads );
		
		GRID_OMP_FOR_BEGIN( dims, cell_a, loc_a, schedule(dynamic))
		{
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
				
				if(n_particles_b > 0)
				{
					cellsb_ids[cell_a*27 + nb] = cell_b;
					nb++;
				}
			}
			
			number_of_cells_neighbors[cell_a]  = nb;
		}
		GRID_OMP_FOR_END
	}
	
	onika::memory::CudaMMVector<int> cells_a;
	onika::memory::CudaMMVector<int> incr_cells_a;
	
	auto [cell_ptr, cell_size] = traversal_real->info();
	
	int incr_cell = 0;
	
	for(int i = 0; i < n_cells; i++)
	{
		if( cells[i].size() > 0 && is_in( i, cell_ptr, cell_size ) ){ cells_a.push_back(i); incr_cells_a.push_back( incr_cell ); incr_cell+= number_of_cells_neighbors[i]; } 
	}
	
	//onika::memory::CudaMMVector<int> cellsa;
	std::vector<int> cellsa;
	cellsa.resize(incr_cell);
	//onika::memory::CudaMMVector<int> cellsb;
	std::vector<int> cellsb;
	cellsb.resize(incr_cell);
	//onika::memory::CudaMMVector<int> ghost_cells;
	std::vector<int> ghost_cells;
	ghost_cells.resize(incr_cell);
	
#	pragma omp parallel for
	for(int i = 0; i < cells_a.size(); i++)
	{
		int index = cells_a[i];
		
		int nb_nbh = number_of_cells_neighbors[index];
		
		int incr = incr_cells_a[i];
		
		for(int j = 0; j < nb_nbh; j++)
		{
			cellsa[incr + j] = index;
			
			cellsb[incr + j] = cellsb_ids[index*27 + j];
			ghost_cells[incr + j] = g.is_ghost_cell( cellsb_ids[index*27 + j] );
		}
	}
	
	constexpr int block_x = 32;
	constexpr int block_y = 32;
	
	auto &drvs = *drivers;
	Cylinder &driver = drvs.get_typed_driver<Cylinder>(0);
	
	dim3 BlockSize(	block_x, block_y, 1);
	
	auto m_origin = g.origin();
	auto m_offset = g.offset();
	auto m_cell_size = g.cell_size();
	
	onika::memory::CudaMMVector<int> res;
	res.resize(1);
	
	onika::memory::CudaMMVector<int> nb_nbh;
	//int* nb_nbh;
	nb_nbh.resize( cellsa.size() );
	//cudaMalloc(&nb_nbh, cellsa.size() * sizeof(int));
	
	int* cellsa_GPU;
	cudaMalloc(&cellsa_GPU, cellsa.size() * sizeof(int));
	cudaMemcpy(cellsa_GPU, cellsa.data(), cellsa.size()*sizeof(int), cudaMemcpyHostToDevice);
	
	int* cellsb_GPU;
	cudaMalloc(&cellsb_GPU, cellsb.size()*sizeof(int));
	cudaMemcpy(cellsb_GPU, cellsb.data(), cellsb.size()*sizeof(int), cudaMemcpyHostToDevice);
	
	int* ghost_cells_GPU;
	cudaMalloc(&ghost_cells_GPU, ghost_cells.size() * sizeof(int) );
	cudaMemcpy(ghost_cells_GPU, ghost_cells.data(), ghost_cells.size() * sizeof(int), cudaMemcpyHostToDevice);
	
	kernelUN<<<cellsa.size(), BlockSize>>>( cells, dims, cellsa_GPU, cellsb_GPU, ghost_cells_GPU, *nbh_dist_lab, domain->xform(), *rcut_inc, nb_nbh.data(), res.data(), m_origin, m_offset, m_cell_size );
	
	int* cells_a_GPU;
	cudaMalloc(&cells_a_GPU, cells_a.size() * sizeof(int));
	cudaMemcpy(cells_a_GPU, cells_a.data(), cells_a.size() * sizeof(int), cudaMemcpyHostToDevice );
	
	onika::memory::CudaMMVector<int> interaction_driver;
	//int* interaction_driver;
	interaction_driver.resize(cells_a.size());
	//cudaMalloc(&interaction_driver, cells_a.size() * sizeof(int));
	
	kernelDriver<<<cells_a.size(), BlockSize>>>( cells, dims, cells_a_GPU, *rcut_inc, driver, interaction_driver.data());
	
	cudaDeviceSynchronize();
	
	printf("RES: %d\n", res[0]);
	
	onika::memory::CudaMMVector<int> nb_nbh_incr;
	//int* nb_nbh_incr;
	//cudaMalloc(&nb_nbh_incr, cellsa.size() * sizeof(int));
	nb_nbh_incr.resize(cellsa.size());

	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), cellsa.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), cellsa.size());
	
	cudaFree(d_temp_storage);
	
	/*int nb_nbh_end;
	cudaMemcpy(&nb_nbh_end, nb_nbh + cellsa.size() - 1, sizeof(int), cudaMemcpyHostToDevice);
	
	int nb_nbh_incr_end;
	cudaMemcpy(&nb_nbh_incr_end, nb_nbh_incr + cellsa.size() - 1, sizeof(int), cudaMemcpyHostToDevice);*/
	
	int total_interactions = nb_nbh[cellsa.size() - 1] + nb_nbh_incr[cellsa.size() - 1];
	
	onika::memory::CudaMMVector<int> interaction_driver_incr;
	interaction_driver_incr.resize(cells_a.size());
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, interaction_driver.data(), interaction_driver_incr.data(), cells_a.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, interaction_driver.data(), interaction_driver_incr.data(), cells_a.size());
	
	cudaFree(d_temp_storage);
	
	int total_interactions_driver = interaction_driver[cells_a.size() - 1] + interaction_driver_incr[cells_a.size() - 1];	
	
	auto& type0 = c.get_wave(0);
	auto& type4 = c.get_wave(4);
	
	type0.clear();
	
	onika::memory::CudaMMVector<uint64_t> &id_i = type0.id_i;
	//uint64_t* id_i;
	onika::memory::CudaMMVector<uint64_t> &id_j = type0.id_j;
	//uint64_t* id_j;
	onika::memory::CudaMMVector<uint32_t> &cell_i = type0.cell_i;
	//uint32_t* cell_i;
	onika::memory::CudaMMVector<uint32_t> &cell_j = type0.cell_j;
	//uint32_t* cell_j;
	onika::memory::CudaMMVector<uint16_t> &p_i = type0.p_i;
	//uint16_t* p_i;
	onika::memory::CudaMMVector<uint16_t> &p_j = type0.p_j;
	//uint16_t* p_j;
	
	uint64_t* keys_new0;
	cudaMalloc(&keys_new0, total_interactions * sizeof(uint64_t));
	int* indices_new0;
	cudaMalloc(&indices_new0, total_interactions * sizeof(int));
	
	uint64_t* keys_sorted0;
	cudaMalloc(&keys_sorted0, total_interactions * sizeof(uint64_t));
	int* indices_sorted0;
	cudaMalloc(&indices_sorted0, total_interactions * sizeof(int));
	
	onika::memory::CudaMMVector<uint64_t> &id_driver = type4.id_i;
	//uint64_t* id_driver;
	onika::memory::CudaMMVector<uint32_t> &cell_driver = type4.cell_i;
	//uint32_t* cell_driver;
	onika::memory::CudaMMVector<uint16_t> &p_driver = type4.p_i;
	//uint16_t* p_driver;
	
	//uint64_t* keys_new4;
	//cudaMalloc(&keys_new4, total_interactions_driver * sizeof(uint64_t));
	int* indices_new4;
	cudaMalloc(&indices_new4, total_interactions_driver * sizeof(int));
	
	uint64_t* keys_sorted4;
	cudaMalloc(&keys_sorted4, total_interactions_driver * sizeof(uint64_t));
	int* indices_sorted4;
	cudaMalloc(&indices_sorted4, total_interactions_driver * sizeof(int));
	
	id_driver.resize(total_interactions_driver);
	//cudaMalloc(&id_driver, total_interactions_driver * sizeof(uint64_t));
	cell_driver.resize(total_interactions_driver); 
	//cudaMalloc(&cell_driver, total_interactions_driver * sizeof(uint32_t));
	p_driver.resize(total_interactions_driver);
	//cudaMalloc(&p_driver, total_interactions_driver * sizeof(uint16_t));
	
	id_i.resize(total_interactions);
	//cudaMalloc(&id_i, total_interactions * sizeof(uint64_t));
	id_j.resize(total_interactions);
	//cudaMalloc(&id_j, total_interactions * sizeof(uint64_t));
	cell_i.resize(total_interactions);
	//cudaMalloc(&cell_i, total_interactions * sizeof(uint32_t));
	cell_j.resize(total_interactions);
	//cudaMalloc(&cell_j, total_interactions * sizeof(uint32_t));
	p_i.resize(total_interactions);
	//cudaMalloc(&p_i, total_interactions * sizeof(uint16_t));
	p_j.resize(total_interactions);
	//cudaMalloc(&p_j, total_interactions * sizeof(uint16_t));
	
	int min = 0;
	int max = grid->number_of_particles() - 1;
	
	kernelDEUX<<<cellsa.size(), BlockSize>>>( cells, dims, cellsa_GPU, cellsb_GPU, ghost_cells_GPU, *nbh_dist_lab, domain->xform(), *rcut_inc, nb_nbh_incr.data(), res.data(), m_origin, m_offset, m_cell_size, id_i.data(), id_j.data(), cell_i.data(), cell_j.data(), p_i.data(), p_j.data(), keys_new0, indices_new0, min, max );
	
	//kernelDriver<<<cells_a.size(), BlockSize>>>( cells, dims, cells_a_GPU, *rcut_inc, driver, interaction_driver.data());
	
	kernelTROIS<<<cells_a.size(), BlockSize>>>( cells, dims, cells_a_GPU, *rcut_inc, driver, interaction_driver_incr.data(), id_driver.data(), cell_driver.data(), p_driver.data(), indices_new4, min, max );
	
	cudaDeviceSynchronize();
	
	/*__global__ void find_common_elements(const uint64_t* keys, const uint64_t* keys_old, size_t size1, size_t size2, double* ftx, double* fty, double* ftz, double* ftx_old, double* fty_old, double* ftz_old, double* momx, double* momy, double* momz, double* momx_old, double* momy_old, double* momz_old, int* indices, int* indices_old)
	
	if(size0 > 0)
	{
		int numBlocks = (total_interactions + 256 - 1) / 256;
	
		sortWithIndices( keys_new0, indices_new0, keys_sorted0, indices_sorted0, total_interactions);
		
		find_common_elements<<<numBlocks, 256>>>( keys_sorted0, keys_0, total_interactions, size0, 
	}
	
	if(size4 > 0)
	{
		int numBlocks = (total_interactions_driver + 256 - 1) / 256;
		
		sortWithIndices( id_driver, indices_new4, keys_sorted4, indices_sorted4, total_interactions_driver);		
	}*/
	
	cudaFree(cellsa_GPU);
	cudaFree(cellsb_GPU);
	cudaFree(ghost_cells_GPU);
	//cudaFree(nb_nbh);
	//cudaFree(nb_nbh_incr);
	
	/*cudaFree(id_i);
	cudaFree(id_j);
	cudaFree(p_i);
	cudaFree(p_j);
	cudaFree(cell_i);
	cudaFree(cell_j);*/
	
	cudaFree(cells_a_GPU);
	/*cudaFree(id_driver);
	cudaFree(cell_driver);
	cudaFree(p_driver);*/
	
	//cudaFree(keys_0);
	//cudaFree(indices_0);
	
	//cudaFree(keys_4);
	//cudaFree(indices_4);
	
	cudaFree(keys_new0);
	cudaFree(indices_new0);
	
	cudaFree(indices_new4);
	
	cudaFree(keys_sorted0);
	cudaFree(indices_sorted0);
	
	cudaFree(keys_sorted4);
	cudaFree(indices_sorted4);
    }
  };
  

  // === register factories ===
  ONIKA_AUTORUN_INIT(chunk_neighbors_contact) { OperatorNodeFactory::instance()->register_factory("chunk_neighbors_contact", make_grid_variant_operator<ChunkNeighborsContact>); }

} // namespace exaDEM
