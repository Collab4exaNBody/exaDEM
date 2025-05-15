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
  
  struct InteractionOLD
  {
  	template <typename T> using VectorT = onika::memory::CudaMMVector<T>;
  	
  	VectorT<uint64_t> keys;
  	
  	VectorT<double> ft_x;
  	VectorT<double> ft_y;
  	VectorT<double> ft_z;
  	
  	VectorT<double> mom_x;
  	VectorT<double> mom_y;
  	VectorT<double> mom_z;
  	
  	VectorT<int> indices;
  	
  	size_t size = 0;
  	
  	void set(size_t s)
  	{
  		size = s;
  		
  		keys.clear();
  		keys.resize(s);
  		
  		ft_x.clear();
  		ft_y.clear();
  		ft_z.clear();
  		
  		ft_x.resize(s);
  		ft_y.resize(s);
  		ft_z.resize(s);
  		
  		mom_x.clear();
  		mom_y.clear();
  		mom_z.clear();
  		
  		mom_x.resize(s);
  		mom_y.resize(s);
  		mom_z.resize(s);
  		
  		indices.clear();
  		indices.resize(s);
  	}
  };
  
  struct InteractionOLDWrapper
  {
  	uint64_t * keys;
  	
  	double * ft_x;
  	double * ft_y;
  	double * ft_z;
  	
  	double * mom_x;
  	double * mom_y;
  	double * mom_z;
  	
  	int * indices;
  	
  	size_t size = 0;
  	
  	InteractionOLDWrapper(InteractionOLD& data)
  	{
  		keys = onika::cuda::vector_data(data.keys);
  		
  		ft_x = onika::cuda::vector_data(data.ft_x);
  		ft_y = onika::cuda::vector_data(data.ft_y);
  		ft_z = onika::cuda::vector_data(data.ft_z);
  		
  		mom_x = onika::cuda::vector_data(data.mom_x);
  		mom_y = onika::cuda::vector_data(data.mom_y);
  		mom_z = onika::cuda::vector_data(data.mom_z);
  		
  		indices = onika::cuda::vector_data(data.indices);
  		
  		size = data.size;
  	}

  };
  
  struct Unclassifier
  {
  	template <typename T> using VectorT = onika::memory::CudaMMVector<T>;
  	
  	static constexpr int types = 13;
  	std::vector<InteractionOLD> waves;
  	
  	bool use = false; 
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
                              Vec3d origin,
                              IJK offset,
                             double cell_size,
                             uint64_t* id_i,
                             uint64_t* id_j,
                             uint32_t* cell_i,
                             uint32_t* cell_j,
                             uint16_t* p_i,
                             uint16_t* p_j)
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
							uint16_t* p_driver)
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
			
			nb2++;
		}
	}	
	
}

   __global__ void filtre_un( double* ft_x,
  			double* ft_y,
  			double* ft_z,
  			double* mom_x,
  			double* mom_y,
  			double* mom_z,
  			int* filter,
  			size_t size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	int nb = 0;
  	
  	if(idx < size)
  	{
  		if(ft_x[idx]!=0 || ft_y[idx]!=0 || ft_z[idx]!=0 || mom_x[idx]!=0 || mom_y[idx]!=0 || mom_z[idx]!=0)
  		{
  			filter[idx] = 1;
  		}
  	}
  }
  
  __global__ void filtre_deux( uint64_t* id_i,
  				uint64_t* id_j,
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
  		}
  	}
  }
  
  __global__ void generateKeys( uint64_t* keys, 
  				const uint64_t* id_i, 
  				const uint64_t* id_j,
  				int* indices,
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
  		
  		indices[idx] = idx;
  	}
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

	    // Free temporary storage and double-buffered arrays*/
	    cudaFree(d_temp_storage);
	}
	
  __global__ void find_common_elements(const uint64_t* keys, const uint64_t* keys_old, int size1, int size2, double* ftx, double* fty, double* ftz, double* ftx_old, double* fty_old, double* ftz_old, double* momx, double* momy, double* momz, double* momx_old, double* momy_old, double* momz_old, int* indices, int* indices_old)
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
  
/*template<
    typename T, typename U, typename V
>
__global__ void find_common_elements(
    const uint64_t* __restrict__ keys,
    const uint64_t* __restrict__ keys_old,
    int size1, int size2,
    const double* __restrict__ ftx_old,
    const double* __restrict__ fty_old,
    const double* __restrict__ ftz_old,
    const double* __restrict__ momx_old,
    const double* __restrict__ momy_old,
    const double* __restrict__ momz_old,
    double* __restrict__ ftx,
    double* __restrict__ fty,
    double* __restrict__ ftz,
    double* __restrict__ momx,
    double* __restrict__ momy,
    double* __restrict__ momz,
    const int* __restrict__ indices_old,
    int* __restrict__ indices)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size1) return;

    // Lecture en read-only cache
    uint64_t key = __ldg(keys + idx);
    int out_i = indices[idx];

    int lo = 0, hi = size2 - 1;
    while (lo <= hi) {
        int mid = lo + ((hi - lo) >> 1);
        uint64_t ko = __ldg(keys_old + mid);
        if (ko == key) {
            int in_i = __ldg(indices_old + mid);
            // Chargement optimal des valeurs
            ftx[out_i]  = __ldg(ftx_old  + in_i);
            fty[out_i]  = __ldg(fty_old  + in_i);
            ftz[out_i]  = __ldg(ftz_old  + in_i);
            momx[out_i] = __ldg(momx_old + in_i);
            momy[out_i] = __ldg(momy_old + in_i);
            momz[out_i] = __ldg(momz_old + in_i);
            return;
        }
        else if (ko < key) lo = mid + 1;
        else            hi = mid - 1;
    }
}*/

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
    
    ADD_SLOT(Drivers, drivers, INPUT_OUTPUT, DocString{"List of Drivers"});
    
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});
    
    ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
    ADD_SLOT(Classifier2, ic2, INPUT_OUTPUT);
    
    ADD_SLOT(InteractionSOA, interaction_type0, INPUT_OUTPUT);
    ADD_SLOT(InteractionSOA, interaction_type4, INPUT_OUTPUT);

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
      
       //auto& c = *ic;
       auto& c = *ic2;
       
       Unclassifier unc;
       
       unc.waves.resize(13);
       
       //printf("UNC\n");
       
       if(c.use )
       {
       		for(int type = 0; type < 13; type++)
       		{
       			auto [/*data*/interactions, size] = c.get_info(type);
       			
       			if(size > 0)
       			{
       				//InteractionWrapper<InteractionSOA> interactions(data);
       				
       				int blockSize = 256;
       				int numBlocks = (size + blockSize - 1) / blockSize;
       				
       				onika::memory::CudaMMVector<int> filtre;
       				//int* filtre;
       				filtre.resize(size);
       				//cudaMalloc(&filtre, size * sizeof(int) );
       				
       				filtre_un<<<numBlocks, blockSize>>>( interactions.ft_x, interactions.ft_y, interactions.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, filtre.data(), size);
       				
       				cudaDeviceSynchronize();
       				
       				onika::memory::CudaMMVector<int> filtre_incr;
       				//int* filtre_incr;
				filtre_incr.resize(size);
				//cudaMalloc(&filtre_incr, size * sizeof(int) );

				void* d_temp_storage = nullptr;
				size_t temp_storage_bytes = 0;
	
				cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, filtre.data(), filtre_incr.data(), size);
	
				cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
				cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, filtre.data(), filtre_incr.data(), size);
	
				cudaFree(d_temp_storage);
				
				/*int filtre_end;
				cudaMemcpy(&filtre_end, filtre + size - 1, sizeof(int), cudaMemcpyHostToDevice);*/
				
				/*int filtre_incr_end;
				cudaMemcpy(&filtre_incr_end, filtre_incr + size - 1, sizeof(int), cudaMemcpyHostToDevice);*/
	
				int total = /*filtre_end + filtre_incr_end;*/filtre[size - 1] + filtre_incr[size - 1];
				
				//onika::memory::CudaMMVector<uint64_t> id_i;
				uint64_t* id_i;
				//id_i.resize(total);
				cudaMalloc(&id_i, total * sizeof(uint64_t) );
				
				//onika::memory::CudaMMVector<uint64_t> id_j;
				uint64_t* id_j;
				//id_j.resize(total);
				cudaMalloc(&id_j, total * sizeof(uint64_t) );
				
				//onika::memory::CudaMMVector<int> indices;
				int* indices;
				//indices.resize(total);
				cudaMalloc(&indices, total * sizeof(uint64_t) );
				
				//auto& old = unc.waves[type];
				auto& unc_type = unc.waves[type];
				
				/*auto* old_keys = old.keys;
				auto* old_ftx = old.ft_x;
				auto* old_fty = old.ft_y;
				auto* old_ftz = old.ft_z;
				auto* old_momx = old.mom_x;
				auto* old_momy = old.mom_y;
				auto* old_momz = old.mom_z;
				auto* old_indices = old.indices;*/
  		
  				/*cudaMalloc(&old.keys, total * sizeof(uint64_t) );
  				cudaMalloc(&old.ft_x, total * sizeof(double) );
  				cudaMalloc(&old.ft_y, total * sizeof(double) );
  				cudaMalloc(&old.ft_z, total * sizeof(double) );
  				cudaMalloc(&old.mom_x, total * sizeof(double) );
  				cudaMalloc(&old.mom_y, total * sizeof(double) );
  				cudaMalloc(&old.mom_z, total * sizeof(double) );
  				cudaMalloc(&old.indices, total * sizeof(int) );*/
  				
  				//old.size = total;
  				
  				unc_type.set(total);
				
				InteractionOLDWrapper old(unc_type);
       				
       				filtre_deux<<<numBlocks, blockSize>>>( interactions.id_i, interactions.id_j, id_i, id_j, interactions.ft_x, interactions.ft_y, interactions.ft_z, old.ft_x, old.ft_y, old.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, old.mom_x, old.mom_y, old.mom_z, filtre_incr.data(), size);
       				
       				cudaDeviceSynchronize();
       				
       				int min = 0;
       				int max = grid->number_of_particles() - 1;
       				
       				numBlocks = ( total + blockSize - 1 ) / blockSize;
       				
       				//onika::memory::CudaMMVector<uint64_t> keys;
       				uint64_t* keys;
       				//keys.resize(total);
       				cudaMalloc(&keys, total * sizeof(uint64_t) );
       				
       				generateKeys<<<numBlocks, blockSize>>>( keys, id_i, id_j, indices, min, max, type, total);
       				
       				sortWithIndices( keys, indices, old.keys, old.indices, total);
       				
       				//cudaFree(filtre);
       				//cudaFree(filtre_incr);*/
       				cudaFree(id_i);
       				cudaFree(id_j);
       				cudaFree(keys);
       				cudaFree(indices);
       				
       				/*cudaError_t err = cudaFree(interactions.ft_x);
       				err = cudaFree(interactions.ft_y);
       				err = cudaFree(interactions.ft_z);
       				
       				err = cudaFree(interactions.mom_x);
       				err = cudaFree(interactions.mom_y);
       				err = cudaFree(interactions.mom_z);
       				
       				err = cudaFree(interactions.id_i);
       				err = cudaFree(interactions.id_j);
       				
       				err = cudaFree(interactions.cell_i);
       				err = cudaFree(interactions.cell_j);
       				
       				err = cudaFree(interactions.p_i);
       				err = cudaFree(interactions.p_j);
       				
       				err = cudaFree(interactions.sub_i);
       				err = cudaFree(interactions.sub_j);
       				
       				interactions.ft_x = nullptr;
       				interactions.ft_y = nullptr;
       				interactions.ft_z = nullptr;
       				interactions.mom_x = nullptr;
       				interactions.mom_y = nullptr;
       				interactions.mom_z = nullptr;
       				interactions.id_i = nullptr;
       				interactions.id_j = nullptr;
       				interactions.cell_i = nullptr;
       				interactions.cell_j = nullptr;
       				interactions.p_i = nullptr;
       				interactions.p_j = nullptr;
       				interactions.sub_i = nullptr;
       				interactions.sub_j = nullptr;
       				
       				interactions.size2 = 0;*/
       			}
       		}
       		
       		//printf("UNCLASSIFY END\n");
       }
       
       
       //printf("UNC END\n");
      
 	std::vector<int> cellsb_ids;
 	
 	std::vector<int> number_of_cells_neighbors;
 	
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
	std::vector<int> incr_cells_a;
	
	auto [cell_ptr, cell_size] = traversal_real->info();
	
	int incr_cell = 0;
	
	for(int i = 0; i < n_cells; i++)
	{
		if( cells[i].size() > 0 && is_in( i, cell_ptr, cell_size ) ){ cells_a.push_back(i); incr_cells_a.push_back( incr_cell ); incr_cell+= number_of_cells_neighbors[i]; } 
	}
	
	onika::memory::CudaMMVector<int> cellsa;
	//int* cellsa = (int*) malloc(incr_cell * sizeof(int));
	cellsa.resize(incr_cell);
	
	onika::memory::CudaMMVector<int> cellsb;
	//std::vector<int> cellsb;
	cellsb.resize(incr_cell);
	//int* cellsb = (int*) malloc(incr_cell * sizeof(int));
	
	onika::memory::CudaMMVector<int> ghost_cells;
	//int* ghost_cells = (int*) malloc(incr_cell * sizeof(int));
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
	
	onika::memory::CudaMMVector<int> nb_nbh;
	//int* nb_nbh;
	nb_nbh.resize( incr_cell );
	//cudaMalloc(&nb_nbh, cellsa.size() * sizeof(int) );
	
	/*int* cellsa_GPU;
	cudaMalloc(&cellsa_GPU, incr_cell * sizeof(int) );
	cudaMemcpy(cellsa_GPU, cellsa, incr_cell * sizeof(int), cudaMemcpyHostToDevice );*/
	
	/*int* cellsb_GPU;
	cudaMalloc(&cellsb_GPU, incr_cell * sizeof(int) );
	cudaMemcpy(cellsb_GPU, cellsb.data(), incr_cell * sizeof(int), cudaMemcpyHostToDevice );*/
	
	/*int* ghost_cells_GPU;
	cudaMalloc(&ghost_cells_GPU, incr_cell * sizeof(int) );
	cudaMemcpy(ghost_cells_GPU, ghost_cells, incr_cell * sizeof(int), cudaMemcpyHostToDevice);*/
	
	kernelUN<<<incr_cell, BlockSize>>>( cells, dims, cellsa.data(), cellsb.data(), ghost_cells.data(), *nbh_dist_lab, domain->xform(), *rcut_inc, nb_nbh.data(), m_origin, m_offset, m_cell_size );
	
	onika::memory::CudaMMVector<int> interaction_driver;
	//int* interaction_driver;
	interaction_driver.resize(cells_a.size());
	//cudaMalloc(&interaction_driver, cells_a.size() * sizeof(int) );
	
	/*int* cells_a_GPU;
	cudaMalloc(&cells_a_GPU, cells_a.size() * sizeof(int) );
	cudaMemcpy(cells_a_GPU, cells_a.data(), cells_a.size() * sizeof(int), cudaMemcpyHostToDevice );*/
	
	kernelDriver<<<cells_a.size(), BlockSize>>>( cells, dims, cells_a.data(), *rcut_inc, driver, interaction_driver.data());
	
	cudaDeviceSynchronize();
	
	onika::memory::CudaMMVector<int> nb_nbh_incr;
	//int* nb_nbh_incr;
	nb_nbh_incr.resize( incr_cell );
	//cudaMalloc(&nb_nbh_incr, cellsa.size() * sizeof(int) );

	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), incr_cell );
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), incr_cell );
	
	cudaFree(d_temp_storage);
	
	/*int nb_nbh_incr_final;
	cudaMemcpy( &nb_nbh_incr_final, nb_nbh_incr + cellsa.size() - 1, sizeof(int), cudaMemcpyDeviceToHost);
	int nb_nbh_final;
	cudaMemcpy( &nb_nbh_final, nb_nbh + cellsa.size() - 1, sizeof(int), cudaMemcpyDeviceToHost);*/
	
	//int total_interactions = nb_nbh_incr_final + nb_nbh_final;
	
	int total_interactions = nb_nbh[incr_cell - 1] + nb_nbh_incr[incr_cell - 1];
	
	onika::memory::CudaMMVector<int> interaction_driver_incr;
	//int* interaction_driver_incr;
	interaction_driver_incr.resize(cells_a.size());
	//cudaMalloc(&interaction_driver_incr, cells_a.size() * sizeof(int) );
	
	d_temp_storage = nullptr;
	temp_storage_bytes = 0;
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, interaction_driver.data(), interaction_driver_incr.data(), cells_a.size());
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, interaction_driver.data(), interaction_driver_incr.data(), cells_a.size());
	
	cudaFree(d_temp_storage);
	
	int total_interactions_driver = interaction_driver[cells_a.size() - 1] + interaction_driver_incr[cells_a.size() - 1];
	
	//printf("PREPARATION\n");
	
	/*int interaction_driver_final;
	cudaMemcpy( &interaction_driver_final, interaction_driver + cells_a.size() - 1, sizeof(int), cudaMemcpyDeviceToHost);
	int interaction_driver_incr_final;
	cudaMemcpy( &interaction_driver_incr_final, interaction_driver_incr + cells_a.size() - 1, sizeof(int), cudaMemcpyDeviceToHost);
	
	int total_interactions_driver = interaction_driver_final + interaction_driver_incr_final;*/
	
	//printf("PREPARATION\n");
	
	if(!c.use)
	{
		c.initialize();
	}
	//else
	//{
		//c.resize();	
	//}
	
	auto& type0 = c.get_wave(0);
	auto& type4 = c.get_wave(4);
	
	type0.clear();
	type4.clear();
	
	//type0.resize(total_interactions, 0);
	//type4.resize(total_interactions_driver, 4);
	
	type0.resize(total_interactions, 0);
	type4.resize(total_interactions_driver, 4);
	
	type0.size2 = total_interactions;
	type4.size2 = total_interactions_driver;
	
	//onika::memory::CudaMMVector<uint64_t> &id_i = type0.id_i;
	cudaMalloc(&type0.id_i, total_interactions * sizeof(uint64_t) );
	//onika::memory::CudaMMVector<uint64_t> &id_j = type0.id_j;
	cudaMalloc(&type0.id_j, total_interactions * sizeof(uint64_t) );
	//onika::memory::CudaMMVector<uint32_t> &cell_i = type0.cell_i;
	cudaMalloc(&type0.cell_i, total_interactions * sizeof(uint32_t) );
	//onika::memory::CudaMMVector<uint32_t> &cell_j = type0.cell_j;
	cudaMalloc(&type0.cell_j, total_interactions * sizeof(uint32_t) );
	//onika::memory::CudaMMVector<uint16_t> &p_i = type0.p_i;
	cudaMalloc(&type0.p_i, total_interactions * sizeof(uint16_t) );
	//onika::memory::CudaMMVector<uint16_t> &p_j = type0.p_j;
	cudaMalloc(&type0.p_j, total_interactions * sizeof(uint16_t) );
	
	cudaMalloc(&type0.ft_x, total_interactions * sizeof(double) );
	cudaMalloc(&type0.ft_y, total_interactions * sizeof(double) );
	cudaMalloc(&type0.ft_z, total_interactions * sizeof(double) );
	
	cudaMalloc(&type0.mom_x, total_interactions * sizeof(double) );
	cudaMalloc(&type0.mom_y, total_interactions * sizeof(double) );
	cudaMalloc(&type0.mom_z, total_interactions * sizeof(double) );
	
	cudaMalloc(&type0.sub_i, total_interactions * sizeof(uint16_t) );
	cudaMalloc(&type0.sub_j, total_interactions * sizeof(uint16_t) );
	
	/*onika::memory::CudaMMVector<uint64_t> &id_driver = type4.id_i;
	onika::memory::CudaMMVector<uint32_t> &cell_driver = type4.cell_i;
	onika::memory::CudaMMVector<uint16_t> &p_driver = type4.p_i;*/
	
	cudaMalloc(&type4.id_i, total_interactions_driver * sizeof(uint64_t) );
	cudaMalloc(&type4.id_j, total_interactions_driver * sizeof(uint64_t) );

	cudaMalloc(&type4.cell_i, total_interactions_driver * sizeof(uint32_t) );
	cudaMalloc(&type4.cell_j, total_interactions_driver * sizeof(uint32_t) );

	cudaMalloc(&type4.p_i, total_interactions_driver * sizeof(uint16_t) );
	cudaMalloc(&type4.p_j, total_interactions_driver * sizeof(uint16_t) );
	
	cudaMalloc(&type4.ft_x, total_interactions_driver* sizeof(double) );
	cudaMalloc(&type4.ft_y, total_interactions_driver * sizeof(double) );
	cudaMalloc(&type4.ft_z, total_interactions_driver * sizeof(double) );
	
	cudaMalloc(&type4.mom_x, total_interactions_driver * sizeof(double) );
	cudaMalloc(&type4.mom_y, total_interactions_driver * sizeof(double) );
	cudaMalloc(&type4.mom_z, total_interactions_driver * sizeof(double) );
	
	cudaMalloc(&type4.sub_i, total_interactions_driver * sizeof(uint16_t) );
	cudaMalloc(&type4.sub_j, total_interactions_driver * sizeof(uint16_t) );
	
	//printf("UN\n");
	
	kernelDEUX<<<incr_cell, BlockSize>>>( cells, dims, cellsa.data(), cellsb.data(), ghost_cells.data(), *nbh_dist_lab, domain->xform(), *rcut_inc, nb_nbh_incr.data(), m_origin, m_offset, m_cell_size, /*id_i.data()*/type0.id_i, /*id_j.data()*/type0.id_j, /*cell_i.data()*/type0.cell_i, /*cell_j.data()*/type0.cell_j, /*p_i.data()*/type0.p_i, /*p_j.data()*/type0.p_j );
	
	cudaDeviceSynchronize();

	//printf("DEUX\n");
	
	kernelTROIS<<<cells_a.size(), BlockSize>>>( cells, dims, cells_a.data()/*cells_a_GPU*/, *rcut_inc, driver, interaction_driver_incr.data(), /*id_driver.data()*/type4.id_i, /*cell_driver.data()*/type4.cell_i, /*p_driver.data()*/type4.p_i );
	
	cudaDeviceSynchronize();
	
	//printf("TROIS\n");
	
	/*cudaFree(nb_nbh);
	cudaFree(nb_nbh_incr);
	
	cudaFree(interaction_driver);
	cudaFree(interaction_driver_incr);*/
	
	
	/*printf("SIZE_ZERO : %d\n", type0.size());
	printf("SIZE_QUATTRO : %d\n", type4.size() );*/
	
	//printf("CLASSIFY\n");
	
	if(c.use)
	{
		for(int type = 0; type < 13; type++)
		{
			auto [/*data*/interactions, size] = c.get_info(type);
			
			if(size > 0)
			{
				//printf("TYPE:%d\n", type);
				//InteractionWrapper<InteractionSOA> interactions(data);
				
				int blockSize = 256;
				int numBlocks = ( size + blockSize - 1 ) / blockSize;
				
				//onika::memory::CudaMMVector<uint64_t> keys;
				uint64_t* keys;
				//keys.resize(size);
				cudaMalloc(&keys, size * sizeof(uint64_t) );
				
				//onika::memory::CudaMMVector<uint64_t> keys_sorted;
				uint64_t* keys_sorted;
				//keys_sorted.resize(size);
				cudaMalloc(&keys_sorted, size * sizeof(uint64_t) );
				
				//onika::memory::CudaMMVector<int> indices;
				int* indices;
				//indices.resize(size);
				cudaMalloc(&indices, size * sizeof(int) );
				
				//onika::memory::CudaMMVector<int> indices_sorted;
				int* indices_sorted;
				//indices_sorted.resize(size);
				cudaMalloc(&indices_sorted, size * sizeof(int) );
				
				int min = 0;
				int max = grid->number_of_particles() - 1;
				
				generateKeys<<<numBlocks, blockSize>>>( keys, interactions.id_i, interactions.id_j, indices, min, max, type, size);
				
				sortWithIndices( keys, indices, keys_sorted, indices_sorted, size);
				
				auto& unc_type = unc.waves[type];
				
				/*auto* old_keys = old.keys;
				auto* old_ftx = old.ft_x;
				auto* old_fty = old.ft_y;
				auto* old_ftz = old.ft_z;
				auto* old_momx = old.mom_x;
				auto* old_momy = old.mom_y;
				auto* old_momz = old.mom_z;
				auto* old_indices = old.indices;*/
				
				//numBlocks = (size + old.size + blockSize - 1) / blockSize;
				
				InteractionOLDWrapper old(unc_type);
				
				find_common_elements<<<numBlocks, blockSize>>>( keys_sorted, old.keys, size, old.size, interactions.ft_x, interactions.ft_y, interactions.ft_z, old.ft_x, old.ft_y, old.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, old.mom_x, old.mom_y, old.mom_z, indices_sorted, old.indices);
				
				cudaDeviceSynchronize();
  				/*cudaFree(old.keys);
  				cudaFree(old.ft_x);
  				cudaFree(old.ft_y);
  				cudaFree(old.ft_z);
  				cudaFree(old.mom_x);
  				cudaFree(old.mom_y);
  				cudaFree(old.mom_z);
  				cudaFree(old.indices);*/
  				
  				cudaFree(keys);
  				cudaFree(keys_sorted);
  				cudaFree(indices);
  				cudaFree(indices_sorted);
				//printf("TYPE END\n");
			}
		}
	}
	else
	{
	
		c.use = true;
	}
	
	//cudaFree(cellsa_GPU);
	//cudaFree(ghost_cells_GPU);
	//cudaFree(cellsb_GPU);
	
	//free(cellsa);
	//free(ghost_cells);
	//free(cellsb);
	
	printf("CLASSIFY_END\n");

    }
  };
  

  // === register factories ===
  ONIKA_AUTORUN_INIT(chunk_neighbors_contact) { OperatorNodeFactory::instance()->register_factory("chunk_neighbors_contact", make_grid_variant_operator<ChunkNeighborsContact>); }

} // namespace exaDEM
