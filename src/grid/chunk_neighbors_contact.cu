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

#include <exaDEM/traversal.h>
#include <exaDEM/CellsInfoNeighborsSearch.h>
#include <exaDEM/classifier/classifier.hpp>

#include <exaDEM/drivers.h>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shape_detection_driver.hpp>

#include <exaDEM/nbh_polyhedron.h>

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
  
  
  
__device__ bool intersectOBB_GPU(const OBB& a, const OBB& b, double tol = 1e-10)
{
    double R[3][3], AbsR[3][3];
    double ra, rb;

    const vec3r A[3] = { a.e1, a.e2, a.e3 };
    const vec3r B[3] = { b.e1, b.e2, b.e3 };

    // Compute rotation matrix and its absolute value
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            R[i][j] = A[i] * B[j];               // dot product
            AbsR[i][j] = fabs(R[i][j]) + tol;    // avoid arithmetic error
        }
    }

    // Compute translation vector in A's frame
    vec3r t_world = b.center - a.center;
    vec3r t(dot(t_world, A[0]), dot(t_world, A[1]), dot(t_world, A[2]));

    const double Ea[3] = { a.extent.x, a.extent.y, a.extent.z };
    const double Eb[3] = { b.extent.x, b.extent.y, b.extent.z };

    // Test axes of A: A0, A1, A2
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        ra = Ea[i];
        rb = Eb[0] * AbsR[i][0] + Eb[1] * AbsR[i][1] + Eb[2] * AbsR[i][2];
        if (fabs(t[i]) > ra + rb) return false;
    }

    // Test axes of B: B0, B1, B2
    #pragma unroll
    for (int j = 0; j < 3; ++j)
    {
        ra = Ea[0] * AbsR[0][j] + Ea[1] * AbsR[1][j] + Ea[2] * AbsR[2][j];
        rb = Eb[j];
        double proj = t.x * R[0][j] + t.y * R[1][j] + t.z * R[2][j];
        if (fabs(proj) > ra + rb) return false;
    }

    // Test cross products of axes: A[i] x B[j]
    #pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        int u = (i + 1) % 3;
        int v = (i + 2) % 3;

        #pragma unroll
        for (int j = 0; j < 3; ++j)
        {
            ra = Ea[u] * AbsR[v][j] + Ea[v] * AbsR[u][j];
            rb = Eb[(j+1)%3] * AbsR[i][(j+2)%3] + Eb[(j+2)%3] * AbsR[i][(j+1)%3];
            double proj = t[v] * R[u][j] - t[u] * R[v][j];
            if (fabs(proj) > ra + rb) return false;
        }
    }

    return true; // no separating axis found
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
			      IJK dims,
                              int* cellsa,
                              int* cellsb,
                              int* ghost_cell,
                              shapes shps,
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
	
	cell_accessors cellA(cells[cell_a]);
	
	int cell_b = cellsb[blockIdx.x];
	int num_particles_b = cells[cell_b].size();
	auto* __restrict__ rx_b = cells[cell_b][field::rx];
	auto* __restrict__ ry_b = cells[cell_b][field::ry];
	auto* __restrict__ rz_b = cells[cell_b][field::rz];
	auto* __restrict__ id_b = cells[cell_b][field::id];
	auto* __restrict__ rad_b = cells[cell_b][field::radius];
	IJK loc_b = grid_index_to_ijk( dims, cell_b );
	AABB cellb_AABB_pre = AABB{ (origin+((offset+loc_b)*cell_size)), (origin+((offset+loc_b+1)*cell_size))};
	AABB cellb_AABB = enlarge(  cellb_AABB_pre, rcut_inc + 2 * rad_b[0] );
	
	cell_accessors cellB(cells[cell_b]);
	
	double rcut2 = dist_lab * dist_lab;
	int nb_interactions = 0;
	
	bool is_ghost = ghost_cell[blockIdx.x];
	
	//if(threadIdx.y < num_particles_a)
	for(int p_a = threadIdx.y; p_a < num_particles_a; p_a+= blockDim.y)
	{
		const Vec3d pos_a = { rx_a[p_a] , ry_a[p_a] , rz_a[p_a] };
		
		int ida = id_a[p_a];
		
		//if(threadIdx.x < num_particles_b)
		//if( is_inside( cellb_AABB, pos_a))
		//{
		for(int p_b = threadIdx.x; p_b < num_particles_b; p_b+= blockDim.x)
		{
			const Vec3d pos_b = { rx_b[p_b], ry_b[p_b], rz_b[p_b] };
			const Vec3d dr = pos_a - pos_b;
				
			double d2 = norm2( xform * dr );
				
			if( nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell_a, p_a, cell_b, p_b) )
			{
				if(ida < id_b[p_b] || is_ghost)
				{
					nb_interactions++;
					//printf("CELL_I: %d P_I: %d CELL_J: %d P_J: %d\n", cell_a, p_a, cell_b, p_b);
					//printf("P_I: %d \n", p_a);
				}
			}

		}
		//}	
	}
	
	int aggregate = BlockReduce(temp_storage).Sum(nb_interactions);
	__syncthreads();
	if(threadIdx.x == 0 && threadIdx.y == 0) nb_nbh[blockIdx.x] = aggregate;
}

template< class GridT > __global__ void kernelOBB(GridT* cells,
                              uint32_t* cellsa,
                              uint32_t* cellsb,
                              uint16_t* pa,
                              uint16_t* pb,
                              shapes shps,
                              int* res,
                              double rcut_inc,
                              int size)
{
	//using BlockReduce = cub::BlockReduce<int, 256>;
	//__shared__ typename BlockReduce::TempStorage temp_storage;
	
	//__shared__ int s[256];
	//s[threadIdx.x] = 0;
	//__syncthreads();
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < size)
	{
		int cell_a = cellsa[idx];
		cell_accessors cellA(cells[cell_a]);
	
		int cell_b = cellsb[idx];
		cell_accessors cellB(cells[cell_b]);

		int p_a = pa[idx];
		particle_info p(shps, p_a, cellA);
		OBB obb_i = p.shp->obb;
		quat conv_orient_i = p.get_quat();
		obb_i.rotate(conv_orient_i);
		obb_i.translate(vec3r{p.r.x, p.r.y, p.r.z});
		obb_i.enlarge(rcut_inc);

		int p_b = pb[idx];
		particle_info p_nbh(shps, p_b, cellB);
		OBB obb_j = p_nbh.shp->obb;
		obb_j.rotate(p_nbh.get_quat());
		obb_j.translate(vec3r{p_nbh.r.x, p_nbh.r.y, p_nbh.r.z});
	
		//if(idx == 1) printf("(%d %d), (%d, %d)\n", cell_a, p_a, cell_b, p_b);
		
		//int aggregate = BlockReduce(temp_storage).Sum(
		
		int sum = 0;
	
		if(intersectOBB_GPU(obb_i, obb_j)) 
		{
			sum = 1;
			//s[threadIdx.x] = 0;
			atomicAdd(&res[blockIdx.x], 1);
		}
		
		//if(blockIdx.x == 0) printf("IDX: %d (CELL_I:%d P_I:%d CELL_J:%d P_J:%d) VAL:%d\n", idx, cell_a, p_a, cell_b, p_b, sum);
		//atomicAdd(&res[blockIdx.x], sum);
		
		//if(sum == 0) printf("POUR %d PAS DE INTERSEXT\n", idx);
		//__syncthreads();
		
		//if(threadIdx.x == 0)
		//{
		//	for(int i = 0; i < 256; i++)
		//	{
		//		res[blockIdx.x]+= s[i]; 
		//	}	
		//}
		//__syncthreads();
		
		//int aggregate = BlockReduce(temp_storage).Sum(sum);
		//__syncthreads();
		//if(threadIdx.x == 0)
		//{
			//if(idx == 1) printf("ANCHE QUI?\n");
		//	res[blockIdx.x] = aggregate;
		//}
	}
}

template< class GridT > __global__ void kernelOBB2(GridT* cells,
                              uint32_t* cellsa,
                              uint32_t* cellsb,
                              uint16_t* pa,
                              uint16_t* pb,
                              shapes shps,
                              int* res_incr,
                              double rcut_inc,
                              uint32_t* cell_i,
                              uint32_t* cell_j,
                              uint16_t* p_i, 
                              uint16_t* p_j,
                              int size)
{
	//using BlockScan = cub::BlockScan<int, 256>;
	//__shared__ typename BlockScan::TempStorage temp_storage;
	
	__shared__ int s[256];
	s[threadIdx.x] = 0;
	__syncthreads();	
	
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < size)
	{

		//__shared__ int s[256];
		//s[threadIdx.x] = 0;
		//__syncthreads();
	
		int cell_a = cellsa[idx];
		int num_particles_a = cells[cell_a].size();
		auto* __restrict__ rx_a = cells[cell_a][field::rx];
		auto* __restrict__ ry_a = cells[cell_a][field::ry];
		auto* __restrict__ rz_a = cells[cell_a][field::rz];
		auto* __restrict__ id_a = cells[cell_a][field::id];
		auto* __restrict__ rad_a = cells[cell_a][field::radius];
	
		cell_accessors cellA(cells[cell_a]);
	
		int cell_b = cellsb[idx];
		int num_particles_b = cells[cell_b].size();
		auto* __restrict__ rx_b = cells[cell_b][field::rx];
		auto* __restrict__ ry_b = cells[cell_b][field::ry];
		auto* __restrict__ rz_b = cells[cell_b][field::rz];
		auto* __restrict__ id_b = cells[cell_b][field::id];
		auto* __restrict__ rad_b = cells[cell_b][field::radius];	
	
		cell_accessors cellB(cells[cell_b]);

		int p_a = pa[idx];
		particle_info p(shps, p_a, cellA);
		OBB obb_i = p.shp->obb;
		quat conv_orient_i = p.get_quat();
		obb_i.rotate(conv_orient_i);
		obb_i.translate(vec3r{p.r.x, p.r.y, p.r.z});
		obb_i.enlarge(rcut_inc);

		int p_b = pb[idx];
		particle_info p_nbh(shps, p_b, cellB);
		OBB obb_j = p_nbh.shp->obb;
		obb_j.rotate(p_nbh.get_quat());
		obb_j.translate(vec3r{p_nbh.r.x, p_nbh.r.y, p_nbh.r.z});
		
		int prefix = 0;
		
		//int val = 0;
		
		//if(idx == 1) printf("KERNEL DEUX (%d %d), (%d, %d)\n", cell_a, p_a, cell_b, p_b);

		if(intersectOBB_GPU(obb_i, obb_j)) 
		{
			//val = 1;
			s[threadIdx.x] = 1;
			//if(blockIdx.x == 0) printf("INTERSECTIONNNNNNNNN %d\n", idx);
			
		}	
       		
       		//BlockScan(temp_storage).ExclusiveSum( val , prefix );
        	__syncthreads();
        	
        	//if(blockIdx.x==0) printf("OBB IDX:%d (CELL_I:%d P_I:%d CELL_J:%d P_J:%d) VAL:%d\n", idx, cell_a, p_a, cell_b, p_b, s[threadIdx.x]);
 
 		//if(threadIdx.x > 0)
 		//{
 		for(int i = 0; i < threadIdx.x; i++)
 		{
 			prefix+= s[i];
 		//}
 		       	
        	//prefix+= res_incr[blockIdx.x];
        	}
        	prefix+= res_incr[blockIdx.x];
        	__syncthreads();

		//if(idx == 1)
		//{
		//	printf("PREFIX: %d CELLI: %d CELLJ: %d PI: %d PJ: %d\n", prefix, cell_i[prefix], cell_j[prefix], p_i[prefix], p_j[prefix]);
		//}
        	if(s[threadIdx.x] > 0)
        	{
			cell_i[prefix] = cell_a;
			cell_j[prefix] = cell_b;
			p_i[prefix] = p_a;
			p_j[prefix] = p_b;
			//if(blockIdx.x == 0) printf("IDX: %d PREFIX: %d VAL: %d RES: %d\n", idx, prefix, val, res_incr[blockIdx.x]); 
			//if(prefix>=0 && prefix<=244) printf("INDEX:%d BASE(CELL_I:%d P_I:%d CELL_J:%d P_J:%d)  OBB(CELL_I:%d P_I:%d CELL_J:%d P_J:%d)\n", prefix, cell_a, p_a, cell_b, p_b, cell_i[prefix], p_i[prefix], cell_j[prefix], p_j[prefix]);
        	}
        	
        	//if(idx == 1)
		//{
		//	printf("AFTER PREFIX: %d CELLI: %d CELLJ: %d PI: %d PJ: %d\n", prefix, cell_i[prefix], cell_j[prefix], p_i[prefix], p_j[prefix]);
		//}
        }
}

template< class GridT > __global__ void kernelDEUX(GridT* cells,
			      IJK dims,
                              int* cellsa,
                              int* cellsb,
                              int* ghost_cell,
                              shapes shps,
                              const double dist_lab, 
                              Mat3d xform, 
                              double rcut_inc,
                              int* nb_nbh_incr,
                              uint32_t* cell_i,
                              uint32_t* cell_j,
                              uint16_t* p_i,
                              uint16_t* p_j,
                              Vec3d origin,
                              IJK offset,
                              double cell_size)
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
	
	cell_accessors cellA(cells[cell_a]);
	
	int cell_b = cellsb[blockIdx.x];
	int num_particles_b = cells[cell_b].size();
	auto* __restrict__ rx_b = cells[cell_b][field::rx];
	auto* __restrict__ ry_b = cells[cell_b][field::ry];
	auto* __restrict__ rz_b = cells[cell_b][field::rz];
	auto* __restrict__ id_b = cells[cell_b][field::id];
	auto* __restrict__ rad_b = cells[cell_b][field::radius];
	IJK loc_b = grid_index_to_ijk( dims, cell_b );
	AABB cellb_AABB_pre = AABB{ (origin+((offset+loc_b)*cell_size)), (origin+((offset+loc_b+1)*cell_size))};
	AABB cellb_AABB = enlarge(  cellb_AABB_pre, rcut_inc + 2 * rad_b[0] );
	
	cell_accessors cellB(cells[cell_b]);
	
	double rcut2 = dist_lab * dist_lab;
	int nb_interactions = 0;
	int prefix = 0;
	
	bool is_ghost = ghost_cell[blockIdx.x];
	
	int count = 0;
	
	//if(threadIdx.y < num_particles_a)
	for(int p_a = threadIdx.y; p_a < num_particles_a; p_a+= blockDim.y)
	{
		const Vec3d pos_a = { rx_a[p_a] , ry_a[p_a] , rz_a[p_a] };
		
		int ida = id_a[p_a];

		//if(threadIdx.x < num_particles_b)
		//if( is_inside( cellb_AABB, pos_a ))
		//{
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
					count++;
				}
			}
		}
		//}	
	}

        BlockScan(temp_storage).ExclusiveSum( nb_interactions , prefix );
        __syncthreads();
        prefix+= nb_nbh_incr[blockIdx.x];
        
        int nb2 = 0;
    
    	if(count > 0)
    	//if(threadIdx.y < num_particles_a)
	for(int p_a = threadIdx.y; p_a < num_particles_a; p_a+= blockDim.y)
	{
		const Vec3d pos_a = { rx_a[p_a] , ry_a[p_a] , rz_a[p_a] };
		
		int ida = id_a[p_a];

		//if(threadIdx.x < num_particles_b)
		//if( is_inside(cellb_AABB, pos_a))
		//{
		for(int p_b = threadIdx.x; p_b < num_particles_b; p_b+= blockDim.x)
		{
			const Vec3d pos_b = { rx_b[p_b], ry_b[p_b], rz_b[p_b] };
			const Vec3d dr = pos_a - pos_b;
				
			double d2 = norm2( xform * dr );
				
			if( nbh_filter_GPU(cells, rcut_inc, d2, rcut2, cell_a, p_a, cell_b, p_b))
			{
				if(ida < id_b[p_b] || is_ghost)
				{
					cell_i[prefix + nb2] = cell_a;
					cell_j[prefix + nb2] = cell_b;
					p_i[prefix + nb2] = p_a;
					p_j[prefix + nb2] = p_b;
						
					nb2++;
				}
			}
		}
		//}	
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
    
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});

    ADD_SLOT(ChunkNeighborsConfig, config, INPUT, ChunkNeighborsConfig{});
    ADD_SLOT(ChunkNeighborsScratchStorage, chunk_neighbors_scratch, PRIVATE);
    
    //ADD_SLOT(InteractionSOA, interactions_intermediaire, INPUT_OUTPUT);
    
    ADD_SLOT(shapes, shapes_collection, INPUT, DocString{"Collection of shapes"});
    
    ADD_SLOT(Interactions_intermediaire, interactions_inter, INPUT_OUTPUT);
    //ADD_SLOT(InteractionSOA, interactions_inter, INPUT_OUTPUT);
    

    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator builds the neighbor lists used to detect contacts between particles.
        )EOF";
    }

    inline void execute() override final
    {
      //printf("CHUNK\n");
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
      
        auto& g = *grid;
        IJK dims = g.dimension();
        const size_t n_cells = g.number_of_cells();
        
        auto& amr_grid_pairs2 = *amr_grid_pairs;
        const unsigned int loc_max_gap = amr_grid_pairs2.cell_layers();

	auto &shps = *shapes_collection;

 	std::vector<int> cellsb_ids;
 	std::vector<int> number_of_cells_neighbors;
 	
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
	
	std::vector<int> cells_a;
	std::vector<int> incr_cells_a;
	
	auto [cell_ptr, cell_size] = traversal_real->info();
	
	int incr_cell = 0;
	
	//printf("NUMERO DE CELLULES: %d\n", n_cells);
	
	int particles_average = 0;
	
	for(int i = 0; i < n_cells; i++)
	{
		if( cells[i].size() > 0 && is_in( i, cell_ptr, cell_size ) )
		{ 
			cells_a.push_back(i); 
			incr_cells_a.push_back( incr_cell ); 
			incr_cell+= number_of_cells_neighbors[i];
			particles_average+= cells[i].size(); 
		} 
	}
	
	//printf("NOMBRE DE CELLULES: %d\n", cells_a.size());
	
	//printf("MOYENNE PARTICULES: %d\n", (int)(particles_average/cells_a.size()));

	onika::memory::CudaMMVector<int> cellsa;
	cellsa.resize(incr_cell);
	
	onika::memory::CudaMMVector<int> cellsb;
	cellsb.resize(incr_cell);
	
	onika::memory::CudaMMVector<int> ghost_cells;
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
	
	dim3 BlockSize(	block_x, block_y, 1);
	
	auto m_origin = g.origin();
	auto m_offset = g.offset();
	auto m_cell_size = g.cell_size();
	
	onika::memory::CudaMMVector<int> nb_nbh;

	nb_nbh.resize( incr_cell );

	int nb_interactions=0;
	
	kernelUN<<<incr_cell, BlockSize>>>( cells, dims, cellsa.data(), cellsb.data(), ghost_cells.data(), shps, *nbh_dist_lab, domain->xform(), *rcut_inc, nb_nbh.data(), m_origin, m_offset, m_cell_size );
	
	cudaDeviceSynchronize();

	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;
	
	onika::memory::CudaMMVector<int> nb_nbh_incr;

	nb_nbh_incr.resize( incr_cell );
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), incr_cell );
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, nb_nbh.data(), nb_nbh_incr.data(), incr_cell );
	
	cudaFree(d_temp_storage);

	int total_interactions = nb_nbh[incr_cell - 1] + nb_nbh_incr[incr_cell - 1];

	uint16_t* p_i;
	uint16_t* p_j;
	uint32_t* cell_i;
	uint32_t* cell_j;

	cudaMalloc(&p_i, total_interactions * sizeof(uint16_t));
	cudaMalloc(&p_j, total_interactions * sizeof(uint16_t));
	cudaMalloc(&cell_i, total_interactions * sizeof(uint32_t));
	cudaMalloc(&cell_j, total_interactions * sizeof(uint32_t));

	kernelDEUX<<<incr_cell, BlockSize>>>( cells, dims, cellsa.data(), cellsb.data(), ghost_cells.data(), shps, *nbh_dist_lab, domain->xform(), *rcut_inc, nb_nbh_incr.data(), cell_i/*.data()*/, cell_j/*.data()*/, p_i/*.data()*/, p_j/*.data()*/, m_origin, m_offset, m_cell_size );

	cudaDeviceSynchronize();

        int numBlocks = ( total_interactions + 256 - 1) / 256;
        onika::memory::CudaMMVector<int> res;
        res.resize(numBlocks);

        kernelOBB<<<numBlocks, 256>>>( cells, cell_i/*.data()*/, cell_j/*.data()*/, p_i/*.data()*/, p_j/*.data()*/, shps, res.data(), *rcut_inc, total_interactions);
        
        cudaDeviceSynchronize();

        onika::memory::CudaMMVector<int> res_incr;

        res_incr.resize(numBlocks);

        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
        
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, res.data(), res_incr.data(), numBlocks );
	
	cudaMalloc(&d_temp_storage, temp_storage_bytes);
	
	cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, res.data(), res_incr.data(), numBlocks );
	
	cudaFree(d_temp_storage);
	
	d_temp_storage = nullptr;

	int total_interactions2 = res[numBlocks - 1] + res_incr[numBlocks - 1];

       	auto& ints2 = *interactions_inter;
       	
       	cudaFree(ints2.p_i);
       	cudaFree(ints2.p_j);
       	cudaFree(ints2.cell_i);
       	cudaFree(ints2.cell_j);

       	cudaMalloc(&ints2.p_i, total_interactions2 * sizeof(uint16_t));
       	cudaMalloc(&ints2.p_j, total_interactions2 * sizeof(uint16_t));
       	cudaMalloc(&ints2.cell_i, total_interactions2 * sizeof(uint32_t));
       	cudaMalloc(&ints2.cell_j, total_interactions2 * sizeof(uint32_t));

       	auto& size = ints2.size;
       	size = total_interactions2;
       	
       	kernelOBB2<<<numBlocks, 256>>>( cells, cell_i/*.data()*/, cell_j/*.data()*/, p_i/*.data()*/, p_j/*.data()*/, shps, res_incr.data(), *rcut_inc, ints2.cell_i/*.data()*/, ints2.cell_j/*.data()*/, ints2.p_i/*.data()*/, ints2.p_j/*.data()*/, total_interactions);
       	
        cudaDeviceSynchronize();
       	       	
       	cudaFree(cell_i);
       	cudaFree(cell_j);
       	cudaFree(p_i);
       	cudaFree(p_j);

    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(chunk_neighbors_contact) { OperatorNodeFactory::instance()->register_factory("chunk_neighbors_contact", make_grid_variant_operator<ChunkNeighborsContact>); }

} // namespace exaDEM
