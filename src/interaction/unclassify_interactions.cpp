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
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <memory>

#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/compute_contact_force.h>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/classifier/interactionSOA.hpp>
#include <exaDEM/classifier/interactionAOS.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shape_detection_driver.hpp>
#include <exaDEM/mutexes.h>
#include <exaDEM/drivers.h>
#include <cub/cub.cuh>

namespace exaDEM
{
  using namespace exanb;
  

  
  __global__ void filtre_un( double* ft_x,
  			double* ft_y,
  			double* ft_z,
  			double* mom_x,
  			double* mom_y,
  			double* mom_z,
  			size_t size,
  			int* blocks,
  			int* total)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if(idx < size)
  	{
  		if(ft_x[idx]!=0 || ft_y[idx]!=0 || ft_z[idx]!=0 || mom_x[idx]!=0 || mom_y[idx]!=0 || mom_z[idx]!=0)
  		{
  			atomicAdd(&blocks[blockIdx.x], 1);
  			atomicAdd(&total[0], 1);
  		}
  	}
  }
  
  __global__ void filtre_deux( uint64_t* id_i,
  				uint64_t* id_j,
  				uint16_t* sub_j,
  				uint64_t* id_i_res,
  				uint64_t* id_j_res,
  				uint16_t* sub_j_res,
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
  				int* blocks_incr,
  				int* indices,
  				uint64_t* keys,
  				int size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	//printf("RAZ\n");
  	
  	int incr = blocks_incr[blockIdx.x];
  	
  	__shared__ int s[256];
  	
  	s[threadIdx.x] = 0;
  	
  	__syncthreads();
  	
  	
  	if(idx < size)
  	{
  		if(ft_x[idx]!=0 || ft_y[idx]!=0 || ft_z[idx]!=0 || mom_x[idx]!=0 || mom_y[idx]!=0 || mom_z[idx]!=0)
  		{
  			s[threadIdx.x] = 1;
  		}
  		
  		__syncthreads();
  		
  		int index = 0;
  		
  		if(s[threadIdx.x] == 1)
  		{
  			for(int i = 0; i < threadIdx.x; i++)
  			{
  				if( s[i] == 1 ) index++;
  			}
  			
  			id_i_res[index + incr] = id_i[idx];
  			id_j_res[index + incr] = id_j[idx];
  			sub_j_res[index + incr] = sub_j[idx];
  			ft_x_res[index + incr] = ft_x[idx];
  			ft_y_res[index + incr] = ft_y[idx];
  			ft_z_res[index + incr] = ft_z[idx];
  			mom_x_res[index + incr] = mom_x[idx];
  			mom_y_res[index + incr] = mom_y[idx];
  			mom_z_res[index + incr] = mom_z[idx];

  			indices[incr + index] = incr + index;
  			
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

  template <typename GridT, class = AssertGridHasFields<GridT>> class UnclassifyInteractions : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_vrot, field::_arot>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
    ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
    ADD_SLOT(OldClassifiers, ic_olds, INPUT_OUTPUT);

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
                )EOF";
    }

    inline void execute() override final
    {
      // using data_t = std::variant<exaDEM::Cylinder, exaDEM::Surface, exaDEM::UndefinedDriver>;
      
      //printf("UNCLASSIFY\n");
      
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
      
      
      onika::memory::CudaMMVector<int> blocks;
      blocks.resize(numBlocks);
      
      onika::memory::CudaMMVector<int> blocks_incr;
      blocks_incr.resize(numBlocks);
      
      onika::memory::CudaMMVector<int> total;
      total.resize(1);
      
      filtre_un<<<numBlocks, blockSize>>>( interactions.ft_x, interactions.ft_y, interactions.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, size, blocks.data(), total.data());
      
      exclusive_sum( blocks.data(), blocks_incr.data(), numBlocks );
      
      onika::memory::CudaMMVector<uint64_t> id_i_res;
      id_i_res.resize(total[0]);
      
      onika::memory::CudaMMVector<uint64_t> id_j_res;
      id_j_res.resize(total[0]);
      
      onika::memory::CudaMMVector<uint16_t> sub_j_res;
      sub_j_res.resize(total[0]);
	
      onika::memory::CudaMMVector<int> indices;
      indices.resize(total[0]);
      
       onika::memory::CudaMMVector<uint64_t> keys;
       keys.resize(total[0]);
      
      auto &o = olds.waves[type];
      
      o.set( total[0] );
      
      OldClassifierWrapper old(o);
      
      filtre_deux<<<numBlocks, blockSize>>>( interactions.id_i, interactions.id_j, interactions.sub_j, id_i_res.data(), id_j_res.data(), sub_j_res.data(), interactions.ft_x, interactions.ft_y, interactions.ft_z, old.ft_x, old.ft_y, old.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, old.mom_x, old.mom_y, old.mom_z, blocks_incr.data(), indices.data(), keys.data(), size);
      
       cudaDeviceSynchronize();
       
       int min = 0;
       int max = grid->number_of_particles() - 1;
       
       numBlocks = ( total[0] + blockSize - 1 ) / blockSize;

       generateKeys<<<numBlocks, blockSize>>>( keys.data(), id_i_res.data(), id_j_res.data(), min, max, type, total[0]);
       
       sortWithIndices( keys.data(), indices.data(), old.keys, old.indices, total[0]);
       
      }
      }
      
      /*if (grid->number_of_cells() == 0)
      {
        return;
      }
      if (!ic.has_value())
        return;
      ic->unclassify(*ges);*/
      
     //printf("UNCLASSIFY_END\n");
      
    }
  };

  template <class GridT> using UnclassifyInteractionsTmpl = UnclassifyInteractions<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("unclassify_interactions", make_grid_variant_operator<UnclassifyInteractionsTmpl>); }
} // namespace exaDEM
