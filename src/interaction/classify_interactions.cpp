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
#include <exanb/core/grid.h>
#include <memory>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/classifier/interactionSOA.hpp>
#include <exaDEM/classifier/interactionAOS.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/traversal.hpp>
#include <cub/cub.cuh>

#include <exaDEM/classifier/interactionSOA.hpp>

namespace exaDEM
{
  using namespace exanb;
  
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
  		
  		if(type == 0){ keys[idx] = (id_i[idx] - min) * range + (id_j[idx] - min); }
  		else if(type == 4){ keys[idx] = id_i[idx]; }
  		
  		indices[idx] = idx;
  	}
  }

	void sortWithIndices2(uint64_t* id_in, int *indices_in, uint64_t* id_out, int* indices_out, int size) {
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

  template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>> class ClassifyInteractions : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_vrot, field::_arot>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(GridCellParticleInteraction, ges, INPUT, DocString{"Interaction list"});
    ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});
    //ADD_SLOT(OldClassifier, ic_old, INPUT_OUTPUT);
    ADD_SLOT(OldClassifiers, ic_olds, INPUT_OUTPUT);
    
    //ADD_SLOT(InteractionSOA, interaction_type0, INPUT_OUTPUT);
    //ADD_SLOT(InteractionSOA, interaction_type4, INPUT_OUTPUT);

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
                )EOF";
    }

    inline void execute() override final
    {
    
      //printf("CLASSIFY\n");
    
      if (grid->number_of_cells() == 0)
      {
        return;
      }
      /*auto [cell_ptr, cell_size] = traversal_real->info();
      if (!ic.has_value())
        ic->initialize();
      ic->classify(*ges, cell_ptr, cell_size);
      ic->prefetch_memory_on_gpu(); // GPU only*/

     auto &olds = *ic_olds;
            
      auto& c = *ic;
      
      /*auto [data, size] = c.get_info(0);
      
      auto& type0 = *interaction_type0;
      
      printf("TYPE0: OLD: %d NEW: %d\n", size, type0.ft_x.size());
      
      auto [data2, size2] = c.get_info(4);
      
      auto& type4 = *interaction_type4;
      
      printf("TYPE4: OLD: %d NEW: %d\n", size2, type4.ft_x.size());
      
      if(size!=type0.ft_x.size())
      {
      	printf("ERREUR DE TAILLE\n"); getchar();
      }
      	
      	int loop = std::min(size, type0.ft_x.size());
      	
      	int err = 0;
      	
      	for(int i = 0; i < loop; i++)
      	{
      		if(data.id_i[i] != type0.id_i[i] || data.id_j[i] != type0.id_j[i] || data.cell_i[i] != type0.cell_i[i] || data.cell_j[i] != type0.cell_j[i] || data.p_i[i] != type0.p_i[i] || data.p_j[i] != type0.p_j[i]){ err++; printf("I:%d   ",i);       	printf("PREUVE: INTERACTION_UN(ID_I:%d ID_J:%d CELL_I:%d CELL_J:%d P_I:%d P_J:%d)   INTERACTION_DEUX(ID_I:%d ID_J:%d CELL_I:%d CELL_J:%d P_I:%d P_J:%d)\n", data.id_i[i], data.id_j[i], data.cell_i[i], data.cell_j[i], data.p_i[i], data.p_j[i], type0.id_i[i], type0.id_j[i], type0.cell_i[i], type0.cell_j[i], type0.p_i[i], type0.p_j[i]);}//printf("ERREUR_%d\n", i);
      	}
      	if(err > 0){ printf("ERREURS: %d\n", err); getchar(); }
      	//getchar();*/
      
      /*auto& waves = ic->waves;
      
      waves[0].clear();
      waves[0] = *interaction_type0;
      
      waves[4].clear();
      waves[4] = *interaction_type4;*/
      
      for(int type = 0; type < 13; type++)
      {
      
      auto [data, size] = c.get_info(type);
      
      if(size > 0)
      {

      InteractionWrapper<InteractionSOA> interactions(data);
      
      int blockSize = 256;
      int numBlocks = ( size + blockSize - 1 ) / blockSize;
      
      onika::memory::CudaMMVector<uint64_t> keys;
      keys.resize(size);
      
      onika::memory::CudaMMVector<uint64_t> keys_sorted;
      keys_sorted.resize(size);
      
      onika::memory::CudaMMVector<int> indices;
      indices.resize(size);
      
      onika::memory::CudaMMVector<int> indices_sorted;
      indices_sorted.resize(size);
      
      int min = 0;
      int max = grid->number_of_particles() - 1;

      generateKeys<<<numBlocks, blockSize>>>( keys.data(), interactions.id_i, interactions.id_j, indices.data(), min, max, type, size);

      sortWithIndices2( keys.data(), indices.data(), keys_sorted.data(), indices_sorted.data(), size);
      
      auto &o = olds.waves[type];
      
      OldClassifierWrapper old(o);
      
      find_common_elements<<<numBlocks, blockSize>>>( keys.data(), old.keys, size, old.size, interactions.ft_x, interactions.ft_y, interactions.ft_z, old.ft_x, old.ft_y, old.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, old.mom_x, old.mom_y, old.mom_z, indices_sorted.data(), old.indices);
      
      }
      }
      
     //printf("END_CLASSIFY\n");  
    }
  };

  template <class GridT> using ClassifyInteractionsTmpl = ClassifyInteractions<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("classify_interactions", make_grid_variant_operator<ClassifyInteractionsTmpl>); }
} // namespace exaDEM
