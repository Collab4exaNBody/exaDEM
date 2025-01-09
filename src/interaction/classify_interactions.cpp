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

namespace exaDEM
{
  using namespace exanb;
  
  __global__ void kernel( double* ft_x,
  			double* ft_y,
  			double* ft_z,
  			double* mom_x,
  			double* mom_y,
  			double* mom_z,
  			uint64_t* id_i,
  			uint64_t* id_j,
  			double* ft_x_old,
  			double* ft_y_old,
  			double* ft_z_old,
  			double* mom_x_old,
  			double* mom_y_old,
  			double* mom_z_old,
  			uint64_t* id_i_old,
  			uint64_t* id_j_old,
  			int* indices,
  			int* particles,
  			int* particles_incr,
  			size_t size)
  {
  	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  	
  	if( idx < size )
  	{
  		int id = id_i[idx];
  		
  		if(particles[id] > 0)
  		{
  			int p = particles_incr[id];
  			
  			int id2 = id_j[idx];
  			
  			for(int i = 0; i < particles[id]; i++)
  			{
  				if( id2 == id_j_old[indices[p + i]])
  				{
  					ft_x[idx] = ft_x_old[indices[p + i]];
  					ft_y[idx] = ft_y_old[indices[p + i]]; 
  					ft_z[idx] = ft_z_old[indices[p + i]];
  					
  					mom_x[idx] = mom_x_old[indices[p + i]];  
  					mom_y[idx] = mom_y_old[indices[p + i]];  
  					mom_z[idx] = mom_z_old[indices[p + i]];   
  				}
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
    
    ADD_SLOT(CellListWrapper, cell_list, INPUT, DocString{"list of non empty cells within the current grid"});
    ADD_SLOT(OldClassifier, ic_old, INPUT_OUTPUT);

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
      auto [cell_ptr, cell_size] = traversal_real->info();
      if (!ic.has_value())
        ic->initialize();
      ic->classify(*ges, cell_ptr, cell_size);
      //ic->prefetch_memory_on_gpu(); // GPU only
      
      auto [data, size] = ic->get_info(0);
      
      InteractionWrapper<InteractionSOA> interactions(data);
      
      int blockSize = 256;
      int numBlocks = ( size + blockSize - 1 ) / blockSize;
      
      auto& old = *ic_old;
      
      /*for(int i = 0; i < old.id_i.size(); i++)
      {
      	printf("ID_I: %d   ID_J: %d   FTX: %f   FTY: %f   FTZ: %f   MOMX: %f   MOMY: %f   MOMZ: %f   Particles: %d   Particles_Incr: %d\n", old.id_i[i], old.id_j[i], old.ft_x[i], old.ft_y[i], old.ft_z[i], old.mom_x[i], old.mom_y[i], old.mom_z[i], old.particles[old.id_i[i]], old.particles_incr[old.id_i[i]]);
      }
      
      getchar();*/
      
      kernel<<<numBlocks, blockSize>>>( interactions.ft_x, interactions.ft_y, interactions.ft_z, interactions.mom_x, interactions.mom_y, interactions.mom_z, interactions.id_i, interactions.id_j, old.ft_x.data(), old.ft_y.data(), old.ft_z.data(), old.mom_x.data(), old.mom_y.data(), old.mom_z.data(), old.id_i.data(), old.id_j.data(), old.indices.data(), old.particles.data(), old.particles_incr.data(), size);
      
      //printf("CLASSIFY END\n");
      
    }
  };

  template <class GridT> using ClassifyInteractionsTmpl = ClassifyInteractions<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("classify_interactions", make_grid_variant_operator<ClassifyInteractionsTmpl>); }
} // namespace exaDEM
