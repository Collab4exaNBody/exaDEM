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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <memory>

#include <exaDEM/hooke_force_parameters.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/classifier.hpp>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/shape/shape_detection.hpp>
#include <exaDEM/shape/shape_detection_driver.hpp>
#include <exaDEM/mutexes.h>
#include <exaDEM/drivers.h>

namespace exaDEM
{
  using namespace exanb;

  template<typename GridT , class = AssertGridHasFields< GridT, field::_radius >>
    class ClassifyInteractions : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_vrot, field::_arot >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT                       , grid , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( GridCellParticleInteraction , ges  , INPUT , DocString{"Interaction list"} );
    ADD_SLOT( Classifier                  , ic   , INPUT_OUTPUT , DocString{"Interaction lists classified according to their types"} );
    ADD_SLOT( std::vector<size_t>         , idxs , INPUT , DocString{"List of non empty cells"});


    public:

    inline std::string documentation() const override final
    {
      return R"EOF(
                )EOF";
    }

    inline void execute () override final
    {
      if( grid->number_of_cells() == 0 ) { return; }
      if(!ic.has_value()) ic->initialize();
      ic->classify(*ges, *idxs);
    }
  };

  template<class GridT> using ClassifyInteractionsTmpl = ClassifyInteractions<GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "classify_interactions", make_grid_variant_operator< ClassifyInteractionsTmpl > );
  }
}

