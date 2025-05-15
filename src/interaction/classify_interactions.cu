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
#include <exanb/core/grid.h>
#include <memory>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/classifier/interactionSOA.hpp>
#include <exaDEM/classifier/interactionAOS.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/shapes.hpp>

#include <exaDEM/traversal.h>
#include <cub/cub.cuh>

#include <exaDEM/classifier/interactionSOA.hpp>


namespace exaDEM
{
  using namespace exanb;
  

  template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>> class ClassifyInteractions : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_vrot, field::_arot>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(GridCellParticleInteraction, ges, INPUT, DocString{"Interaction list"});
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});
    
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
    
 
    }
  };

  template <class GridT> using ClassifyInteractionsTmpl = ClassifyInteractions<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(classify_interactions) { OperatorNodeFactory::instance()->register_factory("classify_interactions", make_grid_variant_operator<ClassifyInteractionsTmpl>); }
} // namespace exaDEM
