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
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interface/interface.hpp>

namespace exaDEM
{
  using namespace exanb;

  template<typename GridT>
    class CheckClassifierInteractionPair : public OperatorNode
  {
    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(Classifier, ic, INPUT, DocString{"Interaction lists classified according to their types"});

    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator ... 

        YAML example [no option]:

      )EOF";
    }

    std::string operator_name() { return "check_classifier_interaction_pair"; }

    inline void execute() override final
    {
      auto& classifier = *ic;
      auto cells = grid->cells();

      for(int i=0 ; i<Classifier::typesPP ; i++)
      {
        auto& idata = classifier.get_data<InteractionType::ParticleParticle>(i);
        auto  isize = classifier.get_size(i);
        const bool is_particle_particle = i < Classifier::typesParticles;
        for(size_t j=0 ; j<isize ; j++)
        {
          size_t cellId = idata.cell_i[j]; 
          size_t particlePosition = idata.p_i[j]; 
          auto& cell = cells[cellId];
          if( particlePosition >= cell.size() )
          {
            color_log::warning(operator_name(), "Details -> wave: " + std::to_string(i) 
                + " position in the classifier: "  + std::to_string(j) 
                + " looking for the cell: " + std::to_string(cellId) 
                + " at the position: " + std::to_string(particlePosition));
            color_log::error(operator_name(), "The first part of the interaction points to a location in storage that does not exist or no longer exists.");
          }
          if( is_particle_particle )
          {
            size_t cellId = idata.cell_j[j]; 
            size_t particlePosition = idata.p_j[j]; 
            auto& cell = cells[cellId];
            if( particlePosition >= cell.size() )
            {
              color_log::warning(operator_name(), "Details -> wave: " + std::to_string(i) 
                  + " position in the classifier: " + std::to_string(j) 
                  + " looking for the cell: " + std::to_string(cellId) 
                  + " at the position: " + std::to_string(particlePosition));
              color_log::error(operator_name(), "The second part of the interaction points to a location in storage that does not exist or no longer exists.");
            }
          }
        }
      }

      // InnerBond
      {
        size_t i = Classifier::InnerBondTypeId;
        auto& idata = classifier.get_data<InteractionType::InnerBond>(i);
        auto  isize = classifier.get_size(i);
        const bool is_particle_particle = true;
        for(size_t j=0 ; j<isize ; j++)
        {
          size_t cellId = idata.cell_i[j];
          size_t particlePosition = idata.p_i[j];
          auto& cell = cells[cellId];
          if( particlePosition >= cell.size() )
          {
            color_log::warning(operator_name(), "Details -> wave: " + std::to_string(i) 
                + " position in the classifier: " + std::to_string(j) 
                + " looking for the cell: " + std::to_string(cellId) 
                + " at the position: " + std::to_string(particlePosition));
            color_log::error(operator_name(), "The first part of the interaction points to a location in storage that does not exist or no longer exists.");
          }
          if( is_particle_particle )
          {
            size_t cellId = idata.cell_j[j];
            size_t particlePosition = idata.p_j[j];
            auto& cell = cells[cellId];
            if( particlePosition >= cell.size() )
            {
              color_log::warning(operator_name(), "Details -> wave: " + std::to_string(i) 
                  + " position in the classifier: " + std::to_string(j) 
                  + " looking for the cell: " + std::to_string(cellId) 
                  + " at the position: " + std::to_string(particlePosition));
              color_log::error(operator_name(), "The second part of the interaction points to a location in storage that does not exist or no longer exists.");
            }
          }
        }
      }      

      color_log::highlight(operator_name(), "The “pair” parts (i.e., interaction_pair) of the interactions in the classifier define existing locations in the grid.");
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(check_classifier_interaction_pair) 
  { 
    OperatorNodeFactory::instance()->register_factory("check_classifier_interaction_pair", make_grid_variant_operator<CheckClassifierInteractionPair>); 
  }
} // namespace exaDEM
