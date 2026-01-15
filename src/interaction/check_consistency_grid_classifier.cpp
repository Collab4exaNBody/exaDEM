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
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <memory>
#include <array>

#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/compute_contact_force.h>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/classifier/interactionSOA.hpp>
#include <exaDEM/classifier/interactionAOS.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/classifier/classifier.hpp>

namespace exaDEM
{
  using namespace exanb;
  class CheckConsistencyGridClassifier : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_vrot, field::_arot>;
    static constexpr ComputeFields compute_field_set{};

		ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, REQUIRED,
				DocString{"List of particle interactions within each grid cell"});
		ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT,
				DocString{"Interaction lists, classified and grouped by interaction type"});
		ADD_SLOT(bool, verbosity, PRIVATE, false,
				DocString{"Enable detailed messages to verify consistency between the classifier and GridCellParticleInteraction"});

		public:
		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator checks whether the number of active interactions is consistent between the Classifier and the GridCellParticleInteraction.

        YAML example:

          - check_consistency_grid_classifier:
             verbosity: true
        )EOF";
		}

		std::string operator_name() { return "check_consistency_grid_classifier"; }

		inline void execute() override final
		{
			auto& classifier = *ic;
			auto& grid = *ges;

			struct InteractionCounter 
			{
				std::array<uint64_t, Classifier<InteractionSOA>::types> counts_by_type = {};
			};

			auto extract_data = [] (InteractionCounter& input_counter, const Interaction& I) -> void
			{
				if(I.is_active())
				{
					input_counter.counts_by_type[I.type] += 1;
				}
			};

			InteractionCounter classifier_side;
			InteractionCounter grid_side;
			// get data from classifier;

			// Note : Sequential 
			// Classifier
			for(int i = 0; i < Classifier<InteractionSOA>::types ; i++)
			{
				auto [data, size] = classifier.get_info(i);
				for(size_t j = 0; j < size ; j++)
				{
					Interaction I = data[j];
					extract_data(classifier_side, I);
				}
			}
			// GridCellParticleInteraction
			auto &ces = grid.m_data;
			for(size_t i = 0; i < ces.size(); i++)
			{
				auto &interactions = ces[i];
				const unsigned int n_interactions_in_cell = interactions.m_data.size();
				exaDEM::Interaction *const __restrict__ data_ptr = interactions.m_data.data();
				for (size_t it = 0; it < n_interactions_in_cell; it++)
				{
					const Interaction &I = data_ptr[it];
					extract_data(grid_side, I);
				}
			}

			// Verify values
			bool error = false;
			for(int i = 0; i < Classifier<InteractionSOA>::types ; i++)
			{
				if(grid_side.counts_by_type[i] != classifier_side.counts_by_type[i])
				{
					std::string msg = "Mismatch in the number of interactions for type ";
					msg += std::to_string(i);
					msg += ": the classifier reports ";
					msg += std::to_string(classifier_side.counts_by_type[i]);
					msg += " active interactions, while GridCellParticleInteraction reports ";
					msg += std::to_string(grid_side.counts_by_type[i]);
					msg += ".";
					color_log::warning(operator_name(), msg); 
          error = true;
				}
			} 
			if(error) color_log::error(operator_name(), "The number of active interactions between the classifier and the GridCellParticleInteraction is inconsistent. Ensure that this operator is invoked after the unclassify operator and before any compute_force call.");
			if(*verbosity) color_log::highlight(operator_name(), "The classifier and the GridCellParticleInteraction contain the same active interactions. To disable this message, set the verbosity slot to false.");
		}
	};

	// === register factories ===
	ONIKA_AUTORUN_INIT(check_consistency_grid_classifier) 
	{ 
		OperatorNodeFactory::instance()->register_factory("check_consistency_grid_classifier", make_simple_operator<CheckConsistencyGridClassifier>); 
	}
} // namespace exaDEM
