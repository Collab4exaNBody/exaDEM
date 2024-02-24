#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <memory>

#include <exaDEM/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/migration_test.hpp>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT >
		>
		class CheckInfoInteractionConsistency : public OperatorNode
		{
			ADD_SLOT( GridT       , grid              , INPUT_OUTPUT , REQUIRED );
			ADD_SLOT( GridCellParticleInteraction , grid_interaction  , INPUT , DocString{"Grid of Extra Data Storage (per cell)"} );


			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
					"This opertor checks if for each particle information the offset and size are correct"
				        )EOF";
			}

			inline void execute () override final
			{
				if( grid->number_of_cells() == 0 ) { return; }
				auto & cell_interactions = grid_interaction->m_data;

				for(size_t current_cell = 0 ; current_cell < cell_interactions.size() ; current_cell++)
				{
					auto storage = cell_interactions[current_cell];
					size_t n_particles_stored = storage.number_of_particles();
					auto* info_ptr = storage.m_info.data();
					bool is_okay = migration_test::check_info_consistency( info_ptr, n_particles_stored);
					assert(is_okay && "CheckInteractionConsistency");
				}
		}
};

template<class GridT> using CheckInfoInteractionConsistencyTmpl = CheckInfoInteractionConsistency<GridT>;

// === register factories ===  
CONSTRUCTOR_FUNCTION
{
	OperatorNodeFactory::instance()->register_factory( "check_info_interaction_consistency", make_grid_variant_operator< CheckInfoInteractionConsistencyTmpl > );
}
}

