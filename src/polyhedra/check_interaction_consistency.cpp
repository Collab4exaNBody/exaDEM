#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>

#include <memory>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/migration_test.hpp>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT >
		>
		class CheckInteractionConsistency : public OperatorNode
		{
			ADD_SLOT( GridT       , grid              , INPUT_OUTPUT , REQUIRED );
			ADD_SLOT( GridCellParticleInteraction , grid_interaction  , INPUT , DocString{"Interaction list"} );


			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
					"This opertor checks if a interaction related to a particle contains its particle id. (i.e. , I_id(i,j), id == item.id_i || item.id_j)"
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
					auto* data_ptr = storage.m_data.data();
					[[maybe_unused]] bool is_okay = interaction_test::check_extra_interaction_storage_consistency( n_particles_stored, info_ptr, data_ptr);
					assert(is_okay && "CheckInteractionConsistency");
				}
		}
};

template<class GridT> using CheckInteractionConsistencyTmpl = CheckInteractionConsistency<GridT>;

// === register factories ===  
CONSTRUCTOR_FUNCTION
{
	OperatorNodeFactory::instance()->register_factory( "check_interaction_consistency", make_grid_variant_operator< CheckInteractionConsistencyTmpl > );
}
}
