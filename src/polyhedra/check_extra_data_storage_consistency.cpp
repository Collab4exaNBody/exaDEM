#include <exanb/extra_storage/check_extra_data_storage_consistency.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>

namespace exaDEM
{
	using namespace exanb;

	template<class GridT> using CheckInfoInteractionConsistencyTmpl = CheckInfoConsistency<GridT, GridCellParticleInteraction>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "check_info_interaction_consistency", make_grid_variant_operator< CheckInfoInteractionConsistencyTmpl > );
	}
}
