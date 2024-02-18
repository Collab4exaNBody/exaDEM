#include <exanb/core/operator.h>
#include <exanb/core/domain.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/check_particles_inside_cell.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/log.h>
#include <exanb/core/thread.h>

#include <vector>

#include <exanb/grid_cell_particles/move_particles_across_cells.h>
#include <exaDEM/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>


namespace exaDEM
{
  using namespace exanb;

  template<class GridT>
  class MovePaticlesWithInteraction : public OperatorNode
  { 
    using ParticleT = typename exanb::MoveParticlesHelper<GridT>::ParticleT;
    using ParticleVector = typename exanb::MoveParticlesHelper<GridT>::ParticleVector;
    using MovePaticlesScratch = typename exanb::MoveParticlesHelper<GridT>::MovePaticlesScratch;

    ADD_SLOT( Domain , domain , INPUT );
    ADD_SLOT( GridT , grid , INPUT_OUTPUT );
    ADD_SLOT( ParticleVector , otb_particles , OUTPUT );
    
    ADD_SLOT( GridCellParticleInteraction , grid_interaction  , INPUT_OUTPUT , DocString{"Interaction list"} );
    ADD_SLOT( InteractionCellMoveBuffer , buffer_interaction , INPUT_OUTPUT, InteractionCellMoveBuffer{} , DocString{"interaction data of particles moving outside the box"} );
 
    ADD_SLOT( MovePaticlesScratch, move_particles_scratch , PRIVATE );

  public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        				This operator moves particles and interaction data across cells.
				        )EOF";
		}

		inline void execute () override final
		{
			buffer_interaction->clear();
			InteractionGridMoveBuffer interaction_opt_buffer = { grid_interaction->m_data , *buffer_interaction };
			exanb::move_particles_across_cells( ldbg, *domain, *grid, *otb_particles, *move_particles_scratch, interaction_opt_buffer );
		}    
	};

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "move_particles_interaction", make_grid_variant_operator< MovePaticlesWithInteraction > );
	}

}

