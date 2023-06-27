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
#include <exaDEM/neighbor_friction.h>

namespace exaDEM
{
  using namespace exanb;

  template<class GridT>
  class MovePaticlesWithFriction : public OperatorNode
  { 
    using ParticleT = typename exanb::MoveParticlesHelper<GridT>::ParticleT;
    using ParticleVector = typename exanb::MoveParticlesHelper<GridT>::ParticleVector;
    using MovePaticlesScratch = typename exanb::MoveParticlesHelper<GridT>::MovePaticlesScratch;

    ADD_SLOT( Domain , domain , INPUT );
    ADD_SLOT( GridT , grid , INPUT_OUTPUT );
    ADD_SLOT( ParticleVector , otb_particles , OUTPUT );
    
    ADD_SLOT( GridCellParticleNeigborFriction , nbh_friction , INPUT_OUTPUT , GridCellParticleNeigborFriction{} , DocString{"Neighbor particle friction term"} );
    ADD_SLOT( ParticleNeighborFrictionCellMoveBuffer , nbh_friction_otb , INPUT_OUTPUT, ParticleNeighborFrictionCellMoveBuffer{} , DocString{"friction data of particles moving outside the box"} );
    
    ADD_SLOT( MovePaticlesScratch, move_particles_scratch , PRIVATE );

  public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        				This operator moves particles and friction data across cells.
				        )EOF";
		}

		inline void execute () override final
		{
			ldbg << "MovePaticlesWithFriction: nbh_friction size = " << nbh_friction->m_cell_friction.size()<<std::endl;
			nbh_friction_otb->clear();
			ParticleNeighborFrictionGridMoveBuffer nbh_friction_opt_buffer = { nbh_friction->m_cell_friction , *nbh_friction_otb };
			exanb::move_particles_across_cells( ldbg, *domain, *grid, *otb_particles, *move_particles_scratch, nbh_friction_opt_buffer );

			size_t n_friction_otb = nbh_friction_otb->number_of_particles();
			ldbg << "particle friction otb particles = "<< n_friction_otb <<std::endl;
#     ifndef NDEBUG
			for(size_t i=0;i<n_friction_otb;i++)
			{
				ldbg << "\totb friction "<<i<<" : id="<<nbh_friction_otb->particle_id(i)<<", nb_pairs="<<nbh_friction_otb->particle_pair_count(i)<<std::endl;
			}
#     endif      
		}    
	};

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "move_particles_friction", make_grid_variant_operator< MovePaticlesWithFriction > );
	}

}

