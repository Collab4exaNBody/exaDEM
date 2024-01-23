//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

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
#include <exaDEM/interaction.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>

namespace exaDEM
{
  using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_radius >
    >
  class MigrateInteractions : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet< field::_vrot, field::_arot >;
    static constexpr ComputeFields compute_field_set {};

    ADD_SLOT( GridT  , grid     , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( exanb::GridChunkNeighbors , chunk_neighbors   , INPUT , OPTIONAL , DocString{"neighbor list"} );
    ADD_SLOT( std::vector<Interaction> , nbh_interactions , INPUT_OUTPUT , DocString{"TODO"} );
    ADD_SLOT( shapes , shapes_collection, INPUT , DocString{"Collection of shapes"});
		ADD_SLOT(double , rcut_inc          , INPUT , 0.0 , DocString{"value added to the search distance to update neighbor list less frequently. in physical space"} );

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
				        )EOF";
		}

		inline void execute () override final
		{
			const auto cells = grid->cells();

			const IJK dims = grid->dimension();
			const int gl = grid->ghost_layers();

			auto & interactions = *nbh_interactions;

			auto & shps = *shapes_collection;
			double rVerlet = *rcut_inc;

			std::vector<Interaction> to_move;

			// remove inactive interactions : 
			exanb::Vec3d null = {0,0,0};
			int last = interactions.size() - 1;
			for(int i = last ; i >= 0 ; i--)
			{
				if(interactions[i].mom == null && interactions[i].friction == null)
				{
					interactions[i] = interactions[last--];
				}
			}

			// local
			for(int i = last ; i >= 0 ; i--)
			{
				const cell_i = interactions[i].cell_i;
				const p_i = interactions[i].idx_i;
				const double* __restrict__ rx = cells[cell_i][field::rx];
				const double* __restrict__ ry = cells[cell_i][field::ry];
				const double* __restrict__ rz = cells[cell_i][field::rz];
				Vec3d r{rx[p_i],ry[p_i],rz[p_i]};
				IJK dst_loc = domain_periodic_location( domain , r ) - grid_offset;
				
			}

			update_interaction(interaction)


				//	if( ! chunk_neighbors.has_value() ) return;
		}
	};

	template<class GridT> using MigrateInteractionsTmpl = MigrateInteractions<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "migrate_interactions", make_grid_variant_operator< MigrateInteractionsTmpl > );
	}

}

