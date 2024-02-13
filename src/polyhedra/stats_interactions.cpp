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
    , class = AssertGridHasFields< GridT >
    >
  class StatsInteractions : public OperatorNode
  {
    ADD_SLOT( GridT  , grid     , INPUT_OUTPUT , REQUIRED );
    ADD_SLOT( std::vector<Interaction> , nbh_interactions , INPUT_OUTPUT , DocString{"TODO"} );

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
				        )EOF";
		}

		inline void execute () override final
		{
			//const auto cells = grid->cells();
			auto & interactions = *nbh_interactions;

			int nvv(0), nve(0), nvf(0), nee(0);
			int an(0), anvv(0), anve(0), anvf(0), anee(0); // active interactions

			const exanb::Vec3d null = {0.,0.,0.};

			auto incr_interaction_counters = [null] (const Interaction& I, int& count, int& active_count, int& active_global_count) -> void
			{
					count ++;
					if( I.friction != null ) 
					{
						active_count ++;
						active_global_count ++;
					}
			};

#pragma omp parallel for reduction(+:nvv, nve, nvf, nee, an, anvv, anve, anvf, anee)
			for( size_t i = 0 ; i < interactions.size() ; i++ )
			{
				const Interaction & item = interactions[i];
				if( item.type == 0 ) incr_interaction_counters(item, nvv, anvv, an); 
				if( item.type == 1 ) incr_interaction_counters(item, nve, anve, an); 
				if( item.type == 2 ) incr_interaction_counters(item, nvf, anvf, an); 
				if( item.type == 3 ) incr_interaction_counters(item, nee, anee, an); 
			}

			lout << "==================================" << std::endl; 
			lout << "* Type of interaction    : active / total " << std::endl; 
			lout << "* Number of interactions : " << an << " / " << interactions.size() << std::endl; 
			lout << "* Vertex - Vertex        : " << anvv << " / " << nvv << std::endl; 
			lout << "* Vertex - Edge          : " << anve << " / " << nve << std::endl; 
			lout << "* Vertex - Face          : " << anvf << " / " << nvf << std::endl; 
			lout << "* Edge   - Edge          : " << anee << " / " << nee << std::endl; 
			lout << "==================================" << std::endl; 

		}
	};

	template<class GridT> using StatsInteractionsTmpl = StatsInteractions<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "stats_interactions", make_grid_variant_operator< StatsInteractionsTmpl > );
	}

}

