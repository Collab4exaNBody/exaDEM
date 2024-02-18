//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <memory>
#include <mpi.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>


#include <exaDEM/hooke_force_parameters.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
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
		ADD_SLOT( MPI_Comm  , mpi        , INPUT , MPI_COMM_WORLD );
		ADD_SLOT( GridT  , grid     , INPUT_OUTPUT , REQUIRED );
		ADD_SLOT( GridCellParticleInteraction , grid_interaction  , INPUT_OUTPUT , DocString{"Interaction list"} );

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
				        )EOF";
		}

		inline void execute () override final
		{
			//const auto cells = grid->cells();
			auto & cells = grid_interaction->m_data;

			int nvv(0), nve(0), nvf(0), nee(0); // interaction counters
			int an(0), anvv(0), anve(0), anvf(0), anee(0); // active interaction counters

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
			for( size_t i = 0 ; i < cells.size() ; i++ )
			{
				for( auto& item : cells[i].m_data)
				{
					if( item.type == 0 ) incr_interaction_counters(item, nvv, anvv, an); 
					if( item.type == 1 ) incr_interaction_counters(item, nve, anve, an); 
					if( item.type == 2 ) incr_interaction_counters(item, nvf, anvf, an); 
					if( item.type == 3 ) incr_interaction_counters(item, nee, anee, an); 
				}
			}

			std::vector<int> val = {nvv, nve, nvf, nee, an, anvv, anve, anvf, anee};

			int rank;
			MPI_Comm_rank(*mpi, &rank);

			if(rank == 0)
				MPI_Reduce(MPI_IN_PLACE, val.data(), val.size(), MPI_INT, MPI_SUM, 0, *mpi);
			else
				MPI_Reduce(val.data(), val.data(), val.size(), MPI_INT, MPI_SUM, 0, *mpi);

			int idx = 0 ;
			for (auto it : {&nvv, &nve, &nvf, &nee, &an, &anvv, &anve, &anvf, &anee}) *it = val[idx++];

			lout << "==================================" << std::endl; 
			lout << "* Type of interaction    : active / total " << std::endl; 
			lout << "* Number of interactions : " << an << " / " << nvv + nve + nvf + nee << std::endl; 
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

