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
#include <exaDEM/neighbor_type.h>
#include <exaDEM/interaction.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/mutexes.h>

namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT >
		>
		class UpdateMutexes : public OperatorNode
		{
			using ComputeFields = FieldSet<>;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT       , grid              , INPUT_OUTPUT , REQUIRED );
			ADD_SLOT( mutexes     , locks             , INPUT_OUTPUT );


			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
				        )EOF";
			}

			inline void execute () override final
			{
				const auto& cells    = grid->cells();
				const int n_cells    = grid->number_of_cells();
				mutexes & cell_locks = *locks;
				locks -> resize (n_cells) ;
//				int n_locks = 0;
//#pragma omp parallel for reduction (+:n_locks)
#pragma omp parallel for
				for (int c = 0 ; c < n_cells ; c++)
				{
					auto& current_locks = cell_locks.get_mutexes (c); 
					current_locks . resize (cells[c].size());
//          n_locks += cells[c].size();
				}	

//				std::cout << " number of locks: " << n_locks << " n cells : " << n_cells << std::endl;
				cell_locks . initialize();

			}
		};

	template<class GridT> using UpdateMutexesTmpl = UpdateMutexes<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "update_mutexes", make_grid_variant_operator< UpdateMutexesTmpl > );
	}
}

