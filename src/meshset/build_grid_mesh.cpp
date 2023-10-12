#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>
#include <onika/memory/allocator.h> // for ONIKA_ASSUME_ALIGNED macro
#include <exanb/compute/compute_pair_optional_args.h>
#include <vector>
#include <iomanip>

#include <exanb/compute/compute_cell_particles.h>
#include <exaDEM/face.h>
#include <exaDEM/stl_mesh.h>

#include <mpi.h>

namespace exaDEM
{
	using namespace exanb;
	template<	class GridT, class = AssertGridHasFields< GridT >> class BuildGridSTLMeshOperator : public OperatorNode
	{
		using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz>;
		static constexpr ComputeFields compute_field_set {};
		ADD_SLOT( MPI_Comm , mpi      , INPUT , MPI_COMM_WORLD , DocString{"MPI communicator for parallel processing."});
		ADD_SLOT( GridT    , grid     , INPUT_OUTPUT , DocString{"Grid used for computations."} );
		ADD_SLOT( double   , rcut_max , INPUT , 0.0, DocString{"Maximum cutoff radius for computations. Default is 0.0."} );
		ADD_SLOT( std::vector<exaDEM::stl_mesh> , stl_collection, INPUT_OUTPUT , DocString{"Collection of meshes from stl files"});

		public:
		inline std::string documentation() const override final
		{
			return R"EOF( 
    	    			)EOF";
		}

		inline void execute () override final
		{
			auto& collection = *stl_collection;
			const double rad = *rcut_max;

			const auto cells = grid->cells();
			const size_t n_cells = grid->number_of_cells(); // nbh.size();
			const IJK dims = grid->dimension();
			const int gl = grid->ghost_layers();
			for(auto &mesh : collection)
			{
				auto& ind = mesh.indexes;
				ind.resize(n_cells);
				mesh.build_boxes();
#     pragma omp parallel
				{
					GRID_OMP_FOR_BEGIN(dims-2*gl,_,block_loc, schedule(dynamic))
					{
						IJK loc_a = block_loc + gl;
						size_t cell_a = grid_ijk_to_index( dims , loc_a );
						ind[cell_a].clear();
						auto cb = grid->cell_bounds(loc_a);
						Box bx = { cb.bmin - rad , cb.bmax + rad };

						const int n_particles = cells[cell_a].size();
						if (n_particles == 0) continue;
						mesh.update_indexes(cell_a, bx);
					}
					GRID_OMP_FOR_END
				}
			}
		};
	};

	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using BuildGridSTLMeshOperatorTemplate = BuildGridSTLMeshOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "build_grid_stl_mesh", make_grid_variant_operator< BuildGridSTLMeshOperatorTemplate > );
	}
}
