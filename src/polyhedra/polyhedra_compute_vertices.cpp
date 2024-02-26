#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/core/grid.h>
#include <memory>
#include <random>
#include <exaDEM/shapes.hpp>
#include <exaDEM/compute_vertices.hpp>


namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_type, field::_homothety, field::_orient, field::_vertices>
		>
		class PolyhedraComputeVertices : public OperatorNode
		{
    	using ComputeFields = FieldSet< field::_type, field::_rx, field::_ry, field::_rz, field::_homothety, field::_orient, field::_vertices >;
    	static constexpr ComputeFields compute_field_set {};
			ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
			ADD_SLOT( shapes  , shapes_collection, INPUT_OUTPUT , DocString{"Collection of shapes"});

			// -----------------------------------------------
			// ----------- Operator documentation ------------
			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator applies the same density to all particles. If you want to apply various densities according to their material properties, use set_densities_multiple_materials.
        )EOF";
			}

			public:
			inline void execute () override final
			{
        PolyhedraComputeVerticesFunctor func {*shapes_collection};
        compute_cell_particles( *grid , true , func , compute_field_set , parallel_execution_context() );
			}
		};

	template<class GridT> using PolyhedraComputeVerticesTmpl = PolyhedraComputeVertices<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "polyhedra_compute_vertices", make_grid_variant_operator< PolyhedraComputeVerticesTmpl > );
	}

}

