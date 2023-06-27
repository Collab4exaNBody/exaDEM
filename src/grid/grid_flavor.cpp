#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/field_sets.h>
#include <exanb/core/grid.h>

#include <mpi.h>

namespace exaDEM
{
using namespace exanb;
  // 
  template<class GridT , class FieldSubSetT = typename GridT::field_set_t>
  struct InitGridFlavorNode : public OperatorNode
  {
    ADD_SLOT(GridT, grid, INPUT_OUTPUT );

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator initializes the DEM grid.
        )EOF";
		}
		

    inline InitGridFlavorNode()
    {
      set_profiling(false);
    }


    inline void execute () override final
    {
      if( grid->number_of_cells() == 0 )
      {
        grid->set_cell_allocator_for_fields( FieldSubSetT{} );
        grid->rebuild_particle_offsets();
      }
    }      
  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory(
      "grid_flavor_dem",
      make_compatible_operator< InitGridFlavorNode< GridFromFieldSet<DEMFieldSet> > >
      );
  }

}

