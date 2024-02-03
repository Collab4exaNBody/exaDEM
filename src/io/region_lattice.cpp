#include <exanb/grid_cell_particles/lattice_generator.h>

namespace exaDEM
{
	using namespace exanb;
  template<class GridT> using RegionLatticeTmpl = exanb::RegionLattice<GridT,field::_type>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory("lattice", make_grid_variant_operator< RegionLatticeTmpl >);
  }

}
