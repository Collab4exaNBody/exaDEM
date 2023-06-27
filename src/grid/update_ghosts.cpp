#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/log.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/grid.h>
#include <exanb/grid_cell_particles/grid_cell_values.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/particle_id_codec.h>
#include <exanb/field_sets.h>
#include <exanb/core/check_particles_inside_cell.h>

#include <onika/soatl/field_tuple.h>

#include <vector>
#include <string>
#include <list>
#include <algorithm>
#include <tuple>

#include <mpi.h>
#include <exanb/mpi/update_ghost_utils.h>
#include <exanb/mpi/ghosts_comm_scheme.h>
#include <exanb/mpi/update_ghosts.h>
#include <exanb/mpi/data_types.h>

namespace exaDEM
{
  using namespace exanb;
  using namespace UpdateGhostsUtils;

  // === register factory ===
  //template<typename GridT> using UpdateGhostsRQ = UpdateGhostsNode< GridT , FieldSet<field::_rx, field::_ry, field::_rz , field::_radius, field::_vrot,  field::_orient > , false >;
  template<typename GridT> using UpdateGhostsRQ = UpdateGhostsNode< GridT , FieldSet<field::_rx, field::_ry, field::_rz , field::_vx, field::_vy, field::_vz, field::_radius, field::_mass, field::_vrot,  field::_orient > , false >;

  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "ghost_update_rq",     make_grid_variant_operator<UpdateGhostsRQ> );
  }

}

