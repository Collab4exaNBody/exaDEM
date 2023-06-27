#include <exanb/core/basic_types_yaml.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/log.h>
#include <exanb/core/domain.h>

#include <iostream>
#include <fstream>
#include <string>

#include <exanb/io/sim_dump_writer.h>
#include <exaDEM/neighbor_friction.h>
#include <exaDEM/particle_dump_filter.h>

namespace exaDEM
{
  using namespace exanb;

  template<class GridT>
  class SimDumpWriteParticles : public OperatorNode
  {
    ADD_SLOT( MPI_Comm    , mpi             , INPUT );
    ADD_SLOT( GridT       , grid     , INPUT );
    ADD_SLOT( Domain      , domain   , INPUT );
    ADD_SLOT( std::string , filename , INPUT );
    ADD_SLOT( long        , timestep      , INPUT , DocString{"Iteration number"} );
    ADD_SLOT( double      , physical_time , INPUT , DocString{"Physical time"} );
    ADD_SLOT( long        , compression_level , INPUT , 0 , DocString{"Zlib compression level"} );

    ADD_SLOT( GridCellParticleNeigborFriction , nbh_friction  , INPUT , GridCellParticleNeigborFriction{} , DocString{"Neighbor particle friction term"} );
//    ADD_SLOT( ParticleSpecies , species , INPUT_OUTPUT , REQUIRED );

  public:
    inline void execute () override final
    {
      using DumpFieldSet = FieldSet<field::_rx,field::_ry,field::_rz, field::_vx,field::_vy,field::_vz, field::_mass, field::_homothety, field::_radius, field::_orient , field::_mom , field::_vrot , field::_arot, field::_inertia , field::_id , field::_shape >;
      exanb::write_dump( *mpi, ldbg, *grid, *domain, *physical_time, *timestep, *filename, *compression_level, DumpFieldSet{} , ParticleDumpFilter<GridT,DumpFieldSet>{*nbh_friction,*grid} );
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "write_dump_particles" , make_grid_variant_operator<SimDumpWriteParticles> );
  }

}

