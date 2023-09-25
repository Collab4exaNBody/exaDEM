#pragma once

#include <exanb/fields.h>
#include <exanb/core/field_set_utils.h>
//#include <exaStamp/particle_species/particle_specie.h>
#include <onika/soatl/field_tuple.h>
#include <exanb/io/sim_dump_io.h>
#include <iostream>

#include <exaDEM/neighbor_friction.h>

#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>
#include <exanb/core/quaternion_operators.h>
#include <exanb/core/quaternion_stream.h>

namespace exaDEM
{
  using namespace exanb;

  template< class GridT , class DumpFieldSet >
  struct ParticleDumpFilter
  {
    using GridFieldSet = typename GridT::Fields;
    using TupleT = onika::soatl::FieldTupleFromFieldIds< DumpFieldSet >;
    using StorageType = TupleT; //onika::soatl::FieldTupleFromFieldIds< FieldSet<fids...,field::nbhfriction,field::drvfriction> >;

    GridCellParticleNeigborFriction& nbh_friction;
    GridT& grid;
    ParticleFrictionReadHelper friction_read_helper;
    double scale_cell_size = 1.0;
    bool enable_friction = true;

    // for forward compatibility with dump_reader_allow_initial_position_xform branch
    inline void process_domain(Domain& domain , Mat3d& pos_read__xform)
    {
      pos_read__xform = make_identity_matrix();
      this->process_domain(domain);
    }

    inline void process_domain(Domain& domain)
    {
      if( scale_cell_size != 1.0 )
      {
        Vec3d dom_size = domain.bounds_size();
        double desired_cell_size = domain.cell_size() * scale_cell_size;
        IJK grid_dims = make_ijk( Vec3d{0.5,0.5,0.5} + ( dom_size / desired_cell_size ) ); // round to nearest
        domain.set_grid_dimension( grid_dims );
        domain.set_cell_size( desired_cell_size );
        domain.set_bounds( { domain.origin() , domain.origin() + ( grid_dims * desired_cell_size ) } );
      }
    }

    inline void update_sats( const StorageType & ) { }
    inline void initialize_write() { }
    inline void finalize_write() { }

    inline void initialize_read()
    {
      friction_read_helper.initialize( grid.number_of_cells() );
    }

    inline void finalize_read()
    {
      if( enable_friction && ! friction_read_helper.m_out_friction.empty() )
      {
        auto cells = grid.cells();
        auto particle_id_func = [cells]( size_t cell_idx, size_t p_idx ) -> uint64_t { return cells[cell_idx][field::id][p_idx]; } ;
        friction_read_helper.finalize( nbh_friction , particle_id_func );
      }
    }

    inline void read_optional_data_from_stream( const uint8_t* stream_start , size_t stream_size )
    {
      if( enable_friction )
      {
        friction_read_helper.read_from_stream( stream_start , stream_size );
      }
    }

    inline void append_cell_particle( size_t cell_idx, size_t p_idx )
    {
      if( enable_friction )
      {
        auto cells = grid.cells();
        friction_read_helper.append_cell_particle( cell_idx , p_idx , cells[cell_idx][field::id][p_idx] );
      }
    }

    inline size_t optional_cell_data_size(size_t cell_index)
    {
      if( enable_friction )
      {
        assert( cell_index < nbh_friction.m_cell_friction.size() );
        return nbh_friction.m_cell_friction[cell_index].storage_size();
      }
      else { return 0; }
    }

    inline const uint8_t* optional_cell_data_ptr(size_t cell_index)
    {
      if( enable_friction )
      {
        assert( cell_index < nbh_friction.m_cell_friction.size() );
        return nbh_friction.m_cell_friction[cell_index].storage_ptr();
      }
      else
      {
        return nullptr;
      }
    }

    template<class WriteFuncT>
    inline size_t write_optional_header( WriteFuncT write_func )
    {
      return 0;
    }
    
    template<class ReadFuncT>
    inline size_t read_optional_header( ReadFuncT read_func )
    {      
      return 0;
    }

    inline StorageType encode( const TupleT & tp )
    {
      update_sats( tp );
      return tp;
    }

    inline TupleT decode( const StorageType & stp )
    {
      update_sats( stp );
      return stp;
    }

  };

}

