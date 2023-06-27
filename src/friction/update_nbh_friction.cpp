//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/operator.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/log.h>
#include <exanb/core/cpp_utils.h>

#include <yaml-cpp/yaml.h>
#include <exanb/core/quantity_yaml.h>

#include <exanb/core/config.h> // for MAX_PARTICLE_NEIGHBORS
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/compute/compute_pair_singlemat.h>

#include <onika/memory/allocator.h> // for DEFAULT_ALIGNMENT
#include <exaDEM/neighbor_friction.h>

namespace exaDEM
{
  using namespace exanb;

  // Functor to check correctness of pair friction data
  template<class GridT>
  struct alignas(onika::memory::DEFAULT_ALIGNMENT) UpdateNbhFrictionOp 
  {
    GridCellParticleNeigborFriction& nbh_friction;
    double rc2;
    
    template<class CellParticlesT, class FrictionT>
    ONIKA_HOST_DEVICE_FUNC inline void operator ()
      (      
      const Vec3d& dr,
      double d2,
      CellParticlesT* cells,
      size_t cell_b,
      size_t p_b,
      FrictionT& friction
      ) const
      {
        if( d2 < rc2) friction.m_friction = Vec3d{d2,0.,rc2};
        else friction.m_friction = Vec3d{0.,0.,0.};
      }
  };

}

namespace exanb
{
  // specialize functor traits to allow Cuda execution space
  template<class GridT>
  struct ComputePairTraits< exaDEM::UpdateNbhFrictionOp<GridT> >
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool ComputeBufferCompatible = false;
    static inline constexpr bool BufferLessCompatible = true;
    static inline constexpr bool CudaCompatible = true;
  };
}

namespace exaDEM
{

  template< class GridT >
  class UpdateNbhFriction : public OperatorNode
  {
    // ========= I/O slots =======================
    ADD_SLOT( exanb::GridChunkNeighbors    , chunk_neighbors   , INPUT , exanb::GridChunkNeighbors{} , DocString{"neighbor list"} );
    ADD_SLOT( GridCellParticleNeigborFriction , nbh_friction , INPUT_OUTPUT , OPTIONAL, DocString{"Neighbor particle friction term"} );
    ADD_SLOT( GridT                 , grid              , INPUT_OUTPUT );
    ADD_SLOT( Domain                , domain            , INPUT , REQUIRED );
    ADD_SLOT( double                , rcut              , INPUT , REQUIRED );
    ADD_SLOT( double                , friction_rcut     , INPUT , REQUIRED );

    // cell particles array type
    using CellParticles = typename GridT::CellParticles;

    // attributes processed during computation
    using ComputeFields = FieldSet< >;
    static constexpr ComputeFields compute_fields {};

  public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        				This operator figures out the neighbor particles and update the corresponding values.
				        )EOF";
		}

    // Operator execution
    inline void execute () override final
    {
      assert( chunk_neighbors->number_of_cells() == grid->number_of_cells() );

      if( grid->number_of_cells() == 0 ) { return; }

      //const IJK dims = grid->dimension();
      //const int gl = grid->ghost_layers();

      bool has_friction = false;
      if( nbh_friction.has_value() ) has_friction = ! nbh_friction->m_cell_friction.empty();
      if( ! has_friction ) return;

      ldbg << "update nbh friction"<< std::endl;
      
      const double frc = *friction_rcut;
      const double frc2 = frc*frc;
      
      ComputePairOptionalLocks<false> cp_locks {};
      exanb::GridChunkNeighborsLightWeightIt<false> nbh_it{ *chunk_neighbors };
      UpdateNbhFrictionOp<GridT> update_op { *nbh_friction , frc2 };
      ParticleNeighborFrictionIterator cp_friction{ nbh_friction->m_cell_friction.data() };
      
      if( domain->xform_is_identity() )
      {
        auto optional = make_compute_pair_optional_args( nbh_it, cp_friction, NullXForm{}, cp_locks );
        compute_pair_singlemat( *grid, *rcut, false /*no ghost*/, optional, make_default_pair_buffer(), update_op , compute_fields );
      }
      else
      {
        auto optional = make_compute_pair_optional_args( nbh_it, cp_friction , LinearXForm{ domain->xform() }, cp_locks );
        compute_pair_singlemat( *grid, *rcut, false /*no ghost*/, optional, make_default_pair_buffer(), update_op , compute_fields );
      }

    }

  };

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {  
    OperatorNodeFactory::instance()->register_factory( "update_nbh_friction" , make_grid_variant_operator< UpdateNbhFriction > );
  }

}


