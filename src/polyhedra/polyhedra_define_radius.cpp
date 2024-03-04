#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <memory>
#include <random>
#include <exaDEM/shape/shapes.hpp>

namespace exaDEM
{
  using namespace exanb;
  struct UpdateRadiusPolyhedronFunctor
  {
    const ParticleRegionCSGShallowCopy region; /**< Shallow copy of a particle region. */
		onika::memory::CudaMMVector<double> list_of_radius; /** filled in PolyhedraDefineRadius. */

    ONIKA_HOST_DEVICE_FUNC inline void operator () (uint8_t type, double& radius) const
    {
			radius = list_of_radius[type];
    }
    // If the region feature is activated
    ONIKA_HOST_DEVICE_FUNC inline void operator () (const double rx, const double ry, const double rz, const uint64_t id, uint8_t type, double& radius) const
    {
      Vec3d r = {rx,ry,rz};
      if( region.contains( r , id ) )
			{
				radius = list_of_radius[type];
			}
		}
	};
}

namespace exanb
{
  template<> struct ComputeCellParticlesTraits<exaDEM::UpdateRadiusPolyhedronFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };
}




namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_radius>
		>
		class PolyhedraDefineRadius : public OperatorNode
		{
      using ComputeFields = FieldSet<field::_type, field::_radius>;
      using ComputeRegionFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_type, field::_radius>;
      static constexpr ComputeFields compute_field_set {};
      static constexpr ComputeRegionFields compute_region_field_set {};


			ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
			ADD_SLOT( shapes  , shapes_collection, INPUT_OUTPUT , DocString{"Collection of shapes"});
			ADD_SLOT( double                , rcut_max          , INPUT_OUTPUT , 0.0 );
      ADD_SLOT( ParticleRegions   , particle_regions , INPUT , OPTIONAL );
      ADD_SLOT( ParticleRegionCSG , region           , INPUT , OPTIONAL );

			// -----------------------------------------------
			// ----------- Operator documentation ------------
			inline std::string documentation() const override final
			{
				return R"EOF(
        )EOF";
			}

			public:
			inline void execute () override final
			{
				// compute the biggest radius for each polyhedron
				const shapes shps = *shapes_collection;
				const size_t size = shps.get_size();
				onika::memory::CudaMMVector<double> r;
				r.resize(size);
				double rmax=0;
				for(size_t i = 0 ; i < size ; i++)
				{
					double rad_max = shps[i]->compute_max_rcut();
					r[i] = rad_max; 
					rmax = std::max(rmax, 2 *  rad_max); // r * maxrcut
				}
				*rcut_max = rmax; 

				// now, fill the radius field
        if( region.has_value() )
        {
        	ParticleRegionCSGShallowCopy prcsg = *region;
        	UpdateRadiusPolyhedronFunctor func = {prcsg, r};
          if( !particle_regions.has_value() )
          {
            fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
          }

          if( region->m_nb_operands==0 )
          {
            ldbg << "rebuild CSG from expr "<< region->m_user_expr << std::endl;
            region->build_from_expression_string( particle_regions->data() , particle_regions->size() );
          }
          compute_cell_particles( *grid , false , func , compute_region_field_set , parallel_execution_context() );
        }
        else
        {
        	ParticleRegionCSGShallowCopy prcsg;
        	UpdateRadiusPolyhedronFunctor func = {prcsg, r};
          compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
        }
			}

		};

	template<class GridT> using PolyhedraDefineRadiusTmpl = PolyhedraDefineRadius<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "polyhedra_define_radius", make_grid_variant_operator< PolyhedraDefineRadiusTmpl > );
	}

}

