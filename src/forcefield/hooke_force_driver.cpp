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
#include <exanb/compute/compute_cell_particles.h>
#include <onika/memory/allocator.h> // for DEFAULT_ALIGNMENT
#include <exaDEM/drivers.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/hooke_force_parameters.h>
#include <exaDEM/cylinder_wall.h>
#include <exaDEM/compute_wall.h>


namespace exaDEM
{
	using namespace exanb;

	template<	class GridT, 	class = AssertGridHasFields< GridT, field::_radius, field::_vx,field::_vy,field::_vz, field::_mass, field::_fx ,field::_fy,field::_fz, field::_vrot, field::_mom, field::_homothety >>
		class HookeForceDriver : public OperatorNode
	{
		// ========= I/O slots =======================
		ADD_SLOT( HookeParams  , config   , INPUT , REQUIRED );
		ADD_SLOT( double       , rcut_max , INPUT_OUTPUT , 0.0 );
		ADD_SLOT( Drivers      , drivers  , INPUT , DocString{"List of Drivers"});
		ADD_SLOT( bool         , ghost    , INPUT , false );
		ADD_SLOT( GridT        , grid     , INPUT_OUTPUT );
		ADD_SLOT( Domain       , domain   , INPUT , REQUIRED );
		ADD_SLOT( double       , dt       , INPUT , REQUIRED );

		// cell particles array type
		using CellParticles = typename GridT::CellParticles;

		// attributes processed during computation
		using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom, field::_friction >;
		static constexpr ComputeFields compute_fields {};

		public:

		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator computes forces between a spherical particle and a driver using the Hooke law.
        )EOF";
		}
		// Operator execution
		inline void execute () override final
		{
			const double rcut = config->rcut;
			*rcut_max = std::max( *rcut_max , rcut );
			const HookeParams params = *config;
			if( grid->number_of_cells() == 0 ) { return; }


			ldbg<<"Hooke: rcut="<<rcut<<std::endl;

			// First drivers
			if ( drivers.has_value() )
			{
				auto& drvs = *drivers;
				for( size_t drvs_idx = 0 ; drvs_idx < drvs.get_size() ; drvs_idx++ )
				{
					if (drvs.type(drvs_idx) == DRIVER_TYPE::CYLINDER)
					{
						// Get driver
						Cylinder& driver = std::get<Cylinder>(drvs.data(drvs_idx)) ;
						// Define Functor
						CylinderWallFunctor func { driver.center , driver.axis , 
							driver.vrot , driver.vel, 
							driver.radius, *dt, params.m_kt, 
							params.m_kn, params.m_kr, params.m_mu, params.m_damp_rate};
						// Apply Functor for every particles
						compute_cell_particles( *grid , false , func , compute_fields , parallel_execution_context() );
					}

					if (drvs.type(drvs_idx) == DRIVER_TYPE::SURFACE)
					{
						// Get driver
						Surface& driver = std::get<Surface>(drvs.data(drvs_idx)) ;
						// Define Functor
						// vel is null
						RigidSurfaceFunctor func {driver.normal, driver.offset, 0.0, *dt, params.m_kt, params.m_kn, params.m_kr, params.m_mu, params.m_damp_rate};
						compute_cell_particles( *grid , false , func , compute_fields , parallel_execution_context() );
					}
				}
			}
		}
	};

	template<class GridT> using HookeForceDriverTmpl = HookeForceDriver<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{  
		OperatorNodeFactory::instance()->register_factory( "hooke_force_driver" , make_grid_variant_operator< HookeForceDriverTmpl > );
	}
}


