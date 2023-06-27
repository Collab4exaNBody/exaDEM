//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/compute/compute_cell_particles.h>
#include <memory>
#include <include/exaDEM/common_compute_kernels.h>
#include <exaDEM/compute_hooke_force.h>
#include <exaDEM/cylinder_wall.h>

namespace exaDEM
{
  using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_fx,field::_fy,field::_fz >
		>
		class CylinderWall : public OperatorNode
		{
			static constexpr Vec3d default_axis = { 1.0, 0.0, 1.0 };
			static constexpr Vec3d null= { 0.0, 0.0, 0.0 };
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom, field::_friction >;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT   , grid  			, INPUT_OUTPUT );
			ADD_SLOT( Vec3d   , center 			, INPUT 	, REQUIRED 	, DocString{"Center of the cylinder"});
			ADD_SLOT( Vec3d   , axis   			, INPUT 	, default_axis 	, DocString{"Define the plan of the cylinder"});
			ADD_SLOT( Vec3d   , cylinder_angular_velocity 	, INPUT 	, null		, DocString{"Angular velocity of the cylinder, default is 0 m.s-"});
			ADD_SLOT( Vec3d   , cylinder_velocity 		, INPUT 	, null		, DocString{"Cylinder velocity, could be used in 'expert mode'"});
			ADD_SLOT( double  , radius 			, INPUT 	, REQUIRED 	, DocString{"Radius of the cylinder, positive and should be superior to the biggest sphere radius in the cylinder"});
			ADD_SLOT( double  , dt                		, INPUT 	, REQUIRED 	, DocString{"Timestep of the simulation"});
			ADD_SLOT( double  , kt  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
			ADD_SLOT( double  , kn  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"} );
			ADD_SLOT( double  , kr  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
			ADD_SLOT( double  , mu  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
			ADD_SLOT( double  , damprate  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator computes forces for interactions beween particles and a cylinder.
        )EOF";
			}

			inline void execute () override final
			{
				CylinderWallFunctor func { *center , *axis , *cylinder_angular_velocity , *cylinder_velocity, *radius, *dt, *kt, *kn, *kr, *mu, *damprate};
				compute_cell_particles( *grid , false , func , compute_field_set , gpu_execution_context() , gpu_time_account_func() );
			}
		};

	template<class GridT> using CylinderWallTmpl = CylinderWall<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "cylinder_wall", make_grid_variant_operator< CylinderWallTmpl > );
	}

}

