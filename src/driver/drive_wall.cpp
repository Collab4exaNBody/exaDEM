/*
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/basic_types.h>
#include <exanb/core/basic_types_operators.h>
#include <exanb/core/basic_types_stream.h>
#include <onika/memory/allocator.h> // for ONIKA_ASSUME_ALIGNED macro
#include <exanb/compute/compute_pair_optional_args.h>
#include <vector>
#include <iomanip>

#include <exanb/compute/compute_cell_particles.h>
#include <exaDEM/compute_wall.h>

#include <mpi.h>

namespace exaDEM
{
	using namespace exanb;

	template<
		class GridT,
					class = AssertGridHasFields< GridT, field::_fx, field::_fy, field::_fz, field::_friction >
						>
						class DriveWallOperator : public OperatorNode
						{

							using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom, field::_friction >;
							static constexpr ComputeFields compute_field_set {};
							ADD_SLOT( GridT  , grid            , INPUT_OUTPUT );
							ADD_SLOT( Domain , domain          , INPUT , REQUIRED );
							ADD_SLOT( double , dt              , INPUT , REQUIRED , DocString{"Timestep of the simulation"});
							ADD_SLOT( long   , timestep        , INPUT , REQUIRED );
							ADD_SLOT( long   , start_movement  , INPUT , long(0));
							ADD_SLOT( double , wall_velocity   , INPUT , double(0));
							ADD_SLOT( Vec3d  , normal  , INPUT , Vec3d{1.0,0.0,0.0} , DocString{"Normal vector of the rigid surface"});
							ADD_SLOT( double , offset  , INPUT , 0.0, DocString{"Offset from the origin (0,0,0) of the rigid surface"} );
							ADD_SLOT( double , kt , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double , kn , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"} );
							ADD_SLOT( double , kr , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double , mu , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
							ADD_SLOT( double , damprate  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});

							public:
							inline std::string documentation() const override final
							{
								return R"EOF(
        					This operator drives the wall offset. The forcefield used to model interation between the wall and particles is Hooke.
        				)EOF";
							}

							inline void execute () override final
							{
								// compute new offset
								double new_offset = 0.0;
								if( (*start_movement) < (*timestep))
								{
									new_offset = (*offset) + ((*timestep) - (*start_movement)) * (*dt) * (*wall_velocity);
								}
								else
								{
									new_offset = *offset;
								}
								RigidSurfaceFunctor func {*normal, new_offset, *wall_velocity, *dt, *kt, *kn, *kr, *mu, *damprate};
								compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
							}
						};


	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using DriveWallOperatorTemplate = DriveWallOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "drive_wall", make_grid_variant_operator< DriveWallOperatorTemplate > );
	}
}

