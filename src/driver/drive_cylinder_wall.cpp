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
//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include "exanb/core/operator.h"
#include "exanb/core/operator_slot.h"
#include "exanb/core/operator_factory.h"
#include "exanb/core/make_grid_variant_operator.h"
#include "exanb/core/parallel_grid_algorithm.h"
#include "exanb/core/grid.h"
#include "exanb/core/domain.h"
#include "exanb/compute/compute_cell_particles.h"
#include <mpi.h>
#include <memory>
#include <exaDEM/cylinder_wall.h>

namespace exanb
{

	template<typename GridT
		, class = AssertGridHasFields< GridT, field::_fx,field::_fy,field::_fz >
		>
		class DriveCylinderWall : public OperatorNode
		{
			static constexpr Vec3d default_axis = { 1.0, 0.0, 1.0 };
			static constexpr Vec3d null= { 0.0, 0.0, 0.0 };
			// attributes processed during computation
			using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom, field::_friction >;
			static constexpr ComputeFields compute_field_set {};

			ADD_SLOT( GridT   , grid  		, INPUT_OUTPUT );
			ADD_SLOT( MPI_Comm , mpi     	, INPUT , MPI_COMM_WORLD);
			ADD_SLOT( Domain  , domain    , INPUT , REQUIRED );
			ADD_SLOT( Vec3d   , center 		, INPUT 	, REQUIRED 	, DocString{"Center of the cylinder"});
			ADD_SLOT( Vec3d   , axis   		, INPUT 	, default_axis 	, DocString{"Define the plan of the cylinder"});
			ADD_SLOT( Vec3d   , cylinder_angular_velocity 	, INPUT 	, null		, DocString{"Angular velocity of the cylinder, default is 0 m.s-"});
			ADD_SLOT( Vec3d   , cylinder_velocity 		, INPUT 	, null		, DocString{"Cylinder velocity, could be used in 'expert mode'"});
			ADD_SLOT( double  , radius 			, INPUT_OUTPUT 	, REQUIRED 	, DocString{"Radius of the cylinder, positive and should be superior to the biggest sphere radius in the cylinder"});
			ADD_SLOT( double  , radius_max  , INPUT , REQUIRED 	, DocString{"Maximum radius desired of the cylinder, positive and should be superior to the biggest sphere radius in the cylinder"});
			ADD_SLOT( double  , dr 				, INPUT 	, REQUIRED 	, DocString{"radius increment"});
			ADD_SLOT( double  , compacity , INPUT 	, 0, DocString{"Compacity desired"});
			ADD_SLOT( double  , dt       	, INPUT 	, REQUIRED 	, DocString{"Time increment of the simulation"});
			ADD_SLOT( double  , kt  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
			ADD_SLOT( double  , kn  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
			ADD_SLOT( double  , kr  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
			ADD_SLOT( double  , mu  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});
			ADD_SLOT( double  , damprate  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact cyclinder/sphere"});

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
        This operator drives the cylinder radius according the compacity desired or a maximum radius.
        )EOF";
			}

			inline void execute () override final
			{
				// note : Particle radius size can change.
				// Compute total volume size
				MPI_Comm comm = *mpi;
				auto cells = grid->cells();
				IJK dims = grid->dimension();
				size_t ghost_layers = grid->ghost_layers();
				IJK dims_no_ghost = dims - (2*ghost_layers);
				double vp = 0.0; // vp particles volume
				const double pi	= 4*std::atan(1);
				const double coeff = (4.0/3.0) * pi;
#       pragma omp parallel
				{
					GRID_OMP_FOR_BEGIN(dims_no_ghost,_,loc_no_ghosts, reduction(+:vp) )
					{
						IJK loc = loc_no_ghosts + ghost_layers;
						size_t cell_i = grid_ijk_to_index(dims,loc);
						auto n = cells[cell_i].size();
						auto* __restrict__ r = cells[cell_i][field::radius];
						for(unsigned int i = 0 ; i < n ; i++)
						{
							vp += coeff * r[i] * r[i] * r[i];
						}
					}
					GRID_OMP_FOR_END;
				}

				MPI_Allreduce(MPI_IN_PLACE, &vp, 1, MPI_DOUBLE, MPI_SUM,comm);

				// compute cylinder volume
				const double new_radius = (*radius) + (*dr);
				Vec3d dom_size = (*domain).bounds_size();
				const double h = dot(dom_size, (Vec3d{1.0,1.0,1.0}-(*axis))); // height
				const double vc = (new_radius)*(new_radius)*(new_radius) * pi * h;// cylinder volume
				const double c = vp / vc;
				if(c>(*compacity) && (new_radius <= (*radius_max)))
				{
					*radius = new_radius;
				}

				CylinderWallFunctor func { *center , *axis , *cylinder_angular_velocity , *cylinder_velocity, *radius, *dt, *kt, *kn, *kr, *mu, *damprate};
				compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
			}
		};

	template<class GridT> using DriveCylinderWallTmpl = DriveCylinderWall<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "drive_cylinder_wall", make_grid_variant_operator< DriveCylinderWallTmpl > );
	}
}

