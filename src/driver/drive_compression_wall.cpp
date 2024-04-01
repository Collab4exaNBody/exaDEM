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

	template<	class GridT, class = AssertGridHasFields< GridT, field::_fx, field::_fy, field::_fz, field::_friction >>
		class DriveCompressionWallOperator : public OperatorNode
	{

		using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_vx ,field::_vy ,field::_vz, field::_vrot, field::_radius , field::_fx ,field::_fy ,field::_fz, field::_mass, field::_mom, field::_friction >;
		static constexpr ComputeFields compute_field_set {};
		ADD_SLOT( MPI_Comm           , mpi                 , INPUT , MPI_COMM_WORLD);
		ADD_SLOT( GridT  , grid            , INPUT_OUTPUT );
		ADD_SLOT( Domain , domain          , INPUT , REQUIRED );
		ADD_SLOT( double , dt              , INPUT , REQUIRED , DocString{"Timestep of the simulation"});
		ADD_SLOT( double , sigma   , INPUT , double(0), DocString{"Sigma is the ..."});
		ADD_SLOT( Vec3d  , normal  , INPUT , Vec3d{1.0,0.0,0.0} , DocString{"Normal vector of the rigid surface"});
		ADD_SLOT( double , offset  , INPUT_OUTPUT , 0.0, DocString{"Offset from the origin (0,0,0) of the rigid surface"} );
		ADD_SLOT( double , kt , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact wall/sphere"});
		ADD_SLOT( double , kn , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact wall/sphere"} );
		ADD_SLOT( double , kr , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact wall/sphere"});
		ADD_SLOT( double , mu , INPUT , REQUIRED , DocString{"Parameter of the force law used to model contact wall/sphere"});
		ADD_SLOT( double , damprate  			, INPUT 	, REQUIRED 	, DocString{"Parameter of the force law used to model contact wall/sphere"});
		ADD_SLOT( double  , w_vel  			, INPUT_OUTPUT 	, double(0.0) 	, DocString{"Velocities of the wall, this value should be equal to 0 if it's the first use"});
		ADD_SLOT( double  , w_acc  			, INPUT_OUTPUT 	, double(0.0) 	, DocString{"Acceleration of the wall, this value should be equal to 0 if it's the first use"});

		public:
		inline std::string documentation() const override final
		{
			return R"EOF(
        This operator drives the compression wall offset. The forcefield used to model the interaction between the wall and particles is Hooke.
        )EOF";
		}

		inline void execute () override final
		{
			// mpi stuff
			MPI_Comm comm = *mpi;
			// first step (time scheme)
			const double sig = *sigma;
			*offset += (*dt) * (*w_vel) + 0.5 * (*dt) * (*dt) * (*w_acc);
			if(sig != 0.0) (*w_vel) += 0.5 * (*dt) * (*w_acc);

			RigidSurfaceFunctor func {*normal, *offset, *w_vel, *dt, *kt, *kn, *kr, *mu, *damprate};
			double sum_f = 0.0; // forces applied on the ball
			double sum_m = 0.0; // summ of masses of particles inside the ball 
													// TODO WARNING : it works for "basic" walls x y or z
			auto bounds = bounds_size(domain->bounds())	;
			Vec3d plan = Vec3d{ 1 - std::abs((*normal).x) , 1 - std::abs((*normal).y), 1 - std::abs((*normal).z) };

			double surface = 1.0;
			if(plan.x == 1) surface *= bounds.x;
			if(plan.y == 1) surface *= bounds.y;
			if(plan.z == 1) surface *= bounds.z;
			auto cells = grid->cells();
			IJK dims = grid->dimension();
			size_t ghost_layers = grid->ghost_layers();
			IJK dims_no_ghost = dims - (2*ghost_layers);

#     pragma omp parallel
			{
				GRID_OMP_FOR_BEGIN(dims_no_ghost,_,loc_no_ghosts, reduction(+: sum_m, sum_f))
				{
					IJK loc = loc_no_ghosts + ghost_layers;
					size_t cell_i = grid_ijk_to_index(dims,loc);
					auto& cell_ptr = cells[cell_i];

					// define fields
					auto* __restrict__ _rx = cell_ptr[field::rx];
					auto* __restrict__ _ry = cell_ptr[field::ry];
					auto* __restrict__ _rz = cell_ptr[field::rz];
					auto* __restrict__ _vx = cell_ptr[field::vx];
					auto* __restrict__ _vy = cell_ptr[field::vy];
					auto* __restrict__ _vz = cell_ptr[field::vz];
					auto* __restrict__ _vrot = cell_ptr[field::vrot];
					auto* __restrict__ _r = cell_ptr[field::radius];
					auto* __restrict__ _fx = cell_ptr[field::fx];
					auto* __restrict__ _fy = cell_ptr[field::fy];
					auto* __restrict__ _fz = cell_ptr[field::fz];
					auto* __restrict__ _m = cell_ptr[field::mass];
					auto* __restrict__ _mom = cell_ptr[field::mom];
					auto* __restrict__ _fric = cell_ptr[field::friction];
					const size_t n = cells[cell_i].size();

					// call BallWallFunctor for each particle
#         pragma omp simd //reduction(+:sum_f, sum_m)
					for(size_t j=0;j<n;j++)
					{
						sum_f += func.run(
								_rx[j], _ry[j], _rz[j],
								_vx[j], _vy[j], _vz[j],
								_vrot[j], _r[j],
								_fx[j], _fy[j], _fz[j],
								_m[j], _mom[j], _fric[j]
								);
						sum_m += _m[j];
					}
				}
				GRID_OMP_FOR_END
			}
			// reduce ball_f
			{
				double tmp[2] = {sum_f, sum_m};
				MPI_Allreduce(MPI_IN_PLACE, tmp, 2, MPI_DOUBLE, MPI_SUM, comm);
				sum_f = tmp[0];
				sum_m = tmp[1];
			}
			// second step (time scheme)
			const double C = 0.5;
			if(sum_m != 0.0)
			{
				*w_acc = ( sum_f - (sig * surface) - ( (*damprate) * (*w_vel)))/ (sum_m * C);
			}
			*w_vel += 0.5 * (*dt) * (*w_acc); 
			//std::cout << *w_acc << std::endl;
		}
	};


	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using DriveCompressionWallOperatorTemplate = DriveCompressionWallOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "drive_compression_wall", make_grid_variant_operator< DriveCompressionWallOperatorTemplate > );
	}
}

