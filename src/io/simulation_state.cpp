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


#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/core/log.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/fields.h>
#include <exanb/core/basic_types.h>
#include <exanb/compute/reduce_cell_particles.h>

#include <mpi.h>
#include <cstring>

#include <exaDEM/simulation_state.h>
#include <exaDEM/dem_simulation_state.h>
#include <exaDEM/cell_list_wrapper.hpp>
#include <exaDEM/interaction/classifier.hpp>

// ================== Thermodynamic state compute operator ======================
namespace exaDEM
{
	using namespace exanb;

	template<class GridT ,	class = AssertGridHasFields< GridT, field::_vx, field::_vy, field::_vz, field::_mass >> struct SimulationStateNode : public OperatorNode
	{
		// compile time constant indicating if grid has type field

		ADD_SLOT( MPI_Comm           , mpi                 , INPUT , MPI_COMM_WORLD);
		ADD_SLOT( GridT              , grid                , INPUT , REQUIRED);
		ADD_SLOT( Domain             , domain              , INPUT , REQUIRED);
    ADD_SLOT( CellListWrapper    , cell_list , INPUT   , DocString{"list of non empty cells within the current grid"});
		ADD_SLOT( SimulationState    , simulation_state    , OUTPUT );

    // DEM data
    ADD_SLOT( Classifier         , ic                  , INPUT , DocString{"Interaction lists classified according to their types"} );
    ADD_SLOT( bool               , symetric            , INPUT , REQUIRED , DocString{"Use of symetric feature (contact law)"});

		static constexpr FieldSet<field::_vx ,field::_vy ,field::_vz, field::_mass> reduce_field_set {};
		static constexpr FieldSet<field::_vx> reduce_vx_field_set {};
		static constexpr FieldSet<field::_vy> reduce_vy_field_set {};
		static constexpr FieldSet<field::_vz> reduce_vz_field_set {};
		static constexpr FieldSet<field::_mass> reduce_mass_field_set {};
		inline void execute () override final
		{
			MPI_Comm comm = *mpi;
			SimulationState& sim_info = *simulation_state;

			Vec3d momentum;  // constructs itself with 0s
			Vec3d kinetic_energy;  // constructs itself with 0s
			double mass = 0.;
			uint64_t total_particles = 0;

      auto [cell_ptr, cell_size] = cell_list->info();
			exaDEM::simulation_state_variables sim {}; //kinetic_energy, momentum, mass, potential_energy, total_particles};
			ReduceSimulationStateFunctor func = {};
			reduce_cell_particles( *grid , false , func , sim, reduce_field_set , parallel_execution_context() , {} , cell_ptr, cell_size );

      // get interaction informations
      Classifier& classifier = *ic;
     	uint64_t active_interactions = classifier.get_active_interactions(*symetric);
     	uint64_t total_interactions  = classifier.number_of_interactions(*symetric);
      double dn                    = classifier.get_min_dn();

			// reduce partial sums and share the result
			{
				double tmpDouble[8] = {
					sim.momentum.x, sim.momentum.y, sim.momentum.z,
					sim.kinetic_energy.x, sim.kinetic_energy.y, sim.kinetic_energy.z,
					sim.mass , dn };
        uint64_t tmpUInt64T[3] = {sim.n_particles, active_interactions , total_interactions};
				MPI_Allreduce(MPI_IN_PLACE, tmpDouble, 8, MPI_DOUBLE, MPI_SUM, comm);
				MPI_Allreduce(MPI_IN_PLACE, tmpUInt64T, 3, MPI_UINT64_T, MPI_SUM, comm);

				momentum.x       = tmpDouble[0];
				momentum.y       = tmpDouble[1];
				momentum.z       = tmpDouble[2];
				kinetic_energy.x = tmpDouble[3];
				kinetic_energy.y = tmpDouble[4];
				kinetic_energy.z = tmpDouble[5];
				mass             = tmpDouble[6];
				dn               = tmpDouble[7];

				total_particles     = tmpUInt64T[0];
				active_interactions = tmpUInt64T[1];
				total_interactions  = tmpUInt64T[2];
			}

			// temperature
			Vec3d temperature = 2. * ( kinetic_energy - 0.5 * momentum * momentum / mass );

			// Volume
			double volume = 1.0;
			if( ! domain->xform_is_identity() )
			{
				Mat3d mat = domain->xform();
				Vec3d a { mat.m11, mat.m21, mat.m31 };
				Vec3d b { mat.m12, mat.m22, mat.m32 };
				Vec3d c { mat.m13, mat.m23, mat.m33 };
				volume = dot( cross(a,b) , c );
			}
			volume *= bounds_volume( domain->bounds() );

			// write results to output
			sim_info.set_temperature( temperature );
			sim_info.set_mass( mass );
			sim_info.set_volume( volume );
			sim_info.set_particle_count( total_particles );
      sim_info.set_active_interaction_count ( active_interactions );
      sim_info.set_interaction_count ( total_interactions );
      sim_info.set_dn ( dn );
		}
	};

	template<class GridT> using SimulationStateNodeTmpl = SimulationStateNode<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "simulation_state", make_grid_variant_operator< SimulationStateNodeTmpl > );
	}

}

