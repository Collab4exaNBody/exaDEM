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
#include <exanb/core/log.h>
#include <exanb/core/string_utils.h>
#include <exanb/core/physics_constants.h>

#include <exaDEM/dem_simulation_state.h>

namespace exaDEM
{
	using namespace exanb;

	class PrintSimulationStateNode : public OperatorNode
	{  
		// thermodynamic state & physics data
		ADD_SLOT( long               , timestep            , INPUT , REQUIRED );
		ADD_SLOT( double             , physical_time       , INPUT , REQUIRED );
		ADD_SLOT( SimulationState    , simulation_state    , INPUT , REQUIRED );

		// printing options
		ADD_SLOT( bool               , print_header        , INPUT, false );
		ADD_SLOT( bool               , internal_units      , INPUT, false );

		// LB and particle movement statistics
		ADD_SLOT( long               , lb_counter          , INPUT_OUTPUT );
		ADD_SLOT( long               , move_counter        , INPUT_OUTPUT );
		ADD_SLOT( long               , domain_ext_counter  , INPUT_OUTPUT );
		ADD_SLOT( double             , lb_inbalance_max    , INPUT_OUTPUT );

		// optional physics quantities
		ADD_SLOT( double             , electronic_energy   , INPUT, OPTIONAL );

		public:
		inline bool is_sink() const override final { return true; }

		inline void execute () override final
		{
			double conv_temperature = 1.e4 * legacy_constant::atomicMass / legacy_constant::boltzmann ;
			//double conv_energy = 1.e4 * legacy_constant::atomicMass / legacy_constant::elementaryCharge;
			//static const std::string header = "     Step     Time (ps)     Particles   Mv/Ext/Imb.  Tot. E. (eV/part)  Kin. E. (eV/part)  Pot. E. (eV/part)  Temperature   Pressure     sMises     Volume       Mass";
			static const std::string header = "     Step     Time (ps)     Particles   Mv/Ext/Imb. Temperature     Volume       Mass  Part/timestep/s";

			if( *internal_units )
			{
				conv_temperature = 1.0;
			}

			bool lb_flag = (*lb_counter) > 0 ;
			long move_count = *move_counter ;
			long domext_count = *domain_ext_counter;
			double lb_inbalance = *lb_inbalance_max;

			//std::cout << "lb_counter = "<< *lb_counter << std::endl;

			*lb_counter = 0;
			*move_counter = 0;
			*domain_ext_counter = 0;
			*lb_inbalance_max = 0.0;

			const SimulationState& sim_info = *(this->simulation_state);

			char lb_move_char = ' ';
			if( move_count >= 1 )
			{
				if( move_count == 1 ) { lb_move_char = 'm'; }
				else if( move_count < 10 )  { lb_move_char = '0'+move_count; }
				else { lb_move_char = 'M'; }
			}

			char domext_char = ' ';
			if( domext_count >= 1 )
			{
				if( domext_count == 1 ) { domext_char = 'd'; }
				else if( domext_count < 10 )  { domext_char = '0'+domext_count; }
				else { domext_char = 'D'; }
			}

			std::string lb_value;
			if( lb_flag )
			{
				if( lb_inbalance == 0.0 )
				{
					lb_value = "  N/A  ";
				}
				else
				{
					lb_value = format_string("%.1e", lb_inbalance);
				}
			}


			if( *print_header )
			{
				lout << header;
				if( electronic_energy.has_value() ) { lout << "  Elect. Energy"; }
				lout << std::endl;
			}


			int new_timestep = *timestep;
			auto new_timepoint = std::chrono::steady_clock::now();
			double throughput = sim_info.compute_particles_throughput (new_timepoint, new_timestep);
			simulation_state->update_timestep_timepoint (new_timepoint, new_timestep);


			lout<<format_string("%9ld % .6e %13ld  %c %c %8s % 11.3f % .3e % .3e      % .4e",
					*timestep, // %9ld
					*physical_time, // %.6e
					sim_info.particle_count(), // %13ld 
					lb_move_char,domext_char,lb_value, // %c %c %.8s
					sim_info.temperature_scal() / sim_info.particle_count() * conv_temperature, //%11.3f
					sim_info.volume(),
					sim_info.mass(),
					throughput) ;

			lout << std::endl;
		}

	};

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "print_simulation_state", make_simple_operator<PrintSimulationStateNode> );
	}

}

