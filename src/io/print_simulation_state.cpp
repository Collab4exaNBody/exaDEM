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
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/log.h>
#include <onika/string_utils.h>

#include <exaDEM/dem_simulation_state.h>
#include <mpi.h>
#include <filesystem> // C++17

namespace exaDEM
{
  using namespace exanb;

  class PrintSimulationStateNode : public OperatorNode
  {
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);

    // physics data
    ADD_SLOT(long, timestep, INPUT, REQUIRED);
    ADD_SLOT(double, physical_time, INPUT, REQUIRED);
    ADD_SLOT(SimulationState, simulation_state, INPUT, REQUIRED);

    // printing options
    ADD_SLOT(bool, print_header, INPUT, false);
    ADD_SLOT(bool, internal_units, INPUT, false);

    // save file
    ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory name."});
    ADD_SLOT(std::string, log_name, INPUT, REQUIRED, DocString{"Write an Output file containing log lines."});
    ADD_SLOT(bool, save_file, INPUT, true, DocString{"Save line logs into a file, default behavior is true."});

    // LB and particle movement statistics
    ADD_SLOT(long, lb_counter, INPUT_OUTPUT);
    ADD_SLOT(long, move_counter, INPUT_OUTPUT);
    ADD_SLOT(long, domain_ext_counter, INPUT_OUTPUT);
    ADD_SLOT(double, lb_inbalance_max, INPUT_OUTPUT);

    // optional physics quantities
    ADD_SLOT(double, electronic_energy, INPUT, OPTIONAL);

  public:
    inline bool is_sink() const override final { return true; }

    inline void execute() override final
    {
      static const std::string header = "     Step     Time          Particles  Mv/Ext/Imb.       |dn|  avg. act. I     avg. I   avg. Ec   avg. Erot     Volume       Mass  Throughput";

      bool lb_flag = (*lb_counter) > 0;
      long move_count = *move_counter;
      long domext_count = *domain_ext_counter;
      double lb_inbalance = *lb_inbalance_max;

      // std::cout << "lb_counter = "<< *lb_counter << std::endl;

      *lb_counter = 0;
      *move_counter = 0;
      *domain_ext_counter = 0;
      *lb_inbalance_max = 0.0;

      const SimulationState &sim_info = *(this->simulation_state);

      char lb_move_char = ' ';
      if (move_count >= 1)
      {
        if (move_count == 1)
        {
          lb_move_char = 'm';
        }
        else if (move_count < 10)
        {
          lb_move_char = '0' + move_count;
        }
        else
        {
          lb_move_char = 'M';
        }
      }

      char domext_char = ' ';
      if (domext_count >= 1)
      {
        if (domext_count == 1)
        {
          domext_char = 'd';
        }
        else if (domext_count < 10)
        {
          domext_char = '0' + domext_count;
        }
        else
        {
          domext_char = 'D';
        }
      }

      std::string lb_value;
      if (lb_flag)
      {
        if (lb_inbalance == 0.0)
        {
          lb_value = "  N/A  ";
        }
        else
        {
          lb_value = onika::format_string("%.1e", lb_inbalance);
        }
      }

      if (*print_header)
      {
        lout << header;
        if (electronic_energy.has_value())
        {
          lout << "  Elect. Energy";
        }
        lout << std::endl;
      }

      int new_timestep = *timestep;
      auto new_timepoint = std::chrono::steady_clock::now();
      double throughput = sim_info.compute_particles_throughput(new_timepoint, new_timestep);
      simulation_state->update_timestep_timepoint(new_timepoint, new_timestep);

      auto active_interactions = sim_info.active_interaction_count();
      auto total_interactions = sim_info.interaction_count();

      double avg_act_I = double(active_interactions) / sim_info.particle_count();
      double avg_I = double(total_interactions) / sim_info.particle_count();

      std::string line = onika::format_string("%9ld % .6e %13ld  %c %c %8s %.3e    %9.3f  %9.3f % .3e % .3e % .3e % .3e % .4e",
                                       *timestep,                           // %9ld
                                       *physical_time,                      // %.6e
                                       sim_info.particle_count(),           // %13ld
                                       lb_move_char, domext_char, lb_value, // %c %c %.8s
                                       std::abs(sim_info.dn()), avg_act_I, avg_I, sim_info.kinetic_energy_scal() / sim_info.particle_count(), sim_info.rotation_energy_scal() / sim_info.particle_count(), sim_info.volume(), sim_info.mass(), throughput);

      lout << line;
      lout << std::endl;

      if (*save_file)
      {
        namespace fs = std::filesystem;
        std::string full_path = (*dir_name) + "/" + (*log_name);
        fs::path path(full_path);
        int rank;
        MPI_Comm_rank(*mpi, &rank);
        fs::create_directory(*dir_name);

        if (rank == 0)
        {
          fs::create_directory(*dir_name);
          std::fstream file(full_path, std::ios::out | std::ios::in | std::ios::app);
          if (*print_header)
            file << header << std::endl;
          file << line << std::endl;
        }
      }
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(print_simulation_state) { OperatorNodeFactory::instance()->register_factory("print_simulation_state", make_simple_operator<PrintSimulationStateNode>); }

} // namespace exaDEM
