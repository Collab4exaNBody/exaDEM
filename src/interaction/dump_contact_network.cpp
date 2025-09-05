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
#include <mpi.h>
#include <cassert>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <memory>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/classifier/classifier_for_all.hpp>
#include <exaDEM/shapes.hpp>
#include <onika/string_utils.h>
#include <exaDEM/network.hpp>

namespace exaDEM
{
  using namespace exanb;

	template <typename GridT, class = AssertGridHasFields<GridT>> class ContactNetwork : public OperatorNode
	{
		ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
		ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
		ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
		ADD_SLOT( std::string , filename , INPUT , "output");
		ADD_SLOT(long, timestep, INPUT, DocString{"Iteration number"});
		
		ADD_SLOT(Classifier2, ic2, INPUT_OUTPUT);

		public:
		inline std::string documentation() const override final
		{
			return R"EOF(
                  This operator creates paraview files containing the contact network.
				        )EOF";
		}

		inline void execute() override final
		{
		        //printf("DUMP\n");
			// mpi stuff
			int rank, size;
			MPI_Comm_rank(*mpi, &rank);
			MPI_Comm_size(*mpi, &size);

			Classifier<InteractionSOA>& classifier = (*ic);
			//auto& classifier2 = *ic2;
			
			NetworkFunctor<GridT> manager(*grid);

			if (rank == 0)
			{
				std::filesystem::create_directories( *filename );
			}

			MPI_Barrier(*mpi);

			// iterate over interaction types
			for (size_t type = 0; type < classifier.number_of_waves(); type++)
			//for (size_t type = 0; type < 4; type++) // skip drivers
			{
				auto& interactions = classifier.waves[type];
				auto& forces = classifier.buffers[type];
        const size_t n = interactions.size();
				manager(n, interactions, forces); 
			}

			if (rank == 0)
			{
				manager.write_pvtp(*filename, size);
			}

			std::string file = *filename + "/%06d.vtp";
			file = onika::format_string(file,  rank);
			auto ids = manager.create_indirection_array();
			manager.fill_position(ids);
			manager.fill_connect_and_value(ids);
			manager.write_vtp(file, ids.size());
			//printf("DUMP END\n");
		}
	};

	template <class GridT> using ContactNetworkTmpl = ContactNetwork<GridT>;

	// === register factories ===
	ONIKA_AUTORUN_INIT(dump_contact_network) { OperatorNodeFactory::instance()->register_factory("dump_contact_network", make_grid_variant_operator<ContactNetwork>); }
} // namespace exaDEM
