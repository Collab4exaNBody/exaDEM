#include <mpi.h>
#include <cassert>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

#include <memory>
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/network.hpp>


namespace exaDEM
{
	using namespace exanb;

	template<typename GridT
		, class = AssertGridHasFields< GridT >
		>
		class ContactNetwork : public OperatorNode
		{
			ADD_SLOT( MPI_Comm , mpi        , INPUT , MPI_COMM_WORLD);
			ADD_SLOT( GridT    , grid       , INPUT_OUTPUT , REQUIRED );
			ADD_SLOT( GridCellParticleInteraction , grid_interaction  , INPUT , DocString{"Interaction list"} );
			ADD_SLOT( shapes                , shapes_collection       , INPUT , DocString{"Collection of shapes"});
			ADD_SLOT( HookeParams , config  , INPUT );
			ADD_SLOT( double      , dt      , INPUT );
			ADD_SLOT( std::string , basename, INPUT , REQUIRED  , DocString{"Output filename"});
			ADD_SLOT( std::string , basedir , INPUT , "network" , DocString{"Output directory, default is network"});
			ADD_SLOT( long        , timestep, INPUT , DocString{"Iteration number"} );

			public:

			inline std::string documentation() const override final
			{
				return R"EOF(
				        )EOF";
			}


			inline void execute () override final
			{
				auto & interactions = grid_interaction->m_data;

				// Fill network and manage paraview output
				NetworkFunctor<GridT> manager( *grid, *shapes_collection, *config, *dt);

				// mpi stuff
				int rank, size;
				MPI_Comm_rank(*mpi, &rank);
				MPI_Comm_size(*mpi, &size);
				std::string directory = (*basedir) + "/" + (*basename) + "_" + std::to_string(*timestep);
				std::string filename = directory + "/" + (*basename) + "_" + std::to_string(*timestep) + "_" + std::to_string(rank) ;
				// prepro
				if(rank == 0)
				{
					namespace fs = std::filesystem;
					fs::create_directory(*basedir);
					fs::create_directory(directory);
					std::string dir = *basedir;
					std::string name = *basename + "_" + std::to_string(*timestep);
					manager.write_pvtp (dir, name ,size);
				}

				MPI_Barrier(*mpi);

				// fill network
				for(size_t c = 0 ; c < interactions.size() ; c++)
				{
					CellExtraDynamicDataStorageT<Interaction>& storage = interactions[c];
					auto& info = storage.m_info;
					auto* data_ptr = storage.m_data.data();
					for (auto& it : info)
					{
						manager ( data_ptr, std::get<0>(it), std::get<1>(it)); // ptr, offset, size
					}
				}

				std::stringstream position, connect, value;
				auto ids = manager.create_indirection_array();
				manager.fill_position(position, ids);
				manager.fill_connect_and_value(connect, value, ids);
				manager.write_vtp (filename, ids.size(), position, connect, value);
			}
		};

	template<class GridT> using ContactNetworkTmpl = ContactNetwork<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "dump_contact_network", make_grid_variant_operator< ContactNetwork > );
	}
}

