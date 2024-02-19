#include <vector>
#include <iomanip>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
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

#include <exanb/compute/compute_cell_particles.h>

#include <mpi.h>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_printer.hpp>

namespace exaDEM
{
	using namespace exanb;
	template<	class GridT, class = AssertGridHasFields< GridT >> class WriteParaviewPolyhedraOperator : public OperatorNode
	{
		using ComputeFields = FieldSet< field::_rx ,field::_ry ,field::_rz, field::_type, field::_orient>;
		static constexpr ComputeFields compute_field_set {};
		ADD_SLOT( MPI_Comm , mpi      , INPUT , MPI_COMM_WORLD);
		ADD_SLOT( GridT    , grid     , INPUT_OUTPUT );
		ADD_SLOT( Domain   , domain   , INPUT , REQUIRED );
		ADD_SLOT( std::string , basename, INPUT , REQUIRED , DocString{"Output filename"});
		ADD_SLOT( std::string , basedir, INPUT , "polyhedra_paraview" , DocString{"Output directory, default is polyhedra_paraview"});
		ADD_SLOT( long        , timestep      , INPUT , DocString{"Iteration number"} );
		ADD_SLOT( shapes , shapes_collection, INPUT_OUTPUT , DocString{"Collection of shapes"});

		public:
		inline std::string documentation() const override final
		{
			return R"EOF( This operator initialize shapes data structure from a shape input file.
    	    			)EOF";
		}

		inline void execute () override final
		{
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
			}

			MPI_Barrier(*mpi);

			auto& shps = *shapes_collection;
			const auto cells = grid->cells();
			const size_t n_cells = grid->number_of_cells();


			size_t count_vertex(0), count_face(0), polygon_offset_in_stream(0);
			std::stringstream buff_vertices; // store vertices
			std::stringstream buff_faces; // store faces
			std::stringstream buff_offsets; // store face offsets


#define __PARAMS__ count_vertex, count_face, buff_vertices, buff_faces, buff_offsets
			// fill string buffers
			for(size_t cell_a = 0 ; cell_a < n_cells ; cell_a++)
			{
				if(grid->is_ghost_cell(cell_a)) continue;
				const int n_particles = cells[cell_a].size();
				auto* __restrict__ rx = cells[cell_a][field::rx];
				auto* __restrict__ ry = cells[cell_a][field::ry];
				auto* __restrict__ rz = cells[cell_a][field::rz];
				auto* __restrict__ type = cells[cell_a][field::type];
				auto* __restrict__ orient = cells[cell_a][field::orient];
				for(int j = 0 ; j < n_particles ; j++)
				{
					exanb::Vec3d pos  {rx[j], ry[j], rz[j]};
					const shape* shp = shps[type[j]];
					build_buffer(pos, shp, orient[j], polygon_offset_in_stream, __PARAMS__); 
				}
			};
			
			if(rank == 0) 
			{
				std::string dir = *basedir;
				std::string name = *basename + "_" + std::to_string(*timestep);
				exaDEM::write_pvtp (dir, name ,size);
			}
			exaDEM::write_vtp (filename, __PARAMS__);
#undef __PARAMS__
		}
	};

	// this helps older versions of gcc handle the unnamed default second template parameter
	template <class GridT> using WriteParaviewPolyhedraOperatorTemplate = WriteParaviewPolyhedraOperator<GridT>;

	// === register factories ===  
	CONSTRUCTOR_FUNCTION
	{
		OperatorNodeFactory::instance()->register_factory( "write_paraview_polyhedra", make_grid_variant_operator< WriteParaviewPolyhedraOperatorTemplate > );
	}
}
