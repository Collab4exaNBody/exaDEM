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
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/memory/allocator.h> // for ONIKA_ASSUME_ALIGNED macro
#include <exanb/compute/compute_pair_optional_args.h>

#include <exanb/compute/compute_cell_particles.h>

#include <mpi.h>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_printer.hpp>
#include <exanb/core/string_utils.h>

namespace exaDEM
{
  using namespace exanb;


  template <class GridT, class = AssertGridHasFields<GridT>> class WriteParaviewOBBParticlesOperator : public OperatorNode
  {
    using ComputeFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_type, field::_orient>;
    static constexpr ComputeFields compute_field_set{};
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(Domain, domain, INPUT, REQUIRED);
    ADD_SLOT(std::string, basename, INPUT, "obb", DocString{"Output filename"});
    ADD_SLOT(std::string, dir_name, INPUT_OUTPUT, REQUIRED, DocString{"Main output directory."});
    ADD_SLOT(long, timestep, INPUT, DocString{"Iteration number"});
    ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, DocString{"Collection of shapes"});

  public:
    inline std::string documentation() const final
    {
      return R"EOF( This operator dumps obb into a paraview output file.
    	    			)EOF";
    }

    inline void execute() final
    {
      // mpi stuff
      int rank, size;
      MPI_Comm_rank(*mpi, &rank);
      MPI_Comm_size(*mpi, &size);

      std::string ts = "%010d"; 
      std::string rk = "%06d"; 

      std::string directory = (*dir_name) + "/ParaviewOutputFiles/" + (*basename) + "_" + ts;
      directory = format_string(directory, *timestep);
      std::string filename = directory + "/" + rk + ".vtp";
      filename  = format_string(filename,  rank);

      // prepro
      if (rank == 0)
      {
        namespace fs = std::filesystem;
        fs::create_directory(*dir_name);
        fs::create_directory(directory);
      }

      MPI_Barrier(*mpi);

      auto &shps = *shapes_collection;
      const auto cells = grid->cells();
      const size_t n_cells = grid->number_of_cells();

      par_obb_helper buffers;

      // fill string buffers
      for (size_t cell_a = 0; cell_a < n_cells; cell_a++)
      {
        if (grid->is_ghost_cell(cell_a))
          continue;
        const int n_particles = cells[cell_a].size();
        auto *__restrict__ rx = cells[cell_a][field::rx];
        auto *__restrict__ ry = cells[cell_a][field::ry];
        auto *__restrict__ rz = cells[cell_a][field::rz];
        auto *__restrict__ id = cells[cell_a][field::id];
        auto *__restrict__ type = cells[cell_a][field::type];
        auto *__restrict__ orient = cells[cell_a][field::orient];
        for (int j = 0; j < n_particles; j++)
        {
          exanb::Vec3d pos{rx[j], ry[j], rz[j]};
          const shape *shp = shps[type[j]];
          build_buffer_obb(pos, id[j], type[j], shp, orient[j], buffers);
        }
      };

      if (rank == 0)
      {
        std::string dir = *dir_name + "/ParaviewOutputFiles/";
        std::string name = *basename + "_" + ts;
        name  = format_string(name,  *timestep); 
        exaDEM::write_pvtp_obb(dir, name, size);
      }
      exaDEM::write_vtp_obb(filename, buffers);
    }
  };

  // this helps older versions of gcc handle the unnamed default second template parameter
  template <class GridT> using WriteParaviewOBBParticlesOperatorTemplate = WriteParaviewOBBParticlesOperator<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("write_paraview_obb_particles", make_grid_variant_operator<WriteParaviewOBBParticlesOperatorTemplate>); }
} // namespace exaDEM
