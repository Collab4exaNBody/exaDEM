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
#include <onika/string_utils.h>

#include <mpi.h>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_printer.hpp>

namespace exaDEM
{
  using namespace exanb;
  template <class GridT, class = AssertGridHasFields<GridT>> class WriteParaviewPolyhedraOperator : public OperatorNode
  {
    using ComputeFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_type, field::_orient>;
    static constexpr ComputeFields compute_field_set{};
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(GridT, grid, INPUT, REQUIRED);
    ADD_SLOT(Domain, domain, INPUT, REQUIRED);
    ADD_SLOT(std::string , filename, INPUT , "output");
    ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Iteration number"});
    ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});

    // optionnal
    ADD_SLOT(bool, mpi_rank, INPUT, false, DocString{"Add a field containing the mpi rank."});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF( 
      This operator initialize shapes data structure from a shape input file.

      YAML example:

        - write_paraview_polyhedra:
           filename: "OptionalFilename_%10d"
           mpi_rank: true
    	    			)EOF";
    }

    inline void execute() override final
    {
      // mpi stuff
      int rank, size;
      MPI_Comm_rank(*mpi, &rank);
      MPI_Comm_size(*mpi, &size);

      if (rank == 0)
      {
        std::filesystem::create_directories( *filename );
      }

      MPI_Barrier(*mpi);

      auto &shps = *shapes_collection;
      const auto cells = grid->cells();
      const size_t n_cells = grid->number_of_cells();
      par_poly_helper buffers = {*mpi_rank}; // it conatins streams 

      // fill string buffers
      for (size_t cell_a = 0; cell_a < n_cells; cell_a++)
      {
        if (grid->is_ghost_cell(cell_a))
          continue;
        const int n_particles = cells[cell_a].size();
        auto *__restrict__ rx = cells[cell_a][field::rx];
        auto *__restrict__ ry = cells[cell_a][field::ry];
        auto *__restrict__ rz = cells[cell_a][field::rz];
        auto *__restrict__ vx = cells[cell_a][field::vx];
        auto *__restrict__ vy = cells[cell_a][field::vy];
        auto *__restrict__ vz = cells[cell_a][field::vz];
        auto *__restrict__ type = cells[cell_a][field::type];
        auto *__restrict__ id = cells[cell_a][field::id];
        auto *__restrict__ orient = cells[cell_a][field::orient];
        for (int j = 0; j < n_particles; j++)
        {
          exanb::Vec3d pos{rx[j], ry[j], rz[j]};
          const shape *shp = shps[type[j]];
          build_buffer_polyhedron(pos, shp, orient[j], id[j], type[j], vx[j], vy[j], vz[j], buffers);
        }
      };

      if (rank == 0)
      {
        exaDEM::write_pvtp_polyhedron(*filename, size, buffers);
      }

      if(buffers.mpi_rank) // add ranks 
      {
        for(int i = 0 ; i < buffers.n_vertices ; i++)
        {
          buffers.ranks << rank << " ";
        }
      }

      std::string file = *filename + "/%06d.vtp";
      file = onika::format_string(file,  rank);
      exaDEM::write_vtp_polyhedron(file, buffers);
    }
  };

  // this helps older versions of gcc handle the unnamed default second template parameter
  template <class GridT> using WriteParaviewPolyhedraOperatorTemplate = WriteParaviewPolyhedraOperator<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(write_paraview_polyhedra) { OperatorNodeFactory::instance()->register_factory("write_paraview_polyhedra", make_grid_variant_operator<WriteParaviewPolyhedraOperatorTemplate>); }
} // namespace exaDEM
