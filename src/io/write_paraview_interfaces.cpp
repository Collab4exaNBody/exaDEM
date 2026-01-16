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
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <onika/math/basic_types.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_stream.h>
#include <onika/string_utils.h>
#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>

#include <cstdlib>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interface/interface.hpp>
#include <exaDEM/interface/interface_printer.hpp>
#include <exaDEM/shapes.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <vector>

namespace exaDEM {
template <class GridT, class = AssertGridHasFields<GridT>>
class WriteParaviewInterfaceOperator : public OperatorNode {
  using ComputeFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_type, field::_orient>;
  static constexpr ComputeFields compute_field_set{};
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(std::string, filename, INPUT, "ParaviewOutputFiles/interface_%010d");
  ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Iteration number"});
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
  ADD_SLOT(InterfaceManager, im, INPUT, DocString{""});
  ADD_SLOT(Classifier, ic, INPUT, DocString{"Interaction lists classified according to their types"});

  // optionnal
  ADD_SLOT(bool, mpi_rank, INPUT, false, DocString{"Add a field containing the mpi rank."});

 public:
  inline std::string documentation() const final {
    return R"EOF( 
      This operator initialize shapes data structure from a shape input file.

      YAML example:

        - write_paraview_interfaces:
           filename: "OptionalFilename_%10d"
           mpi_rank: true
                )EOF";
  }

  inline void execute() final {
    // mpi stuff
    int rank, size;
    MPI_Comm_rank(*mpi, &rank);
    MPI_Comm_size(*mpi, &size);

    auto cells = grid->cells();
    InterfaceManager& interfaces = *im;
    Classifier& classifier = *ic;
    auto& interactions = classifier.get_data<InteractionType::InnerBond>(InteractionTypeId::InnerBond);
    auto& shps = *shapes_collection;
    paraview_interface_helper buffers = {*mpi_rank};  // it conatins streams

    if (rank == 0) {
      std::filesystem::create_directories(*filename);
    }

    MPI_Barrier(*mpi);

    size_t number_of_interfaces = interfaces.data.size();
    buffers.n_vertices = 0;
    buffers.n_polygons = number_of_interfaces;

    // fill string buffers
    for (size_t i = 0; i < number_of_interfaces; i++) {
      auto& interface = interfaces.data[i];

      double E = 0;
      for (size_t j = interface.loc; j < interface.loc + interface.size; j++) {
        exaDEM::InnerBondInteraction interaction = interactions[j];
        auto& loc = interaction.i();
        auto& cell = cells[loc.cell];
        uint16_t type = cell[field::type][loc.p];
        Quaternion quat = cell[field::orient][loc.p];
        auto* shp = shps[type];
        Vec3d r = {cell[field::rx][loc.p], cell[field::ry][loc.p], cell[field::rz][loc.p]};
        double h = cell[field::homothety][loc.p];
        Vec3d v = shp->get_vertex(loc.sub, r, h, quat);
        buffers.vertices << v.x << " " << v.y << " " << v.z << " ";
        buffers.ids << i << " ";
        buffers.connectivities << buffers.n_vertices++ << " ";
        buffers.et << interaction.et << " ";
        buffers.en << interaction.en << " ";
        E += (interaction.en + interaction.et) / interaction.criterion;
      }
      buffers.offsets << buffers.n_vertices << " ";
      for (size_t j = 0; j < interface.size; j++) {
        buffers.fracturation << E << " ";
      }
    };

    if (rank == 0) {
      exaDEM::write_pvtp_interface(*filename, size, buffers);
    }

    if (buffers.mpi_rank) {  // add ranks
      for (int i = 0; i < buffers.n_vertices; i++) {
        buffers.ranks << rank << " ";
      }
    }

    std::string file = *filename + "/%06d.vtp";
    file = onika::format_string(file, rank);
    exaDEM::write_vtp_interface(file, buffers);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(write_paraview_interfaces) {
  OperatorNodeFactory::instance()->register_factory("write_paraview_interfaces",
                                                    make_grid_variant_operator<WriteParaviewInterfaceOperator>);
}
}  // namespace exaDEM
