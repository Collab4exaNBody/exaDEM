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
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/field_value.hpp>
#include <exaDEM/interface/interface.hpp>
#include <exaDEM/shapes.hpp>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

struct BrokenInterfaceDetails {
  exanb::Vec3d pos;        /// center of the owner particle
  exanb::Quaternion quat;  /// orientation of the owner particle
  double homothety;        /// homothety of the owner particle
  long iteration;          /// Iteraction where the interface is broken
  uint64_t id_a;           /// id of particle A (owner)
  uint64_t id_b;           /// id of particle B (partner)
  int sid;                 /// shape ID
  int fid;                 /// face ID, the number/list of vertices is derived from the shape's face definition
};

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT>>
class StoreBrokenInterfaceOp : public OperatorNode {
  // attributes processed during computation
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Time iteration number"});
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(InterfaceManager, im, INPUT_OUTPUT, DocString{""});
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
  ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Main output directory."});
  ADD_SLOT(std::string, BrokenInterfacesFile, INPUT, "BrokenInterfaces.txt",
           DocString{"Output file collecting broken interface records, appended at every call."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator stores

        YAML example [no option]:

          - 
      )EOF";
  }

  inline void execute() final {
    auto& interfaces = *im;
    long step = *timestep;
    InteractionWrapper<InteractionType::InnerBond> data_wrapper = ic->get_sticked_interaction_wrapper();
    auto cells = grid->cells();
    auto& shps = *shapes_collection;

    std::vector<BrokenInterfaceDetails> data;

    // No copy from GPU if the data has not been touuch by the GPU
#pragma omp parallel
    {
      std::vector<int> vertices;
#pragma omp for
      for (size_t i = 0; i < interfaces.size(); i++) {
        // If the interface is broken, we need to update the friction of the interactions and break them
        if (interfaces.break_interface[i] == true) {
          auto [offset, size] = interfaces.data[i];
          auto I = data_wrapper(offset);
          assert(I.pair.type == InteractionTypeId::InnerBond);
          auto owner_particle = I.pair.owner();
          auto type_a = exadem_field_value(cells, owner_particle, field::type);

          const auto& shp = shps[type_a];
          BrokenInterfaceDetails tmp;
          vertices.resize(size);

          for (size_t idx = 0; idx < size; idx++) {
            vertices[idx] = data_wrapper.sub_i[offset + idx];
          }

          tmp.pos = exadem_field_center(cells, owner_particle);
          tmp.quat = exadem_field_value(cells, owner_particle, field::orient);
          tmp.homothety = exadem_field_value(cells, owner_particle, field::homothety);
          tmp.fid = shp->identify_face(vertices);
          tmp.sid = type_a;
          tmp.iteration = step;
          tmp.id_a = owner_particle.id;
          tmp.id_b = I.pair.partner().id;
#pragma omp critical
          {
            data.push_back(std::move(tmp));
          }
        }
      }
    }

    // === Gather every rank's broken interface records onto rank 0 ===
    const int mpi_root = 0;
    int rank = 0, mpi_size = 1;
    MPI_Comm_rank(*mpi, &rank);
    MPI_Comm_size(*mpi, &mpi_size);

    std::vector<int> mpi_n_data(mpi_size, 0);
    std::vector<int> mpi_displ(mpi_size, 0);

    int number_of_data = data.size();
    int all_data = 0;
    MPI_Gather(&number_of_data, 1, MPI_INT, mpi_n_data.data(), 1, MPI_INT, mpi_root, *mpi);

    if (rank == mpi_root) {
      all_data = std::accumulate(mpi_n_data.begin(), mpi_n_data.end(), 0);
      int tot = 0;
      for (int proc = 0; proc < mpi_size; proc++) {
        mpi_n_data[proc] *= sizeof(BrokenInterfaceDetails);
        mpi_displ[proc] = tot;
        tot += mpi_n_data[proc];
      }
    }

    std::vector<BrokenInterfaceDetails> all_broken(all_data);
    MPI_Gatherv(data.data(), data.size() * sizeof(BrokenInterfaceDetails), MPI_CHAR, all_broken.data(),
                mpi_n_data.data(), mpi_displ.data(), MPI_CHAR, mpi_root, *mpi);

    // === Write the gathered records to the output file (rank 0 only) ===
    std::string full_path = (*dir_name) + "/" + (*BrokenInterfacesFile);
    lout << "Write " << all_data << " interface in " << full_path << std::endl;
    if (rank == mpi_root && all_data > 0) {
      std::filesystem::create_directory(*dir_name);
      const bool write_header = !std::filesystem::exists(full_path);
      std::ofstream file(full_path, std::ios::app);
      if (write_header) {
        file << "iteration id_a id_b nb_vertex [vertex_1 vertex_2 vertex_3]..." << std::endl;
      }
      for (auto& d : all_broken) {
        const shape* shp = shps[d.sid];
        auto [vidx, nb_vertices] = shp->get_face(d.fid);

        file << d.iteration << " " << d.id_a << " " << d.id_b << " " << nb_vertices;
        for (int v = 0; v < nb_vertices; v++) {
          exanb::Vec3d vertex = shp->get_vertex(vidx[v], d.pos, d.homothety, d.quat);
          file << " " << vertex.x << " " << vertex.y << " " << vertex.z;
        }
        file << std::endl;
      }
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(store_broken_interface) {
  OperatorNodeFactory::instance()->register_factory("store_broken_interface",
                                                    make_grid_variant_operator<StoreBrokenInterfaceOp>);
}
}  // namespace exaDEM
