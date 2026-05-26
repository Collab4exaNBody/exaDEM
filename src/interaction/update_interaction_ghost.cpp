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
#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// exaNBody
#include <exanb/mpi/ghosts_comm_scheme.h>

// exaDEM
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction_ghost_manager.hpp>
#include <exaDEM/traversal.hpp>

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT>>
class UpdateInteractionGhost : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GhostCommunicationScheme, ghost_comm_scheme, INPUT, REQUIRED);
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
  ADD_SLOT(InteractionGhostManager, interaction_ghost_manager, INPUT_OUTPUT, DocString{""});

 public:
  inline std::string documentation() const final {
    return R"EOF(
       This operator updates the interaction list in the ghost layer of the grid.
       The operator assumes that the ghost layer of the grid is already filled with the particles
       and that the interaction list is already computed for the owner particles.

       YAML example:

        - update_interaction_ghost
       )EOF";
  }

  inline void execute() final {
    // Get slots
    auto& interaction_cells = ges->m_data;
    InteractionGhostManager& manager = *interaction_ghost_manager;
    exanb::GhostCommunicationScheme& ghost_scheme = *ghost_comm_scheme;
    const MPI_Comm& comm = *mpi;
    auto& gridT = *grid;
    const auto cells = gridT.cells();
    const IJK dims = gridT.dimension();
    // Reset Interaction within the grid ghost layer
#pragma omp parallel for
    for (size_t i = 0; i < gridT.number_of_cells(); i++) {
      if (!gridT.is_ghost_cell(i)) {
        continue;
      }
      const unsigned int n_particles = cells[i].size();
      CellExtraDynamicDataStorageT<PlaceholderInteraction>& storage = interaction_cells[i];
      storage.initialize(n_particles);
    }

    // Setup MPI comms here
    // tags constants
    const int TAG_SIZE = 100;
    const int TAG_CONFIG = 101;
    const int TAG_PAYLOAD = 102;

    const auto& partners = ghost_scheme.m_partner;
    size_t n_procs = partners.size();
    int my_rank, mpi_size;
    MPI_Comm_rank(comm, &my_rank);
    MPI_Comm_size(comm, &mpi_size);

    // resize
    manager.rbuf.resize(n_procs);
    manager.sbuf.resize(n_procs);
    manager.send_cell_config.resize(n_procs);
    manager.recv_cell_config.resize(n_procs);

    std::vector<int> config_size;
    config_size.resize(n_procs);

    for (size_t proc = 0; proc < n_procs; proc++) {
      auto& send_config = manager.send_cell_config[proc];
      const auto& partner = partners[proc];
      manager.sbuf[proc].clear();
      send_config.clear();

      // build send buffers and send_config correctly
      uint32_t shift = 0;
      for (auto& it : partner.m_sends) {
        auto& interactions = interaction_cells[it.m_cell_i].m_data;
        if (interactions.empty()) {
          continue;
        }
        uint32_t n_sent = 0;

        for (size_t j = 0; j < interactions.size(); j++) {
          const exaDEM::PlaceholderInteraction& I = interactions[j];  // storage.get_particle_item(p, j);
          if (I.pair.ghost == InteractionPair::OwnerGhost) {
            manager.sbuf[proc].push_back(I);
            ++n_sent;
          }
        }

        // push the *number actually sent* and the current shift
        if (n_sent > 0) {
          send_config.push_back({static_cast<uint32_t>(it.m_partner_cell_i), n_sent, shift});
          shift += n_sent;
        }
      }

      // send cell configs
      if (partner.m_particles_to_send > 0) {
        MPI_Request req1 = MPI_REQUEST_NULL, req2 = MPI_REQUEST_NULL, req3 = MPI_REQUEST_NULL;
        config_size[proc] = static_cast<int>(send_config.size());
        MPI_Isend(&config_size[proc], 1, MPI_INT, proc, TAG_SIZE, comm, &req1);
        if (config_size[proc] > 0) {
          // send_config: send bytes (count = number_of_bytes)
          MPI_Isend(send_config.data(), static_cast<int>(send_config.size() * sizeof(CellGhostDetails)), MPI_BYTE, proc,
                    TAG_CONFIG, comm, &req2);

          // payload: sbuf in bytes
          long long count = static_cast<long long>(manager.sbuf[proc].size()) * sizeof(exaDEM::PlaceholderInteraction);
          MPI_Isend(manager.sbuf[proc].data(), count, MPI_BYTE, proc, TAG_PAYLOAD, comm, &req3);
        }
      }
    }

    for (size_t proc = 0; proc < n_procs; proc++) {
      auto& recv_config = manager.recv_cell_config[proc];
      const auto& partner = partners[proc];
      manager.rbuf[proc].clear();
      recv_config.clear();

      // recv cell configs
      if (partner.m_particles_to_receive > 0) {
        MPI_Request rreq1 = MPI_REQUEST_NULL, rreq2 = MPI_REQUEST_NULL, rreq3 = MPI_REQUEST_NULL;
        MPI_Status s1, s2, s3;
        int isize = 0;

        MPI_Irecv(&isize, 1, MPI_INT, proc, TAG_SIZE, comm, &rreq1);
        MPI_Wait(&rreq1, &s1);

        if (isize > 0) {
          recv_config.resize(isize);
          MPI_Irecv(recv_config.data(), static_cast<int>(recv_config.size() * sizeof(CellGhostDetails)), MPI_BYTE, proc,
                    TAG_CONFIG, comm, &rreq2);
          MPI_Wait(&rreq2, &s2);

          // determine overall buffer size from recv_config: use the last entry
          auto& last = recv_config.back();
          size_t needed = last.m_shift + last.m_size;
          manager.rbuf[proc].resize(needed);

          long long count = manager.rbuf[proc].size() * sizeof(exaDEM::PlaceholderInteraction);
          MPI_Irecv(manager.rbuf[proc].data(), count, MPI_BYTE, proc, TAG_PAYLOAD, comm, &rreq3);
          MPI_Wait(&rreq3, &s3);
        }
      }
    }

    // Fill ghost layers with
#pragma omp parallel for schedule(dynamic)
    for (size_t proc = 0; proc < n_procs; proc++) {
      auto& buffer = manager.rbuf[proc];
      for (auto& config : manager.recv_cell_config[proc]) {
        assert(config.m_size > 0);
        IJK loc = grid_index_to_ijk(dims, config.m_partner_cell_i);
        assert(config.m_partner_cell_i < interaction_cells.size());
        auto& storage = interaction_cells[config.m_partner_cell_i].m_data;
        if (!gridT.is_ghost_cell(config.m_partner_cell_i)) {
          color_log::mpi_error("update_interaction_ghost", "This cell is not a ghost");
        }

        uint64_t partner_id = -1;
        uint32_t partner_cell = -1;
        uint16_t partner_p = -1;

        for (size_t i = config.m_shift; i < config.m_shift + config.m_size; i++) {
          bool found = false;
          assert(i < buffer.size());
          exaDEM::PlaceholderInteraction& item = buffer[i];
          // update info
          auto& owner = item.pair.owner();
          // auto old = owner.cell;
          owner.cell = config.m_partner_cell_i;  /// defined by another mpi process (or maybe himself if there are
                                                 /// periodic boundary conditions)
          item.pair.ghost = InteractionPair::PartnerGhost;  // identify this kind of interaction

          // check that the particle is in this cell and update of the particle offset in the cell
          {
            bool is_idx_exist = false;
            assert(owner.p < cells[owner.cell].size());
            if (owner.p < cells[owner.cell].size()) {
              if (owner.id == cells[owner.cell][field::id][owner.p]) {
                is_idx_exist = true;
              }
            }

            if (!is_idx_exist) {
              const uint64_t* const __restrict__ ids = cells[owner.cell][field::id];
              for (size_t p = 0; p < cells[owner.cell].size(); p++) {
                if (owner.id == ids[p]) {
                  owner.p = p;
                  is_idx_exist = true;
                }
              }
            }

            if (!is_idx_exist) {
              continue;  // can append cause not all particles are copied in ghost cells
            }
          }

          auto& partner = item.pair.partner();

          if (partner.id == partner_id) {
            partner.cell = partner_cell;
            partner.p = partner_p;
            found = true;  // still true
          } else {
            found = false;
          }

          for (int z = loc.k - 1; (z <= (loc.k + 1)) && !found; z++) {
            for (int y = loc.j - 1; (y <= (loc.j + 1)) && !found; y++) {
              for (int x = loc.i - 1; (x <= (loc.i + 1)) && !found; x++) {
                IJK loc_neigh = IJK{x, y, z};
                if (loc_neigh == loc) continue;
                if (gridT.contains(loc_neigh)) {
                  uint32_t neigh = grid_ijk_to_index(dims, loc_neigh);
                  assert(neigh < gridT.number_of_cells());
                  if (gridT.is_ghost_cell(neigh)) continue;
                  const uint64_t* const __restrict__ ids = cells[neigh][field::id];
                  for (size_t p = 0; p < cells[neigh].size(); p++) {
                    if (partner.id == ids[p]) {
                      partner.cell = neigh;
                      partner.p = p;
                      // Cache values
                      partner_id = partner.id;
                      partner_cell = neigh;
                      partner_p = p;
                      found = true;
                    }
                  }
                }
              }
            }
          }

          if (found) {
            // check
            assert(partner.cell < gridT.number_of_cells());
            assert(partner.p < cells[partner.cell].size());
            if (partner.id != cells[partner.cell][field::id][partner.p]) {
              item.print();
              color_log::mpi_error("update_interaction_ghost",
                                   "partner.id: " + std::to_string(partner.id) +
                                       " is != of cells[partner.cell][field::id][partner.p]: " +
                                       std::to_string(cells[partner.cell][field::id][partner.p]));
            }
            storage.push_back(item);
          }
        }
      }
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(update_interaction_ghost) {
  OperatorNodeFactory::instance()->register_factory("update_interaction_ghost",
                                                    make_grid_variant_operator<UpdateInteractionGhost>);
}
}  // namespace exaDEM
