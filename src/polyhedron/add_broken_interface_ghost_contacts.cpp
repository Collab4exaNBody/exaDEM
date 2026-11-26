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
#include <mpi.h>
#include <onika/cuda/cuda_context.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interaction/inner_bond_interaction.hpp>
#include <exaDEM/interface/interface.hpp>
#include <exaDEM/polyhedron/broken_interface_contacts.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/shapes.hpp>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

namespace exaDEM {
using namespace exanb;

/** @brief A bond of a broken interface whose partner is a ghost (sent to the processes holding its copies). */
struct BrokenGhostBond {
  uint64_t id_i;
  uint64_t id_j;
  uint64_t sub_i;
  uint64_t sub_j;
  Vec3d friction;
};

template <typename GridT, class = AssertGridHasFields<GridT>>
class AddBrokenInterfaceGhostContacts : public OperatorNode {
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
  ADD_SLOT(double, rcut_inc, INPUT, REQUIRED,
           DocString{"value added to the search distance to update neighbor list less frequently. in physical space"});
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(InterfaceManager, im, INPUT_OUTPUT, DocString{"Interfaces built from the InnerBond interactions"});
  ADD_SLOT(bool, enable_broken_interface_contacts, INPUT, true,
           DocString{"If false, every broken interface requests a neighbor-list rebuild (previous behavior)."});
  ADD_SLOT(bool, broken_interface, INPUT, REQUIRED, DocString{"An interface is broken at this time step."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator updates the PartnerGhost copies of the interfaces broken at this time step (interfaces
        between a particle and a ghost, across MPI or periodic boundaries), without rebuilding the neighbor lists:
        the broken bonds are sent to every process, their PartnerGhost copies are marked as broken and the
        PartnerGhost contacts (VV, VE, VF, EE) are appended to the classifier, with the friction of the owner.
        It must run after broken_interface_friction_updater and before add_broken_interface_contacts.

        YAML example:

          - add_broken_interface_ghost_contacts
      )EOF";
  }

  inline void execute() final {
    if (!(*broken_interface) || !(*enable_broken_interface_contacts)) {
      return;  // globally consistent
    }

    ONIKA_CU_DEVICE_SYNCHRONIZE();

    auto& interfaces = *im;
    auto data_view = ic->get_sticked_interaction_wrapper();

    // Broken bonds of the local interfaces whose partner is a ghost.
    std::vector<BrokenGhostBond> local;
    size_t n_owner_ghost = 0;
    for (size_t i = 0; i < interfaces.size(); i++) {
      if (!interfaces.break_interface_[i]) {
        continue;
      }
      auto [offset, size] = interfaces.data_[i];
      if (data_view[attr::ghost][offset] != InteractionPair::OwnerGhost) {
        continue;
      }
      n_owner_ghost++;
      for (size_t j = offset; j < offset + size; j++) {
        const InnerBondInteraction bond = data_view(j);
        local.push_back(
            {bond.pair_.pi_.id_, bond.pair_.pj_.id_, bond.pair_.pi_.sub_, bond.pair_.pj_.sub_, bond[attr::friction]});
      }
    }

    // Share them with every process (only a few bonds, only when an interface breaks).
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(*mpi, &rank);
    MPI_Comm_size(*mpi, &nprocs);
    std::vector<int> counts(nprocs), displs(nprocs);
    int local_bytes = local.size() * sizeof(BrokenGhostBond);
    MPI_Allgather(&local_bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, *mpi);
    std::exclusive_scan(counts.begin(), counts.end(), displs.begin(), 0);
    const int total_bytes = displs.back() + counts.back();
    if (total_bytes == 0) {
      return;
    }
    std::vector<BrokenGhostBond> all(total_bytes / sizeof(BrokenGhostBond));
    MPI_Allgatherv(local.data(), local_bytes, MPI_BYTE, all.data(), counts.data(), displs.data(), MPI_BYTE, *mpi);

    std::map<std::pair<uint64_t, uint64_t>, std::vector<const BrokenGhostBond*>> broken;
    for (const auto& b : all) {
      broken[{b.id_i, b.id_j}].push_back(&b);
    }

    // Update the PartnerGhost copies held by this process.
    const auto cells = grid->cells();
    auto& vertex_fields = *cvf;
    const shapes& shps = *shapes_collection;
    const Mat3d xform = domain->xform();
    const bool is_xform = !domain->xform_is_identity();
    uint8_t* __restrict__ unbroken = data_view[attr::unbroken].data();
    Vec3d* __restrict__ bond_friction = data_view[attr::friction].data();

    BrokenInterfaceContacts new_contacts;
    size_t n_updated = 0;
    for (auto [offset, size] : interfaces.partner_ghost_data_) {
      const InnerBondInteraction bond = data_view(offset);
      auto it = broken.find({bond.pair_.pi_.id_, bond.pair_.pj_.id_});
      if (it == broken.end() || !unbroken[offset]) {
        continue;
      }
      n_updated++;
      for (size_t j = offset; j < offset + size; j++) {
        unbroken[j] = false;
        bond_friction[j] = Vec3d{0, 0, 0};
      }

      const auto& bonds = it->second;
      auto history = [&bonds](PlaceholderInteraction& item) {
        for (const BrokenGhostBond* b : bonds) {
          if (b->sub_i == item.pair_.pi_.sub_ && b->sub_j == item.pair_.pj_.sub_) {
            item.as<Interaction>()[attr::friction] = b->friction;
            return;
          }
        }
      };
      PolyhedronPairSide side_a =
          make_polyhedron_pair_side(cells, vertex_fields, shps, xform, is_xform, bond.pair_.owner());
      PolyhedronPairSide side_b =
          make_polyhedron_pair_side(cells, vertex_fields, shps, xform, is_xform, bond.pair_.partner());
      add_broken_pair_contacts(side_a, side_b, InteractionPair::PartnerGhost, *rcut_inc, history, new_contacts);
    }

    check_broken_ghost_balance(*mpi, n_owner_ghost, n_updated, "add_broken_interface_ghost_contacts");

    size_t n_contacts = 0;
    for (int typeID = InteractionTypeId::FirstIdParticle; typeID <= InteractionTypeId::LastIdParticle; typeID++) {
      n_contacts += new_contacts[typeID].size();
      if (!new_contacts[typeID].empty()) {
        ic->append_pp_tail(typeID, new_contacts[typeID]);
      }
    }

    ldbg << "add_broken_interface_ghost_contacts: " << all.size() << " broken ghost bonds received, " << n_updated
         << " PartnerGhost interfaces updated, " << n_contacts << " contacts created" << std::endl;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(add_broken_interface_ghost_contacts) {
  OperatorNodeFactory::instance()->register_factory("add_broken_interface_ghost_contacts",
                                                    make_grid_variant_operator<AddBrokenInterfaceGhostContacts>);
}
}  // namespace exaDEM
