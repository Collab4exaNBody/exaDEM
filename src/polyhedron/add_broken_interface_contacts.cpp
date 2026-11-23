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

#include <array>
#include <cassert>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interaction/inner_bond_interaction.hpp>
#include <exaDEM/interface/interface.hpp>
#include <exaDEM/polyhedron/nbh_polyhedron_pair.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/shapes.hpp>
#include <vector>

namespace exaDEM {
using namespace exanb;

template <typename GridT, class = AssertGridHasFields<GridT>>
class AddBrokenInterfaceContacts : public OperatorNode {
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
  ADD_SLOT(double, rcut_inc, INPUT, REQUIRED,
           DocString{"value added to the search distance to update neighbor list less frequently. in physical space"});
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(InterfaceManager, im, INPUT_OUTPUT, DocString{"Interfaces built from the InnerBond interactions"});
  ADD_SLOT(bool, enable, INPUT, true,
           DocString{"If false, every broken interface requests a neighbor-list rebuild (previous behavior)."});
  ADD_SLOT(bool, result, OUTPUT,
           DocString{"True if a neighbor-list rebuild is required to handle the broken interfaces."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator creates the particle-particle contacts (VV, VE, VF, EE in both directions) between the two
        particles of every interface broken at this time step, without rebuilding the neighbor lists. The
        contacts are appended at the end of the ParticleParticle containers of the classifier, so they are
        processed by the regular contact force operators. The friction of a broken inner bond is transferred
        to the matching vertex-vertex contact. Handled interfaces are removed from the interface manager.

        An interface involving a ghost particle (MPI or periodic boundary) cannot be handled locally: in
        that case, the output 'result' is true on every MPI process and the neighbor lists must be rebuilt
        (previous behavior). This operator must run after store_broken_interface and
        broken_interface_friction_updater.

        YAML example:

          - add_broken_interface_contacts
      )EOF";
  }

  inline void execute() final {
    auto& interfaces = *im;
    const size_t n_interfaces = interfaces.size();

    ONIKA_CU_DEVICE_SYNCHRONIZE();

    auto data_view = ic->get_sticked_interaction_wrapper();

    // Interfaces with a ghost particle require a rebuild.
    int n_broken = 0;
    int require_rebuild = (*enable) ? 0 : 1;
    for (size_t i = 0; i < n_interfaces; i++) {
      if (!interfaces.break_interface_[i]) {
        continue;
      }
      n_broken++;
      auto [offset, size] = interfaces.data_[i];
      for (size_t j = offset; j < offset + size; j++) {
        if (data_view[attr::ghost][j] != InteractionPair::NotGhost) {
          require_rebuild = 1;
        }
      }
    }

    MPI_Allreduce(MPI_IN_PLACE, &require_rebuild, 1, MPI_INT, MPI_MAX, *mpi);
    *result = (require_rebuild != 0);
    if (*result) {
      return;
    }

    if (n_broken == 0) {
      return;  // nothing to do on this process
    }

    const auto cells = grid->cells();
    auto& vertex_fields = *cvf;
    const shapes& shps = *shapes_collection;
    const double rVerlet = *rcut_inc;
    const Mat3d xform = domain->xform();
    const bool is_xform = !domain->xform_is_identity();

    auto make_side = [&](const ParticleSubLocation& loc) -> PolyhedronPairSide {
      const auto& cell = cells[loc.cell_];
      Vec3d r = {cell[field::rx][loc.p_], cell[field::ry][loc.p_], cell[field::rz][loc.p_]};
      if (is_xform) {
        r = xform * r;
      }
      return {loc.id_,
              loc.cell_,
              loc.p_,
              r,
              cell[field::orient][loc.p_],
              cell[field::homothety][loc.p_],
              shps[cell[field::type][loc.p_]],
              {loc.p_, vertex_fields[loc.cell_]}};
    };

    std::array<std::vector<PlaceholderInteraction>, InteractionTypeId::NTypesPP> new_contacts;
    size_t n_history = 0;

    for (size_t i = 0; i < n_interfaces; i++) {
      if (!interfaces.break_interface_[i]) {
        continue;
      }
      auto [offset, size] = interfaces.data_[i];
      const InnerBondInteraction bond = data_view(offset);
      const ParticleSubLocation& owner = bond.pair_.owner();
      const ParticleSubLocation& partner = bond.pair_.partner();
      assert(owner.id_ < partner.id_);

      PolyhedronPairSide side_a = make_side(owner);
      PolyhedronPairSide side_b = make_side(partner);

      auto add_contact = [&](PlaceholderInteraction& item, int sub_i, int sub_j) -> void {
        item.pair_.pi_.sub_ = sub_i;
        item.pair_.pj_.sub_ = sub_j;
        item.reset();
        // Friction history from the broken inner bond (VV).
        if (item.type() == InteractionTypeId::VertexVertex) {
          for (size_t j = offset; j < offset + size; j++) {
            const InnerBondInteraction broken_bond = data_view(j);
            const Interaction history = broke_interaction(broken_bond);
            if (history.pair_ == item.pair_) {
              item.as<Interaction>()[attr::friction] = history[attr::friction];
              n_history++;
              break;
            }
          }
        }
        assert(item.type() < InteractionTypeId::NTypesPP);
        new_contacts[item.type()].push_back(item);
      };

      PlaceholderInteraction item;
      item.clear_placeholder();
      item.pair_.ghost_ = InteractionPair::NotGhost;
      item.pair_.swap_ = false;
      add_polyhedron_pair_interactions(side_a, side_b, rVerlet, item, add_contact);
    }

    size_t n_contacts = 0;
    for (int typeID = InteractionTypeId::FirstIdParticle; typeID <= InteractionTypeId::LastIdParticle; typeID++) {
      n_contacts += new_contacts[typeID].size();
      ic->append_pp_tail(typeID, new_contacts[typeID]);
    }

    size_t n_kept = 0;
    for (size_t i = 0; i < n_interfaces; i++) {
      if (!interfaces.break_interface_[i]) {
        interfaces.data_[n_kept] = interfaces.data_[i];
        interfaces.break_interface_[n_kept] = false;
        n_kept++;
      }
    }
    interfaces.data_.resize(n_kept);
    interfaces.break_interface_.resize(n_kept);

    ldbg << "add_broken_interface_contacts: " << n_broken << " broken interfaces, " << n_contacts
         << " contacts created (" << n_history << " with friction history)" << std::endl;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(add_broken_interface_contacts) {
  OperatorNodeFactory::instance()->register_factory("add_broken_interface_contacts",
                                                    make_grid_variant_operator<AddBrokenInterfaceContacts>);
}
}  // namespace exaDEM
