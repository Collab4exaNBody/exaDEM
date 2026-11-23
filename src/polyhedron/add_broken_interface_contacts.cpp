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
#include <onika/cuda/cuda_context.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <array>
#include <cassert>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interaction/inner_bond_interaction.hpp>
#include <exaDEM/interface/interface.hpp>
#include <exaDEM/polyhedron/broken_interface_contacts.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/shapes.hpp>
#include <vector>

namespace exaDEM {
using namespace exanb;

template <typename GridT, class = AssertGridHasFields<GridT>>
class AddBrokenInterfaceContacts : public OperatorNode {
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
  ADD_SLOT(bool, broken_interface, INPUT_OUTPUT, REQUIRED,
           DocString{"In: an interface is broken. Out: a neighbor-list rebuild is required."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator creates the particle-particle contacts (VV, VE, VF, EE in both directions) between the two
        particles of every interface broken at this time step, without rebuilding the neighbor lists. The
        contacts are appended at the end of the ParticleParticle containers of the classifier, so they are
        processed by the regular contact force operators. The friction of a broken inner bond is transferred
        to the matching vertex-vertex contact. Handled interfaces are removed from the interface manager.

        The PartnerGhost copies of the broken interfaces (MPI or periodic boundaries) are handled by
        add_broken_interface_ghost_contacts, which must run before this operator and after
        broken_interface_friction_updater. broken_interface is set to true (rebuild) only if
        enable_broken_interface_contacts is false.

        YAML example:

          - add_broken_interface_contacts
      )EOF";
  }

  inline void execute() final {
    if (!(*broken_interface)) {
      return;  // globally consistent (reduced by apply_interface_fracture_criterion)
    }
    // broken_interface is now "a rebuild is required"
    *broken_interface = !(*enable_broken_interface_contacts);
    if (*broken_interface) {
      return;
    }

    auto& interfaces = *im;
    const size_t n_interfaces = interfaces.size();

    ONIKA_CU_DEVICE_SYNCHRONIZE();

    auto data_view = ic->get_sticked_interaction_wrapper();
    const auto cells = grid->cells();
    auto& vertex_fields = *cvf;
    const shapes& shps = *shapes_collection;
    const Mat3d xform = domain->xform();
    const bool is_xform = !domain->xform_is_identity();

    BrokenInterfaceContacts new_contacts;
    size_t n_broken = 0;
    size_t n_history = 0;

    for (size_t i = 0; i < n_interfaces; i++) {
      if (!interfaces.break_interface_[i]) {
        continue;
      }
      n_broken++;
      auto [offset, size] = interfaces.data_[i];
      const InnerBondInteraction bond = data_view(offset);
      PolyhedronPairSide side_a =
          make_polyhedron_pair_side(cells, vertex_fields, shps, xform, is_xform, bond.pair_.owner());
      PolyhedronPairSide side_b =
          make_polyhedron_pair_side(cells, vertex_fields, shps, xform, is_xform, bond.pair_.partner());

      // Friction history from the broken inner bonds (VV).
      auto history = [&, offset = offset, size = size](PlaceholderInteraction& item) {
        for (size_t j = offset; j < offset + size; j++) {
          const Interaction hist = broke_interaction(data_view(j));
          if (hist.pair_ == item.pair_) {
            item.as<Interaction>()[attr::friction] = hist[attr::friction];
            n_history++;
            return;
          }
        }
      };
      // NotGhost or OwnerGhost, the PartnerGhost copies are handled by add_broken_interface_ghost_contacts.
      add_broken_pair_contacts(side_a, side_b, bond.pair_.ghost_, *rcut_inc, history, new_contacts);
    }

    if (n_broken == 0) {
      return;
    }

    size_t n_contacts = 0;
    for (int typeID = InteractionTypeId::FirstIdParticle; typeID <= InteractionTypeId::LastIdParticle; typeID++) {
      n_contacts += new_contacts[typeID].size();
      ic->append_pp_tail(typeID, new_contacts[typeID]);
    }

    // The friction now lives in the tail: avoid a stale history when the bonds are unclassified.
    Vec3d* __restrict__ bond_friction = data_view[attr::friction].data();
    size_t n_kept = 0;
    for (size_t i = 0; i < n_interfaces; i++) {
      if (!interfaces.break_interface_[i]) {
        interfaces.data_[n_kept] = interfaces.data_[i];
        interfaces.break_interface_[n_kept] = false;
        n_kept++;
      } else {
        auto [offset, size] = interfaces.data_[i];
        for (size_t j = offset; j < offset + size; j++) {
          bond_friction[j] = Vec3d{0, 0, 0};
        }
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
