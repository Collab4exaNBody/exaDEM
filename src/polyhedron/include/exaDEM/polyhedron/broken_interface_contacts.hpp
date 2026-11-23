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
#pragma once

#include <array>
#include <cassert>
#include <exaDEM/interaction/placeholder_interaction.hpp>
#include <exaDEM/polyhedron/nbh_polyhedron_pair.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/shapes.hpp>
#include <vector>

namespace exaDEM {

using BrokenInterfaceContacts = std::array<std::vector<PlaceholderInteraction>, InteractionTypeId::NTypesPP>;

/** @brief Builds the side of a particle pair from its location in the grid. */
template <typename CellsT>
inline PolyhedronPairSide make_polyhedron_pair_side(CellsT cells, CellVertexField& vertex_fields, const shapes& shps,
                                                    const Mat3d& xform, bool is_xform, const ParticleSubLocation& loc) {
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
}

/** @brief Creates the VV, VE, VF, EE contacts (both directions) between a (owner) and b (partner).
 * history(item) is called for every VV contact and may set its friction.
 */
template <typename HistoryFunc>
inline void add_broken_pair_contacts(PolyhedronPairSide& a, PolyhedronPairSide& b, uint8_t ghost, double rVerlet,
                                     HistoryFunc& history, BrokenInterfaceContacts& contacts) {
  assert(a.id_ < b.id_);
  auto add_contact = [&](PlaceholderInteraction& item, int sub_i, int sub_j) -> void {
    item.pair_.pi_.sub_ = sub_i;
    item.pair_.pj_.sub_ = sub_j;
    item.reset();
    if (item.type() == InteractionTypeId::VertexVertex) {
      history(item);
    }
    assert(item.type() < InteractionTypeId::NTypesPP);
    contacts[item.type()].push_back(item);
  };

  PlaceholderInteraction item;
  item.clear_placeholder();
  item.pair_.ghost_ = ghost;
  item.pair_.swap_ = false;
  add_polyhedron_pair_interactions(a, b, rVerlet, item, add_contact);
}
}  // namespace exaDEM
