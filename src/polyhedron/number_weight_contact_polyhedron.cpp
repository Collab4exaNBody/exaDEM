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
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shapes.hpp>
#include <map>
#include <utility>
#include <vector>

namespace exaDEM {

/**
 * @brief "NumberWeight" strategy: computes, for every active (overlapping)
 * particle-particle polyhedron contact interaction, a weight 1/N_ij used to
 * scale kn/kt down, where N_ij is the number of simultaneously active contact
 * interactions currently detected between the same pair of particles (i, j).
 * (Other weighting strategies, e.g. based on contact area or energy instead of
 * a plain count, could be added later as separate, similarly named operators.)
 *
 * This is only needed for fragmentation-type simulations: two fragments that
 * were bonded by an InnerBond can remain in contact at several points at once
 * (sharing a face: VertexVertex/VertexEdge/VertexFace/EdgeEdge) right after the
 * bond breaks, which would otherwise make the resulting contact stiffer than a
 * single kn/kt contact (each sub-contact applying the full user-defined
 * stiffness). Most simulations never run into this situation (at most one
 * active contact per pair), so this operator is kept separate from
 * contact_polyhedron and should only be added to the pipeline when needed
 * (between classify_interactions and contact_polyhedron), rather than paying
 * its cost (a second, CPU-only pass of geometric contact detection) every
 * step of every simulation.
 */
template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>>
class ComputeContactNumberWeightPolyhedron : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});

  // particle-particle polyhedron contact type ids (VertexVertex, VertexEdge, VertexFace, EdgeEdge)
  static constexpr int poly_type_start = 0;
  static constexpr int poly_type_end = 3;

 public:
  inline std::string documentation() const final {
    return R"EOF(
        "NumberWeight" strategy: this operator computes, for every active
        particle-particle polyhedron contact interaction (VertexVertex,
        VertexEdge, VertexFace, EdgeEdge), a weight equal to 1 / N_ij, where
        N_ij is the number of simultaneously active contact interactions
        currently detected between the same pair of particles (i, j). The
        weight is stored in item[attr::weight] (1.0 by default) and used by
        contact_polyhedron (polyhedron::ContactLawFunc) to scale kn/kt down
        accordingly, so that a pair with several simultaneous sub-contacts
        does not end up with an effective stiffness that is a multiple of
        kn_user/kt_user. Pairs with a single active contact get weight 1.0,
        i.e. unchanged behaviour.

        This is only needed for fragmentation simulations, where two fragments
        that were bonded by an InnerBond can remain in contact at several
        points at once right after the bond breaks. Add this operator to the
        pipeline between classify_interactions and contact_polyhedron only for
        such simulations: it duplicates the geometric contact detection that
        contact_polyhedron performs, so it should not be added where it is not
        needed.

        YAML example [no option]:

          - number_weight_contact_polyhedron
      )EOF";
  }

  /** Returns the particle-pair key (ordered, order-independent) used to group
   *  simultaneous contact interactions between the same two particles. */
  static inline std::pair<uint64_t, uint64_t> pair_key(uint64_t id_a, uint64_t id_b) {
    return (id_a < id_b) ? std::pair<uint64_t, uint64_t>{id_a, id_b} : std::pair<uint64_t, uint64_t>{id_b, id_a};
  }

  /** Detects active (dn<0) contact interactions for one particle-particle contact
   *  type (VertexVertex/VertexEdge/VertexFace/EdgeEdge) and records them, grouped
   *  by particle pair, into pair_count/active/active_keys. Recurses at compile
   *  time over [poly_type_start, poly_type_end] since the geometric detection
   *  functor (detect<type>) is specialized per concrete type.
   *  Host-only: pair counting needs a shared per-pair accumulator, so it cannot
   *  run as an embarrassingly-parallel GPU kernel. */
  template <int type, typename TMPLC, typename TMPLV>
  void scan_pair_weights(Classifier& classifier, TMPLC* const __restrict__ cells, TMPLV* const __restrict__ gv,
                         const shape* const shps, std::map<std::pair<uint64_t, uint64_t>, uint32_t>& pair_count,
                         std::vector<std::pair<int, uint64_t>>& active,
                         std::vector<std::pair<uint64_t, uint64_t>>& active_keys) {
    detect<type> detection;
    auto [data, size] = classifier.get_info<InteractionType::ParticleParticle>(type);
    auto view = data.view();
    for (uint64_t idx = 0; idx < size; idx++) {
      auto item = view(idx);
      auto& pi = item.i();
      auto& pj = item.j();
      auto& cell_i = cells[pi.cell_];
      auto& cell_j = cells[pj.cell_];
      const auto& type_i = cell_i[field::type][pi.p_];
      const auto& type_j = cell_j[field::type][pj.p_];
      const auto& h_i = cell_i[field::homothety][pi.p_];
      const auto& h_j = cell_j[field::homothety][pj.p_];
      const ParticleVertexView vertices_i = {pi.p_, gv[pi.cell_]};
      const ParticleVertexView vertices_j = {pj.p_, gv[pj.cell_]};
      const shape& shp_i = shps[type_i];
      const shape& shp_j = shps[type_j];
      auto [contact, dn, n, contact_position] =
          detection(vertices_i, h_i, pi.sub_, &shp_i, vertices_j, h_j, pj.sub_, &shp_j);
      if (dn < 0) {
        const auto key = pair_key(pi.id_, pj.id_);
        pair_count[key]++;
        active.push_back({type, idx});
        active_keys.push_back(key);
      }
    }
    if constexpr (type + 1 <= poly_type_end) {
      scan_pair_weights<type + 1>(classifier, cells, gv, shps, pair_count, active, active_keys);
    }
  }

  inline void execute() final {
    if (grid->number_of_cells() == 0) {
      return;
    }

    Classifier& classifier = *ic;
    const auto cells = grid->cells();
    auto* vertex_fields = cvf->data();
    const shape* const shps = shapes_collection->data();

    std::map<std::pair<uint64_t, uint64_t>, uint32_t> pair_count;
    std::vector<std::pair<int, uint64_t>> active;            // {type, index}
    std::vector<std::pair<uint64_t, uint64_t>> active_keys;  // same order as active
    scan_pair_weights<poly_type_start>(classifier, cells, vertex_fields, shps, pair_count, active, active_keys);

    for (size_t k = 0; k < active.size(); k++) {
      const auto [type, idx] = active[k];
      double weight = 1.0;
      if (pair_count[active_keys[k]] > 0) {
        weight = 1.0 / double(pair_count[active_keys[k]]);
      }
      auto [data, size] = classifier.get_info<InteractionType::ParticleParticle>(type);
      auto view = data.view();
      auto item = view(idx);
      item[attr::weight] = weight;
      view.update(idx, item);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(number_weight_contact_polyhedron) {
  OperatorNodeFactory::instance()->register_factory("number_weight_contact_polyhedron",
                                                    make_grid_variant_operator<ComputeContactNumberWeightPolyhedron>);
}
}  // namespace exaDEM
