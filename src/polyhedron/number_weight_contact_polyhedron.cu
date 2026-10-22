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

#include <array>
#include <cstdint>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/classifier/classifier_for_all.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shapes.hpp>

namespace exaDEM {

/** @brief Persistent GPU buffers for the pair-counting pass (resized, not reallocated, across steps). */
struct NumberWeightScratch {
  onika::memory::CudaMMVector<double> dn;        ///< overlap, written by the GPU detection pass
  onika::memory::CudaMMVector<uint64_t> key_lo;  ///< min(id_i, id_j) per interaction
  onika::memory::CudaMMVector<uint64_t> key_hi;  ///< max(id_i, id_j) per interaction
  onika::memory::CudaMMVector<uint8_t> active;   ///< 1 if dn < 0 (contact currently active)
  onika::memory::CudaMMVector<uint32_t> count;   ///< N_ij, broadcast back to every active interaction of a pair
};

/** @brief Pass 1 (GPU): detects contact and writes dn/pair-key/active into the scratch buffers. */
template <int type, typename TMPLC, typename TMPLV>
struct DetectPairFunc {
  TMPLC* const __restrict__ cells;                 // pointer to the cells or particles container
  TMPLV* const __restrict__ gv;                    // pointer to the global vertex array (for polyhedron vertices)
  const shape* const shps;                         // pointer to the shape array (for polyhedron shapes)
  size_t offset;                                   // offset into the scratch buffers for this contact type
  mutable onika::cuda::span<double> dn_out;        // overlap, written by the GPU detection pass
  mutable onika::cuda::span<uint64_t> key_lo_out;  // min(id_i, id_j) per interaction
  mutable onika::cuda::span<uint64_t> key_hi_out;  // max(id_i, id_j) per interaction
  mutable onika::cuda::span<uint8_t> active_out;   // 1 if dn < 0 (contact currently active)
  detect<type> detection;                          // the contact detection functor for this contact type

  template <typename InteractionT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t i, InteractionT& item) const {
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
    const size_t g = offset + i;
    const uint64_t a = pi.id_;
    const uint64_t b = pj.id_;
    dn_out[g] = dn;
    key_lo_out[g] = (a < b) ? a : b;
    key_hi_out[g] = (a < b) ? b : a;
    active_out[g] = (dn < 0) ? uint8_t(1) : uint8_t(0);
  }
};

/** @brief Pass 2 (GPU): counts active interactions sharing the same particle pair (O(n) scan per interaction). */
struct CountPairsFunc {
  onika::cuda::span<const uint64_t> key_lo;   // min(id_i, id_j) per interaction
  onika::cuda::span<const uint64_t> key_hi;   // max(id_i, id_j) per interaction
  onika::cuda::span<const uint8_t> active;    // 1 if dn < 0 (contact currently active)
  mutable onika::cuda::span<uint32_t> count;  // see DetectPairFunc's comment on why this needs mutable

  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t g) const {
    if (!active[g]) {
      return;
    }
    uint32_t c = 0;
    const size_t n = active.size();
    for (size_t h = 0; h < n; h++) {
      if (active[h] && key_lo[h] == key_lo[g] && key_hi[h] == key_hi[g]) {
        c++;
      }
    }
    count[g] = c;
  }
};

/** @brief Pass 3 (GPU): writes item[attr::weight] = 1/count for every active interaction. */
struct WriteWeightFunc {
  size_t offset;
  onika::cuda::span<const uint8_t> active_in;
  onika::cuda::span<const uint32_t> count_in;

  template <typename InteractionT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t i, InteractionT& item) const {
    const size_t g = offset + i;
    if (active_in[g]) {
      item[attr::weight] = 1.0 / double(count_in[g]);
    }
  }
};

/** @brief Like WrapperForAll (classifier_for_all.hpp), but commits the mutated Interaction back via update(). */
template <InteractionType IT, typename K>
struct WrapperUpdateForAll {
  mutable typename ClassifierContainer<IT>::View data_;
  const K kernel_;

  WrapperUpdateForAll(typename ClassifierContainer<IT>::View& d, K& k) : data_(std::move(d)), kernel_(k) {}

  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t i) const {
    auto item = data_(i);
    kernel_(i, item);
    data_.update(i, item);
  }
};

template <typename ViewT, typename K>
WrapperUpdateForAll(ViewT&, K&) -> WrapperUpdateForAll<ViewT::kInteractionType, K>;

}  // namespace exaDEM

namespace onika {
namespace parallel {
template <int type, typename TMPLC, typename TMPLV>
struct ParallelForFunctorTraits<exaDEM::DetectPairFunc<type, TMPLC, TMPLV>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};

template <>
struct ParallelForFunctorTraits<exaDEM::CountPairsFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};

template <>
struct ParallelForFunctorTraits<exaDEM::WriteWeightFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};

template <exaDEM::InteractionType IT, typename K>
struct ParallelForFunctorTraits<exaDEM::WrapperUpdateForAll<IT, K>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika

namespace exaDEM {

/**
 * @brief "NumberWeight" strategy: sets item[attr::weight] = 1/N_ij for every active
 * particle-particle polyhedron contact, N_ij being the number of simultaneously active
 * contacts between the same pair (i, j). Used by contact_polyhedron to scale kn/kt down
 * for pairs with several simultaneous sub-contacts (e.g. two fragments still touching at
 * several points right after their bond breaks). Only needed for fragmentation setups;
 * add it to the pipeline before contact_polyhedron only where it matters.
 */
template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>>
class ComputeContactNumberWeightPolyhedron : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
  ADD_SLOT(NumberWeightScratch, scratch, PRIVATE, DocString{"Persistent buffers for the pair-counting pass"});

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

  /** @brief Launches DetectPairFunc for one contact type, recursing over [poly_type_start, poly_type_end]. */
  template <int type, typename TMPLC, typename TMPLV>
  void detect_pairs(Classifier& classifier, TMPLC* const cells, TMPLV* const gv, const shape* const shps,
                    const std::array<size_t, 4>& offsets, NumberWeightScratch& s) {
    auto [data, size] = classifier.get_info<InteractionType::ParticleParticle>(type);
    if (size > 0) {
      typename ClassifierContainer<InteractionType::ParticleParticle>::View view = data.view();
      DetectPairFunc<type, TMPLC, TMPLV> func = {cells,
                                                 gv,
                                                 shps,
                                                 offsets[type - poly_type_start],
                                                 onika::cuda::make_span(s.dn),
                                                 onika::cuda::make_span(s.key_lo),
                                                 onika::cuda::make_span(s.key_hi),
                                                 onika::cuda::make_span(s.active)};
      WrapperForAll wrapper = {view, func};
      ParallelForOptions opts;
      opts.omp_scheduling = OMP_SCHED_STATIC;
      parallel_for(size, wrapper, parallel_execution_context(), opts);
    }
    if constexpr (type + 1 <= poly_type_end) {
      detect_pairs<type + 1>(classifier, cells, gv, shps, offsets, s);
    }
  }

  /** @brief Launches WriteWeightFunc for one contact type, recursing over [poly_type_start, poly_type_end]. */
  template <int type>
  void write_weights(Classifier& classifier, const std::array<size_t, 4>& offsets, NumberWeightScratch& s) {
    auto [data, size] = classifier.get_info<InteractionType::ParticleParticle>(type);
    if (size > 0) {
      typename ClassifierContainer<InteractionType::ParticleParticle>::View view = data.view();
      WriteWeightFunc func = {offsets[type - poly_type_start], onika::cuda::make_const_span(s.active),
                              onika::cuda::make_const_span(s.count)};
      WrapperUpdateForAll wrapper = {view, func};
      ParallelForOptions opts;
      opts.omp_scheduling = OMP_SCHED_STATIC;
      parallel_for(size, wrapper, parallel_execution_context(), opts);
    }
    if constexpr (type + 1 <= poly_type_end) {
      write_weights<type + 1>(classifier, offsets, s);
    }
  }

  /** @brief Runs the 3 GPU passes (detect, count, write weight) for every polyhedron pair. */
  inline void execute() final {
    if (grid->number_of_cells() == 0) {
      return;
    }

    Classifier& classifier = *ic;
    const auto cells = grid->cells();
    auto* vertex_fields = cvf->data();
    const shape* const shps = shapes_collection->data();
    NumberWeightScratch& s = *scratch;

    std::array<size_t, 4> offsets{};
    size_t total = 0;
    for (int t = poly_type_start; t <= poly_type_end; t++) {
      offsets[t - poly_type_start] = total;
      total += classifier.get_size(t);
    }
    if (total == 0) {
      return;
    }

    // Sized to the current step's interaction count; only reallocates when
    // growing past the previously reserved capacity (see NumberWeightScratch).
    s.dn.resize(total);
    s.key_lo.resize(total);
    s.key_hi.resize(total);
    s.active.resize(total);
    s.count.resize(total);

    detect_pairs<poly_type_start>(classifier, cells, vertex_fields, shps, offsets, s);
    ONIKA_CU_DEVICE_SYNCHRONIZE();

    // Pass 2 (GPU): count, per interaction, how many active interactions
    // share its particle pair. Plain O(n) scan per interaction (not a
    // sort/hash-based grouping) -- see CountPairsFunc's comment -- but
    // stays entirely on device, no host round-trip.
    {
      CountPairsFunc func = {onika::cuda::make_const_span(s.key_lo), onika::cuda::make_const_span(s.key_hi),
                             onika::cuda::make_const_span(s.active), onika::cuda::make_span(s.count)};
      ParallelForOptions opts;
      opts.omp_scheduling = OMP_SCHED_STATIC;
      parallel_for(total, func, parallel_execution_context(), opts);
      ONIKA_CU_DEVICE_SYNCHRONIZE();
    }

    write_weights<poly_type_start>(classifier, offsets, s);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(number_weight_contact_polyhedron) {
  OperatorNodeFactory::instance()->register_factory("number_weight_contact_polyhedron",
                                                    make_grid_variant_operator<ComputeContactNumberWeightPolyhedron>);
}
}  // namespace exaDEM
