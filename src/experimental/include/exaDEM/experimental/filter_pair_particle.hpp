#pragma once

#include <onika/cuda/cuda.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/memory/allocator.h>

#include <algorithm>
#ifdef __GLIBCXX__
#include <parallel/algorithm>
#endif
#include <compare>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_utils.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <vector>

namespace exaDEM {

struct IgnorePairEntry {
  uint32_t cell_;
  uint64_t id_a_;
  uint64_t id_b_;
  auto operator<=>(const IgnorePairEntry&) const = default;
};

struct ListOfIgnorePairs {
  std::vector<IgnorePairEntry> list_;
};

inline void build_list_of_ignore_pair(ListOfIgnorePairs& ignore_pairs, const size_t* cell_indices,
                                      size_t active_cell_count, GridCellParticleInteraction& ges) {
  ignore_pairs.list_.clear();

  ONIKA_CU_PROF_RANGE_PUSH("build_list_of_ignore_pair::build");
  // Order doesn't matter here (the list gets sorted right after), so each thread accumulates
  // into a private vector and merges it in once it's done, rather than counting cells upfront.
#pragma omp parallel
  {
    std::vector<IgnorePairEntry> local;

#pragma omp for schedule(guided) nowait
    for (size_t i = 0; i < active_cell_count; i++) {
      const size_t cell_idx = cell_indices[i];  // compacted index i -> absolute grid cell
      auto& interactions = ges.m_data[cell_idx].m_data;
      for (auto& I : interactions) {
        if (I.persistent()) {
          local.push_back(IgnorePairEntry{static_cast<uint32_t>(cell_idx), I.pair_.pi_.id_, I.pair_.pj_.id_});
        }
      }
    }

#pragma omp critical
    ignore_pairs.list_.insert(ignore_pairs.list_.end(), local.begin(), local.end());
  }
  ONIKA_CU_PROF_RANGE_POP();

  ONIKA_CU_PROF_RANGE_PUSH("build_list_of_ignore_pair::sort");
#ifdef __GLIBCXX__
  __gnu_parallel::sort(ignore_pairs.list_.begin(), ignore_pairs.list_.end());
#else
  std::sort(ignore_pairs.list_.begin(), ignore_pairs.list_.end());
#endif
  ONIKA_CU_PROF_RANGE_POP();

  ONIKA_CU_PROF_RANGE_PUSH("build_list_of_ignore_pair::unique");
  ignore_pairs.list_.erase(std::unique(ignore_pairs.list_.begin(), ignore_pairs.list_.end()), ignore_pairs.list_.end());
  ONIKA_CU_PROF_RANGE_POP();
}

struct PersistentInnerBonds {
  std::vector<PlaceholderInteraction> interactions_;
  std::vector<size_t> cell_idx_;  // compacted cell index (0..active_cell_count-1), parallel to interactions_
  std::vector<int> local_rank_;   // rank of this interaction within its cell's inner-bond group
};

inline void collect_persistent_inner_bonds(PersistentInnerBonds& persistent, CellStorage& cell_storage,
                                           const size_t* cell_indices, size_t active_cell_count,
                                           GridCellParticleInteraction& ges) {
  // Pass 1 (parallel): count persistent inner bonds per cell
  std::vector<int> counts(active_cell_count, 0);

  auto test = [](PlaceholderInteraction& I) -> bool {
    return I.type() == InteractionTypeId::InnerBond && I.persistent();
  };

#pragma omp parallel for schedule(guided)
  for (size_t i = 0; i < active_cell_count; i++) {
    const size_t cell_idx = cell_indices[i];
    auto& interactions = ges.m_data[cell_idx].m_data;
    int count = 0;
    for (auto& I : interactions) {
      if (test(I)) {
        count++;
      }
    }
    counts[i] = count;
    if (count > 0) {
      cell_storage.size_[i][InteractionTypeId::InnerBond] += count;
    }
  }

  // Sequential prefix sum over cells (cheap compared to the two interaction-level passes).
  std::vector<size_t> offsets(active_cell_count + 1, 0);
  for (size_t i = 0; i < active_cell_count; i++) {
    offsets[i + 1] = offsets[i] + counts[i];
  }

  persistent.interactions_.resize(offsets[active_cell_count]);
  persistent.cell_idx_.resize(offsets[active_cell_count]);
  persistent.local_rank_.resize(offsets[active_cell_count]);

  // Pass 2 (parallel): each cell writes into its own pre-computed slice.
#pragma omp parallel for schedule(guided)
  for (size_t i = 0; i < active_cell_count; i++) {
    if (counts[i] == 0) {
      continue;
    }
    const size_t cell_idx = cell_indices[i];
    auto& interactions = ges.m_data[cell_idx].m_data;
    size_t offset = offsets[i];
    int local_rank = 0;
    for (auto& I : interactions) {
      if (test(I)) {
        persistent.interactions_[offset] = I;
        persistent.cell_idx_[offset] = i;
        persistent.local_rank_[offset] = local_rank++;
        offset++;
      }
    }
  }
}

inline void fill_classifier_persistent_inner_bonds(const PersistentInnerBonds& persistent,
                                                   const CellStorage::View& cell_storage_accessor,
                                                   InteractionWrapperAccessor& interaction_classifier_accessor) {
  auto& wrapper =
      interaction_classifier_accessor.get_typed_accessor<InteractionType::InnerBond>(InteractionTypeId::InnerBond);
  const size_t n = persistent.interactions_.size();
  size_t total_interactions_copied = 0;
#pragma omp parallel for schedule(guided) reduction(+ : total_interactions_copied)
  for (size_t k = 0; k < n; k++) {
    const size_t cell_i = persistent.cell_idx_[k];
    const int idx = cell_storage_accessor.offset_[cell_i][InteractionTypeId::InnerBond] + persistent.local_rank_[k];
    PlaceholderInteraction item = persistent.interactions_[k];
    wrapper.set(idx, item);
    wrapper.update(idx, item);
    total_interactions_copied++;
  }
}

struct IgnorePairsGPU {
  struct PairKey {
    uint64_t id_a_;
    uint64_t id_b_;
  };

  struct View {
    onika::cuda::span<uint32_t> cell_offset_;  // size == n_cells + 1
    onika::cuda::span<PairKey> pairs_;

    ONIKA_HOST_DEVICE_FUNC inline bool has_pairs_to_ignore(uint32_t cell_id) const {
      if (cell_id + 1 >= cell_offset_.size()) {
        return false;
      }
      return cell_offset_[cell_id] != cell_offset_[cell_id + 1];
    }

    /**
     * @brief Checks whether (id_a, id_b) is an ignored pair within the given cell.
     */
    ONIKA_HOST_DEVICE_FUNC inline bool operator()(uint32_t cell_id, uint64_t id_a, uint64_t id_b) const {
      const size_t n_cells = cell_offset_.size() == 0 ? 0 : cell_offset_.size() - 1;
      if (cell_id >= n_cells) {
        return false;
      }

      uint32_t lo = cell_offset_[cell_id];
      uint32_t hi = cell_offset_[cell_id + 1];

      while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const PairKey& p = pairs_[mid];
        if (p.id_a_ < id_a || (p.id_a_ == id_a && p.id_b_ < id_b)) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }

      return lo < cell_offset_[cell_id + 1] && pairs_[lo].id_a_ == id_a && pairs_[lo].id_b_ == id_b;
    }
  };

  onika::memory::CudaMMVector<uint32_t> cell_offset_;  // size == n_cells + 1 (CSR row offsets)
  onika::memory::CudaMMVector<PairKey> pairs_;         // per-cell contiguous, sorted by (id_a_, id_b_)

  /**
   * @brief Builds the CSR layout from a ListOfIgnorePairs.
   * @param ignore_pairs Source list, assumed sorted by (cell, id_i, id_j) and deduplicated
   *                      (true right after build_list_of_ignore_pair()).
   * @param n_cells Number of cells in the grid (cell_offset_ is sized n_cells + 1).
   * @warning Use OpenMP
   */
  inline void build(const ListOfIgnorePairs& ignore_pairs, size_t n_cells) {
    cell_offset_.assign(n_cells + 1, 0);
    for (auto& [cell, id_i, id_j] : ignore_pairs.list_) {
      assert(cell < n_cells);
      cell_offset_[cell + 1]++;
    }
    for (size_t c = 0; c < n_cells; c++) {
      cell_offset_[c + 1] += cell_offset_[c];
    }

    const size_t n_pairs = ignore_pairs.list_.size();
    pairs_.resize(n_pairs);
#pragma omp parallel for
    for (size_t idx = 0; idx < n_pairs; idx++) {
      auto& [cell, id_i, id_j] = ignore_pairs.list_[idx];
      pairs_[idx] = PairKey{id_i, id_j};
    }
  }

  inline View view() const { return View{to_span(cell_offset_), to_span(pairs_)}; }
};

struct PersistentInteractionScratch {
  ListOfIgnorePairs ignore_pairs_;
  PersistentInnerBonds persistent_inner_bonds_;
  IgnorePairsGPU ignore_pairs_gpu_;
};

}  // namespace exaDEM
