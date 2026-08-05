#pragma once

#include <onika/cuda/cuda.h>
#include <onika/cuda/stl_adaptors.h>
#include <onika/memory/allocator.h>

#include <algorithm>
#include <tuple>
#include <vector>

#include <exaDEM/interaction/grid_cell_interaction.hpp>

namespace exaDEM {

struct ListOfIgnorePairs {
  std::vector<std::tuple<uint32_t, uint16_t, uint16_t>> list_;  // cell, particle_i, particle_j
};

inline void build_list_of_ignore_pair(ListOfIgnorePairs& ignore_pairs, const size_t* cell_indices,
                                      size_t active_cell_count, GridCellParticleInteraction& ges) {
  for (size_t i = 0; i < active_cell_count; i++) {
    const size_t cell_idx = cell_indices[i];  // compacted index i -> absolute grid cell
    auto& interactions = ges.m_data[cell_idx].m_data;
    for (auto& I : interactions) {
      if (I.persistent()) {
        ignore_pairs.list_.emplace_back(cell_idx, I.pair_.pi_.p_, I.pair_.pj_.p_);
      }
    }
  }
  std::sort(ignore_pairs.list_.begin(), ignore_pairs.list_.end());
  ignore_pairs.list_.erase(std::unique(ignore_pairs.list_.begin(), ignore_pairs.list_.end()), ignore_pairs.list_.end());
}

struct IgnorePairsGPU {
  struct PairKey {
    uint16_t pa_;
    uint16_t pb_;
  };

  /// @brief Trivially-copyable, kernel-launch-passable view of an IgnorePairsGPU.
  /// CudaMMVector isn't trivially copyable (it's a std::vector under the hood), so
  /// IgnorePairsGPU itself cannot be passed by value to a __global__ kernel: build it
  /// once on the host (build()), then call view() right before a kernel launch and
  /// pass the resulting View by value instead.
  struct View {
    onika::cuda::span<const uint32_t> cell_offset_;  // size == n_cells + 1
    onika::cuda::span<const PairKey> pairs_;

    /**
     * @brief Checks whether (pa, pb) is an ignored pair within the given cell.
     * @return true if the pair should be ignored, false otherwise (including when
     *         cell_id is out of range, e.g. the view is default-constructed/empty).
     */
    ONIKA_HOST_DEVICE_FUNC inline bool operator()(uint32_t cell_id, uint16_t pa, uint16_t pb) const {
      const size_t n_cells = cell_offset_.size() == 0 ? 0 : cell_offset_.size() - 1;
      if (cell_id >= n_cells) {
        return false;
      }

      uint32_t lo = cell_offset_[cell_id];
      uint32_t hi = cell_offset_[cell_id + 1];

      while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        const PairKey& p = pairs_[mid];
        if (p.pa_ < pa || (p.pa_ == pa && p.pb_ < pb)) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }

      return lo < cell_offset_[cell_id + 1] && pairs_[lo].pa_ == pa && pairs_[lo].pb_ == pb;
    }
  };

  onika::memory::CudaMMVector<uint32_t> cell_offset_;  // size == n_cells + 1 (CSR row offsets)
  onika::memory::CudaMMVector<PairKey> pairs_;         // per-cell contiguous, sorted by (pa_, pb_)

  /**
   * @brief Builds the CSR layout from a ListOfIgnorePairs.
   * @param ignore_pairs Source list, assumed sorted by (cell, pa, pb) and deduplicated
   *                      (true right after build_list_of_ignore_pair()).
   * @param n_cells Number of cells in the grid (cell_offset_ is sized n_cells + 1).
   * @warning Use OpenMP
   */
  inline void build(const ListOfIgnorePairs& ignore_pairs, size_t n_cells) {
    cell_offset_.assign(n_cells + 1, 0);
    for (auto& [cell, pa, pb] : ignore_pairs.list_) {
      assert(cell < n_cells);
      cell_offset_[cell + 1]++;
    }
    for (size_t c = 0; c < n_cells; c++) {
      cell_offset_[c + 1] += cell_offset_[c];
    }

    // list_ is already sorted by (cell, pa, pb), so its order already matches the
    // per-cell-contiguous layout pairs_ needs: element idx of list_ maps directly to
    // element idx of pairs_, so this is an independent, per-index transform.
    const size_t n_pairs = ignore_pairs.list_.size();
    pairs_.resize(n_pairs);
#pragma omp parallel for
    for (size_t idx = 0; idx < n_pairs; idx++) {
      auto& [cell, pa, pb] = ignore_pairs.list_[idx];
      pairs_[idx] = PairKey{pa, pb};
    }
  }

  /// @brief Builds a trivially-copyable View, for passing into __global__ kernels. Host-only:
  /// call this right before a kernel launch and pass the result by value.
  inline View view() const {
    return View{onika::cuda::span<const uint32_t>{onika::cuda::vector_data(cell_offset_),
                                                   onika::cuda::vector_size(cell_offset_)},
               onika::cuda::span<const PairKey>{onika::cuda::vector_data(pairs_), onika::cuda::vector_size(pairs_)}};
  }
};

}  // namespace exaDEM
