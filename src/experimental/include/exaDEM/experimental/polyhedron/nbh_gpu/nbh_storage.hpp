#pragma once

#include <onika/parallel/parallel_execution_operators.h>
#include <onika/parallel/parallel_execution_queue.h>

#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_utils.hpp>

namespace exaDEM {
/**
 * @brief Storage for neighborhood cell information on the host.
 * Stores mapping of each non-empty cell to its owner and partner cells.
 */
struct NbhCellHostStorage {
  // Vector type alias (host-side std::vector)
  template <typename T>
  using VectorT = std::vector<T>;
  VectorT<size_t> owner_cell_;    ///< Index of the cell that owns each non-empty cell
  VectorT<size_t> partner_cell_;  ///< Index of the interacting partner cell
  VectorT<uint8_t> ghost_;        ///< 1 if the partner cell is ghost
};

/**
 * @brief Device / unified memory storage for neighborhood cell-pair interactions.
 * Stores per-cell-pair interaction information including size, offsets, owner/partner
 * mapping and skip flags. Initialized from host-side storage.
 */
struct CellPairStorage {
  // Vector type alias using unified memory (GPU/CPU compatible)
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;

  struct View {
    // WARNING (TEMPORARY): mutable workaround for onika::cuda::span's const
    mutable onika::cuda::span<InteractionTypePerCellCounter>
        size_;  ///< Number of interactions per type for each cell pair
    mutable onika::cuda::span<InteractionTypePerCellCounter>
        offset_;                               ///< Offset for each interaction type per cell pair
    onika::cuda::span<size_t> owner_cell_;     ///< Owner cell index for each non-empty cell pair
    onika::cuda::span<size_t> partner_cell_;   ///< Partner cell index for each non-empty cell pair
    onika::cuda::span<uint8_t> ghost_;         ///< Partner cell is a ghost ?
    mutable onika::cuda::span<uint8_t> skip_;  ///< Flag to skip a cell pair in second pass
  };

  /// @brief Functor to reset member arrays per cell pair (size_/offset_/skip_ only --
  /// owner_cell_/partner_cell_/ghost_ in View are unused here).
  struct ResetCellMembers {
    View view_;

    ONIKA_HOST_DEVICE_FUNC
    inline void operator()(size_t i) const {
      view_.skip_[i] = true;  // used by the second pass
      for (size_t j = 0; j < InteractionTypeId::NTypes; j++) {
        view_.size_[i][j] = 0;
        view_.offset_[i][j] = 0;
      }
    }
  };

  VectorT<InteractionTypePerCellCounter> size_;    ///< Number of interactions per type for each cell pair
  VectorT<InteractionTypePerCellCounter> offset_;  ///< Offset for each interaction type per cell pair
  VectorT<size_t> owner_cell_;                     ///< Owner cell index for each non-empty cell pair
  VectorT<size_t> partner_cell_;                   ///< Partner cell index for each non-empty cell pair
  VectorT<uint8_t> ghost_;                         ///< Flag to skip a partner cell is a ghost
  VectorT<uint8_t> skip_;                          ///< Flag to skip a cell pair in the second pass

  /**
   * @brief Construct device storage from host-side storage.
   *
   * Copies owner and partner cell vectors from host, initializes size, offset, and skip arrays.
   * Performs a parallel reset of members.
   *
   * @tparam ExecCtx Execution context type for parallel_for
   * @param host Host-side storage
   * @param queue Queue the reset is dispatched onto (async: caller must synchronize before reading offset_/size_)
   * @param lane Execution lane to dispatch the reset onto
   * @param exec_ctx Execution context
   */
  template <typename ExecCtx>
  void reset(NbhCellHostStorage& host, onika::parallel::ParallelExecutionQueue& queue, int lane, ExecCtx& exec_ctx) {
    const size_t n_cells = host.owner_cell_.size();

    // Consistency check
    if (host.partner_cell_.size() != n_cells) {
      color_log::mpi_error("CellPairStorage", "Mismatch in host owner/partner cell sizes.");
    }

    // Lambda to transfer host vectors to device/unified memory
    auto transfer_data = [](auto& dest, auto& src, size_t elems) {
      using TDest = typename std::decay_t<decltype(dest)>::value_type;
      using TSrc = typename std::decay_t<decltype(src)>::value_type;
      static_assert(sizeof(TDest) == sizeof(TSrc), "Size mismatch in transfer");
      dest.resize(elems);
#ifdef ONIKA_CUDA_VERSION
      ONIKA_CU_MEMCPY_KIND(dest.data(), src.data(), elems * sizeof(TDest), onikaMemcpyHostToDevice);
#else
      dest = src;
#endif
    };

    // Transfer owner and partner cells
    transfer_data(owner_cell_, host.owner_cell_, n_cells);
    transfer_data(partner_cell_, host.partner_cell_, n_cells);
    transfer_data(ghost_, host.ghost_, n_cells);

    // Resize size, offset, skip vectors
    size_.resize(n_cells);
    offset_.resize(n_cells);
    skip_.resize(n_cells);

    // Reset members (skip flag and arrays)
    ResetCellMembers reset_func = {view()};

    onika::parallel::ParallelForOptions opts;
    opts.omp_scheduling = onika::parallel::OMP_SCHED_GUIDED;
    queue << onika::parallel::set_lane(lane)
          << parallel_for(n_cells, reset_func, exec_ctx("nbh_gpu::reset_cell_pair_storage"), opts)
          << onika::parallel::flush;
  }

  inline View view() const {
    const auto n = owner_cell_.size();
    if (size_.size() != n || offset_.size() != n || skip_.size() != n || ghost_.size() != n ||
        partner_cell_.size() != n) {
      color_log::mpi_error("CellPairStorage", "Inconsistent vector sizes.");
    }
    return View{to_span(size_),         to_span(offset_), to_span(owner_cell_),
                to_span(partner_cell_), to_span(ghost_),  to_span(skip_)};
  }
};

struct CellStorage {
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;

  struct View {
    // WARNING (TEMPORARY): mutable workaround for onika::cuda::span's const
    mutable onika::cuda::span<InteractionTypePerCellCounter> offset_;
    mutable onika::cuda::span<InteractionTypePerCellCounter> size_;
  };

  /// @brief Functor to reset the per-type counters of a CellStorage for a given cell.
  struct ResetCellCounters {
    View view_;

    ONIKA_HOST_DEVICE_FUNC
    inline void operator()(size_t i) const {
      for (size_t j = 0; j < InteractionTypeId::NTypes; j++) {
        view_.size_[i][j] = 0;
        view_.offset_[i][j] = 0;
      }
    }
  };

  VectorT<InteractionTypePerCellCounter> offset_;  ///< Offset (into the classifier) for each interaction type, per cell
  VectorT<InteractionTypePerCellCounter> size_;    ///< Number of interactions for each interaction type, per cell

  // size should be the number of non empty cells
  template <typename ExecCtx>
  void resize(size_t newsize, onika::parallel::ParallelExecutionQueue& queue, int lane, ExecCtx& exec_ctx) {
    size_.resize(newsize);
    offset_.resize(newsize);

    // Reset members
    ResetCellCounters reset_func = {view()};

    onika::parallel::ParallelForOptions opts;
    opts.omp_scheduling = onika::parallel::OMP_SCHED_GUIDED;
    queue << onika::parallel::set_lane(lane)
          << parallel_for(newsize, reset_func, exec_ctx("nbh_gpu::reset_cell_storage"), opts) << onika::parallel::flush;
  }

  inline View view() const { return View{to_span(offset_), to_span(size_)}; }
};

}  // namespace exaDEM

namespace onika {
namespace parallel {
template <>
struct ParallelForFunctorTraits<exaDEM::CellPairStorage::ResetCellMembers> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
template <>
struct ParallelForFunctorTraits<exaDEM::CellStorage::ResetCellCounters> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
