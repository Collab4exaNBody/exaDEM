#pragma once

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
 * @brief Functor to reset member arrays per cell.
 */
struct ResetCellMembers {
  InteractionTypePerCellCounter* __restrict__ size_;    ///< Pointer to array of sizes per interaction type
  InteractionTypePerCellCounter* __restrict__ offset_;  ///< Pointer to array of offsets per interaction type
  uint8_t* __restrict__ skip_;                          ///< Flag array for skipping cells

  /**
   * @brief Reset the data for a given cell index.
   * @param i Index of the cell to reset
   */
  ONIKA_HOST_DEVICE_FUNC
  inline void operator()(size_t i) const {
    skip_[i] = true;  // used by the second pass
    for (size_t j = 0; j < InteractionTypeId::NTypes; j++) {
      size_[i][j] = 0;
      offset_[i][j] = 0;
    }
  }
};

/**
 * @brief Device / unified memory storage for neighborhood cell interactions.
 * Stores per-cell interaction information including size, offsets, owner/partner
 * mapping and skip flags. Initialized from host-side storage.
 */
struct NbhCellStorage {
  // Vector type alias using unified memory (GPU/CPU compatible)
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;

  VectorT<InteractionTypePerCellCounter> size_;    ///< Number of interactions per type for each cell
  VectorT<InteractionTypePerCellCounter> offset_;  ///< Offset for each interaction type per cell
  VectorT<size_t> owner_cell_;                     ///< Owner cell index for each non-empty cell
  VectorT<size_t> partner_cell_;                   ///< Partner cell index for each non-empty cell
  VectorT<uint8_t> ghost_;                         ///< Flag to skip a partner cell is a ghost
  VectorT<uint8_t> skip_;                          ///< Flag to skip a cell in the second pass

  /**
   * @brief Construct device storage from host-side storage.
   *
   * Copies owner and partner cell vectors from host, initializes size, offset, and skip arrays.
   * Performs a parallel reset of members.
   *
   * @tparam ExecCtx Execution context type for parallel_for
   * @param host Host-side storage
   * @param exec_ctx Execution context
   */
  template <typename ExecCtx>
  void reset(NbhCellHostStorage& host, ExecCtx& exec_ctx) {
    const size_t n_cells = host.owner_cell_.size();

    // Consistency check
    if (host.partner_cell_.size() != n_cells) {
      color_log::mpi_error("NbhCellStorage", "Mismatch in host owner/partner cell sizes.");
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
    ResetCellMembers reset_func = {size_.data(), offset_.data(), skip_.data()};

    // Parallel execution over all cells
    onika::parallel::ParallelForOptions opts;
    opts.omp_scheduling = onika::parallel::OMP_SCHED_GUIDED;
    parallel_for(n_cells, reset_func, exec_ctx(), opts);
  }
};

/**
 * @brief Lightweight accessor to NbhCellStorage for fast per-cell access.
 */
struct NbhCellAccessor {
  InteractionTypePerCellCounter* __restrict__ size_;    ///< Number of interactions per type for each cell
  InteractionTypePerCellCounter* __restrict__ offset_;  ///< Offset for each interaction type per cell
  size_t* __restrict__ owner_cell_;                     ///< Owner cell index for each non-empty cell
  size_t* __restrict__ partner_cell_;                   ///< Partner cell index for each non-empty cell
  uint8_t* __restrict__ ghost_;                         ///< Partner cell is a ghost ?
  uint8_t* __restrict__ skip_;                          ///< Flag to skip a cell in second pass

 public:
  /**
   * @brief Construct accessor from NbhCellStorage.
   * @param storage Storage object containing vectors
   */
  NbhCellAccessor(NbhCellStorage& storage)
      : size_(storage.size_.data()),
        offset_(storage.offset_.data()),
        owner_cell_(storage.owner_cell_.data()),
        partner_cell_(storage.partner_cell_.data()),
        ghost_(storage.ghost_.data()),
        skip_(storage.skip_.data()) {
    // Basic consistency check
    const auto n = storage.owner_cell_.size();
    if (storage.size_.size() != n || storage.offset_.size() != n || storage.skip_.size() != n ||
        storage.ghost_.size() != n || storage.partner_cell_.size() != n) {
      color_log::mpi_error("NbhCellAccessor", "Inconsistent vector sizes in wrapped storage.");
    }
  }
};

}  // namespace exaDEM

namespace onika {
namespace parallel {
template <>
struct ParallelForFunctorTraits<exaDEM::NbhCellStorage> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
