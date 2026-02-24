#pragma once

#include <exaDEM/polyhedron/nbh_gpu/nbh_utils.hpp>

namespace exaDEM {
/**
 * @brief Storage for neighborhood cell information on the host.
 * Stores mapping of each non-empty cell to its owner and partner cells.
 */
struct NbhCellHostStorage {
  // Vector type alias (host-side std::vector)
  template<typename T> using VectorT = std::vector<T>;
  VectorT<size_t> owner_cell;   ///< Index of the cell that owns each non-empty cell
  VectorT<size_t> partner_cell; ///< Index of the interacting partner cell
};

/**
 * @brief Functor to reset member arrays per cell.
 */
struct ResetCellMembers {
  InteractionTypePerCellCounter* __restrict__ _size;   ///< Pointer to array of sizes per interaction type
  InteractionTypePerCellCounter* __restrict__ _offset; ///< Pointer to array of offsets per interaction type
  uint8_t* __restrict__ _skip;                         ///< Flag array for skipping cells

  /**
   * @brief Reset the data for a given cell index.
   * @param i Index of the cell to reset
   */
  ONIKA_HOST_DEVICE_FUNC
      inline void operator()(size_t i) const {
        _skip[i] = true;  // used by the second pass
        for (size_t j = 0; j < InteractionTypeId::NTypes; j++) {
          _size[i][j];
          _offset[i][j];
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
  template<typename T> using VectorT = onika::memory::CudaMMVector<T>;

  VectorT<InteractionTypePerCellCounter> size;      ///< Number of interactions per type for each cell
  VectorT<InteractionTypePerCellCounter> offset;    ///< Offset for each interaction type per cell
  VectorT<size_t> owner_cell;                       ///< Owner cell index for each non-empty cell
  VectorT<size_t> partner_cell;                     ///< Partner cell index for each non-empty cell
  VectorT<uint8_t> skip;                            ///< Flag to skip a cell in the second pass

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
  template<typename ExecCtx>
  void reset(NbhCellHostStorage& host, ExecCtx& exec_ctx) {
    const size_t n_cells = host.owner_cell.size();

    // Consistency check
    if (host.partner_cell.size() != n_cells) {
      color_log::mpi_error(
          "NbhCellStorage",
          "Mismatch in host owner/partner cell sizes.");
    }

    // Lambda to transfer host vectors to device/unified memory
    auto transfer_data = [](auto& dest, auto& src, size_t elems) {
      using TDest = typename std::decay_t<decltype(dest)>::value_type;
      using TSrc  = typename std::decay_t<decltype(src)>::value_type;
      static_assert(sizeof(TDest) == sizeof(TSrc), "Size mismatch in transfer");
      dest.resize(elems);
      ONIKA_CU_MEMCPY_KIND(dest.data(), src.data(), elems * sizeof(TDest),
                           onikaMemcpyHostToDevice);
    };

    // Transfer owner and partner cells
    transfer_data(owner_cell, host.owner_cell, n_cells);
    transfer_data(partner_cell, host.partner_cell, n_cells);

    // Resize size, offset, skip vectors
    size.resize(n_cells);
    offset.resize(n_cells);
    skip.resize(n_cells);

    // Reset members (skip flag and arrays)
    ResetCellMembers reset_func = {size.data(), offset.data(), skip.data()};

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
  InteractionTypePerCellCounter* __restrict__ size;      ///< Number of interactions per type for each cell
  InteractionTypePerCellCounter* __restrict__ offset;    ///< Offset for each interaction type per cell
  uint8_t* __restrict__ skip;                            ///< Flag to skip a cell in second pass
  size_t* __restrict__ owner_cell;                       ///< Owner cell index for each non-empty cell
  size_t* __restrict__ partner_cell;                     ///< Partner cell index for each non-empty cell

 public:
  /**
   * @brief Construct accessor from NbhCellStorage.
   * @param storage Storage object containing vectors
   */
  NbhCellAccessor(NbhCellStorage& storage)
      : size(storage.size.data()),
      offset(storage.offset.data()),
      skip(storage.skip.data()),
      owner_cell(storage.owner_cell.data()),
      partner_cell(storage.partner_cell.data())
  {
    // Basic consistency check
    const auto n = storage.owner_cell.size();
    if (storage.size.size() != n
        || storage.offset.size() != n
        || storage.skip.size() != n
        || storage.partner_cell.size() != n) 
    {
      color_log::mpi_error(
          "NbhCellAccessor",
          "Inconsistent vector sizes in wrapped storage.");
    }
  }
};

}  // namespace exaDEM

namespace onika {
namespace parallel {
template<>
struct ParallelForFunctorTraits<exaDEM::NbhCellStorage> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
