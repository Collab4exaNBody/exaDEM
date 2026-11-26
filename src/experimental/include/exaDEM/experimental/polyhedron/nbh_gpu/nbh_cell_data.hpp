#pragma once
#include <omp.h>

#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_driver.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>

static inline onikaError_t ONIKA_PREFETCH(const void* ptr, size_t size, int device, onikaStream_t stream) {
#ifdef ONIKA_CUDA_VERSION
#if CUDART_VERSION >= 13000
  cudaMemLocation loc = {(device == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice,
                         (device == cudaCpuDeviceId) ? 0 : device};
  return cudaMemPrefetchAsync(ptr, size, loc, 0, stream);
#else
  return cudaMemPrefetchAsync(ptr, size, device, stream);
#endif
#else
  return onikaError_t(0);
#endif
}

namespace exaDEM {
/** @brief Active interactions of the classifier PP tail (see Classifier::append_pp_tail), grouped by owner cell. */
struct PPTailByCell {
  std::vector<PlaceholderInteraction> items;  // sorted by owner cell, classifier order within a cell
  std::vector<uint32_t> cells;                // distinct owner cells, ascending
  std::vector<uint32_t> begin;                // items of cells[c] are [begin[c], begin[c + 1])

  std::pair<size_t, size_t> range(size_t cell) const {
    auto it = std::lower_bound(cells.begin(), cells.end(), cell);
    if (it == cells.end() || *it != cell) {
      return {0, 0};
    }
    const size_t c = it - cells.begin();
    return {begin[c], begin[c + 1]};
  }
};

// Stores information about non-empty cells (GPU/CPU friendly)
struct CellInteractionInformation {
  // Vector type using unified memory (GPU/CPU compatible)
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;

  VectorT<size_t> start_cell_;            // start index of interactions for each cell
  VectorT<size_t> number_of_pair_cells_;  // number of interaction pairs in each cell
  VectorT<uint8_t> update_ghost_;         // flag indicating if ghost update is needed

  // Resize all vectors to a given size
  void resize(size_t size) {
    start_cell_.resize(size);
    number_of_pair_cells_.resize(size);
    update_ghost_.resize(size);
  }

  size_t size() const { return start_cell_.size(); }

  // Prefetch all vectors to CPU memory asynchronously
  void prefetch_cpu(onikaStream_t& st) {
#ifdef ONIKA_CUDA_VERSION
    ONIKA_PREFETCH(start_cell_.data(), start_cell_.size() * sizeof(size_t), cudaCpuDeviceId, st);
    ONIKA_PREFETCH(number_of_pair_cells_.data(), number_of_pair_cells_.size() * sizeof(size_t), cudaCpuDeviceId, st);
    ONIKA_PREFETCH(update_ghost_.data(), update_ghost_.size() * sizeof(uint8_t), cudaCpuDeviceId, st);
#endif
  }
};

struct CopierFunc {
  template <InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(typename ClassifierContainer<IT>::View& interactions_view,
                                                PlaceholderInteraction* __restrict__ data_ptr, size_t& shift, int start,
                                                int size) const {
    for (int j = start; j < start + size; j++) {
      // printf("CopierFunc::shift %lu\n", shift);
      data_ptr[shift++] = interactions_view(j);
      // printf("CopierFunc -> shift %lu, id %lu id interactions_view %lu\n", shift, data_ptr[shift-1].pair.pi.id,
      // interactions_view(j).pair.pi.id);
    }
  }
};

struct CopierActiveInteractionFunc {
  template <InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(typename ClassifierContainer<IT>::View& interactions_view,
                                                onika::cuda::span<PlaceholderInteraction> data, size_t& shift,
                                                int start, int size) const {
    for (int j = start; j < start + size; j++) {
      if (interactions_view(j).active()) {
        data[shift++] = interactions_view(j);
      }
    }
  }
};

/// @brief GPU functor backing add_particle_particle_totals: one thread per cell, no cross-cell dependency.
struct AddParticleParticleTotalsFunc {
  onika::cuda::span<size_t> number_of_pair_cells_;
  onika::cuda::span<size_t> start_cell_;
  onika::cuda::span<InteractionTypePerCellCounter> pp_offset_;
  onika::cuda::span<InteractionTypePerCellCounter> pp_size_;
  mutable onika::cuda::span<InteractionTypePerCellCounter> cell_offset_;
  mutable onika::cuda::span<InteractionTypePerCellCounter> cell_size_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t cell_idx) const {
    if (number_of_pair_cells_[cell_idx] == 0) return;

    const size_t first = start_cell_[cell_idx];
    const size_t last = first + number_of_pair_cells_[cell_idx] - 1;
    const InteractionTypePerCellCounter pp_start = pp_offset_[first];
    const InteractionTypePerCellCounter pp_end = pp_offset_[last] + pp_size_[last];

    cell_offset_[cell_idx] = cell_offset_[cell_idx] + pp_start;
    cell_size_[cell_idx] = cell_size_[cell_idx] + (pp_end - pp_start);
  }
};

/**
 * @brief Aggregates CellPairStorage's per-cell-pair particle-particle totals on top of
 * cell_storage's existing (particle-driver) contents, for every cell.
 *
 * @param cell_storage Per-cell storage to aggregate particle-particle totals into.
 * @param info Per-cell cell-pair range boundaries (same indexing as cell_storage).
 * @param pair_storage Cell-pair-level storage, already finalized.
 * @param queue/lane/exec_ctx/opts Dispatched async on `lane`; the caller is responsible for
 *   synchronizing before cell_storage.offset_/size_ are read.
 */
template <typename ExecCtx>
inline void add_particle_particle_totals(CellStorage& cell_storage, CellInteractionInformation& info,
                                         CellPairStorage& pair_storage, onika::parallel::ParallelExecutionQueue& queue,
                                         int lane, ExecCtx& exec_ctx, const onika::parallel::ParallelForOptions& opts) {
  const size_t n_cells = info.size();
  auto cell_accessor = cell_storage.view();

  AddParticleParticleTotalsFunc func{
      to_span(info.number_of_pair_cells_), to_span(info.start_cell_), to_span(pair_storage.offset_),
      to_span(pair_storage.size_),         cell_accessor.offset_,     cell_accessor.size_};

  queue << onika::parallel::set_lane(lane)
        << parallel_for(n_cells, func, exec_ctx("nbh_gpu::add_particle_particle_totals"), opts)
        << onika::parallel::flush;
}

/**
 * @brief Transfer ghost interactions from classifier to grid storage.
 * @param info CellInteractionInformation storing start indices, number of pair cells, and ghost flags.
 * @param cell_storage Per-cell interaction counts/offsets, across all interaction types
 *                     (see CellStorage -- must already be aggregated via
 *                     add_particle_particle_totals()).
 * @param classifier Classifier providing access to interactions per type.
 * @param ges GridCellParticleInteraction storage for the grid.
 * @param ghost_only If true, only transfers interactions flagged as ghost.
 * @param tail If set, active PP tail interactions also merged into their owner cell (an equal interaction is replaced).
 */
template <bool ghost_only, bool active_interaction, bool append = false>
void transfer_classifier_grid(size_t* cell_ptr, CellInteractionInformation& info, CellStorage& cell_storage,
                              ClassifierViewAccessor& interaction_classifier_accessor, GridCellParticleInteraction& ges,
                              const int typeID_start = 0, const int typeID_end = InteractionTypeId::NTypes - 1,
                              const PPTailByCell* tail = nullptr) {
  // Number of non-empty cells to process
  size_t ncells = info.start_cell_.size();
  size_t n_tail_merged = 0;

  // Parallel loop over non-empty cells
#pragma omp parallel for reduction(+ : n_tail_merged)
  for (size_t cell_idx = 0; cell_idx < ncells; cell_idx++) {
    // Skip if we only want ghost cells and this cell is not flagged
    if constexpr (ghost_only) {
      if (info.update_ghost_[cell_idx] == 0) {
        continue;
      }
    }

    // Grid cell that owns this non-empty cell
    size_t owner_cell = cell_ptr[cell_idx];

    auto& first_elem_per_type = cell_storage.offset_[cell_idx];
    auto& n_elem_per_type = cell_storage.size_[cell_idx];

    // Upper bound on the number of interactions in this cell (exact count when !active_interaction,
    // an over-estimate otherwise since some of them may be inactive). Known without touching any
    // individual interaction, so it's always cheap to compute up front.
    size_t number_of_interactions = 0;
    for (int typeID = typeID_start; typeID <= typeID_end; typeID++) {
      number_of_interactions += n_elem_per_type[typeID];
    }
    auto [tail_begin, tail_end] = tail ? tail->range(owner_cell) : std::pair<size_t, size_t>{0, 0};
    number_of_interactions += tail_end - tail_begin;

    // Reference to storage for this grid cell
    auto& storage = ges.m_data[owner_cell];
    auto& info_particles = storage.m_info;

    // Resize storage to fit all interactions (the upper bound above). When active_interaction is
    // true this may over-allocate; it is shrunk back down to the real count below, once it is
    // known, after the single copy pass -- resizing a vector down never reallocates, so this costs
    // nothing and avoids a separate counting pass over every interaction.
    size_t old_size = 0;
    if constexpr (append) {
      old_size = storage.m_data.size();
      storage.m_data.resize(old_size + number_of_interactions);
    } else {
      storage.m_data.resize(number_of_interactions);
    }

    if (number_of_interactions == 0) {
      continue;
    }

    PlaceholderInteraction* __restrict__ data_ptr = storage.m_data.data();

    // Copy classified interactions into storage
    size_t shift = old_size;

    for (int typeID = typeID_start; typeID <= typeID_end; typeID++) {
      int start = first_elem_per_type[typeID];
      int size = n_elem_per_type[typeID];
      if (size > 0) {
        if constexpr (!active_interaction) {
          CopierFunc copier;
          IDispatcher::dispatch(typeID, interaction_classifier_accessor, copier, data_ptr, shift, start, size);
        } else {
          CopierActiveInteractionFunc copier;
          IDispatcher::dispatch(typeID, interaction_classifier_accessor, copier, to_span(storage.m_data), shift, start,
                                size);
        }
      }
    }
    for (size_t k = tail_begin; k < tail_end; k++) {
      data_ptr[shift++] = tail->items[k];
    }
    n_tail_merged += tail_end - tail_begin;

    if constexpr (active_interaction) {
      assert(shift <= old_size + number_of_interactions);
      // Shrink back down to the exact number of active interactions copied above (number_of_interactions
      // was only an upper bound). A vector resize() to a smaller size never reallocates, so this is free.
      storage.m_data.resize(shift);
      data_ptr = storage.m_data.data();
    } else {
      assert(shift == old_size + number_of_interactions);
    }

    // sorted according the particle position in the cell
    std::stable_sort(
        storage.m_data.begin(), storage.m_data.end(),
        [](const PlaceholderInteraction& a, const PlaceholderInteraction& b) { return a.sort_by_owner_p(b); });

    if (tail_end > tail_begin) {
      // equal interactions are adjacent, the tail one last: keep it
      auto& d = storage.m_data;
      size_t w = 0;
      for (size_t k = 0; k < d.size(); k++) {
        if (k + 1 < d.size() && d[k].owner().p_ == d[k + 1].owner().p_ && d[k].pair_ == d[k + 1].pair_) {
          continue;
        }
        d[w++] = d[k];
      }
      d.resize(w);
      data_ptr = d.data();
    }

    // reindex info
    int info_offset = 0;
    for (size_t i = 0; i < info_particles.size(); i++) {
      auto& [_offset, _size, _pid] = info_particles[i];
      _offset = info_offset;
      _size = 0;
      for (size_t j = info_offset; j < storage.m_data.size(); j++) {
#ifndef NDEBUG
        if (!data_ptr[j].consistent()) {
          data_ptr[j].print();
          color_log::mpi_error("transfer_classifier_grid", "This interacion is illformed");
        }
#endif
        if (_pid != data_ptr[j].owner().id_) {
          break;
        }
        _size++;
      }
      info_offset += _size;
    }
  }

  if (tail && n_tail_merged != tail->items.size()) {
    color_log::warning("transfer_classifier_grid",
                       std::to_string(tail->items.size() - n_tail_merged) +
                           " PP tail interactions have an owner cell outside the traversal.");
  }
}
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <>
struct ParallelForFunctorTraits<exaDEM::AddParticleParticleTotalsFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
