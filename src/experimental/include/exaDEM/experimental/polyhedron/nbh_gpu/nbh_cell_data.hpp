#pragma once
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
  ONIKA_HOST_DEVICE_FUNC inline void operator()(InteractionWrapper<IT>& wrapper,
                                                PlaceholderInteraction* __restrict__ data_ptr, size_t& shift, int start,
                                                int size) const {
    for (int j = start; j < start + size; j++) {
      // printf("CopierFunc::shift %lu\n", shift);
      data_ptr[shift++] = wrapper(j);
      // printf("CopierFunc -> shift %lu, id %lu id wrapper %lu\n", shift, data_ptr[shift-1].pair.pi.id,
      // wrapper(j).pair.pi.id);
    }
  }
};

struct CountActiveInteractionFunc {
  template <InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(InteractionWrapper<IT>& wrapper, size_t& count, int start,
                                                int size) const {
    for (int j = start; j < start + size; j++) {
      if (wrapper(j).active()) {
        // printf("count %lu\n", count);
        count++;
      }
    }
  }
};

struct CopierActiveInteractionFunc {
  template <InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(InteractionWrapper<IT>& wrapper,
                                                onika::cuda::span<PlaceholderInteraction> data, size_t& shift,
                                                int start, int size) const {
    for (int j = start; j < start + size; j++) {
      if (wrapper(j).active()) {
        data[shift++] = wrapper(j);
      }
    }
  }
};

/**
 * @brief Aggregates CellPairStorage's per-cell-pair particle-particle totals on top of
 * cell_storage's existing (particle-driver) contents, for every cell.
 *
 * @param cell_storage Per-cell storage to aggregate particle-particle totals into.
 * @param info Per-cell cell-pair range boundaries (same indexing as cell_storage).
 * @param pair_storage Cell-pair-level storage, already finalized.
 */
inline void add_particle_particle_totals(CellStorage& cell_storage, CellInteractionInformation& info,
                                         CellPairStorage& pair_storage) {
  const size_t n_cells = info.size();
#pragma omp parallel for
  for (size_t cell_idx = 0; cell_idx < n_cells; cell_idx++) {
    if (info.number_of_pair_cells_[cell_idx] == 0) continue;

    const size_t first = info.start_cell_[cell_idx];
    const size_t last = first + info.number_of_pair_cells_[cell_idx] - 1;
    const InteractionTypePerCellCounter pp_start = pair_storage.offset_[first];
    const InteractionTypePerCellCounter pp_end = pair_storage.offset_[last] + pair_storage.size_[last];

    cell_storage.offset_[cell_idx] = cell_storage.offset_[cell_idx] + pp_start;
    cell_storage.size_[cell_idx] = cell_storage.size_[cell_idx] + (pp_end - pp_start);
  }
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
 */
template <bool ghost_only, bool active_interaction, bool append = false>
void transfer_classifier_grid(size_t* cell_ptr, CellInteractionInformation& info, CellStorage& cell_storage,
                              InteractionWrapperAccessor& interaction_classifier_accessor, GridCellParticleInteraction& ges,
                              const int typeID_start = 0, const int typeID_end = InteractionTypeId::NTypes) {
  // Number of non-empty cells to process
  size_t ncells = info.start_cell_.size();

  // Parallel loop over non-empty cells
#pragma omp parallel for
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

    // Total number of interactions in this cell
    size_t number_of_interactions = 0;
    if constexpr (!active_interaction) {
      for (int typeID = typeID_start; typeID <= typeID_end; typeID++) {
        number_of_interactions += n_elem_per_type[typeID];
      }
    } else {
      CountActiveInteractionFunc counter;
      for (int typeID = typeID_start; typeID <= typeID_end; typeID++) {
        int start = first_elem_per_type[typeID];
        int size = n_elem_per_type[typeID];
        IDispatcher::dispatch(typeID, interaction_classifier_accessor, counter, number_of_interactions, start, size);
      }
    }

    // Reference to storage for this grid cell
    auto& storage = ges.m_data[owner_cell];
    auto& info_particles = storage.m_info;

    // Resize storage to fit all interactions
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
          IDispatcher::dispatch(typeID, interaction_classifier_accessor, copier, to_span(storage.m_data), shift, start, size);
        }
      }
    }

    assert(shift == old_size + number_of_interactions);

    // sorted according the particle position in the cell
    std::stable_sort(
        storage.m_data.begin(), storage.m_data.end(),
        [](const PlaceholderInteraction& a, const PlaceholderInteraction& b) { return a.sort_by_owner_p(b); });

    // reindex info
    int info_offset = 0;
    for (size_t i = 0; i < info_particles.size(); i++) {
      auto& [_offset, _size, _pid] = info_particles[i];
      _offset = info_offset;
      _size = 0;
      for (size_t j = info_offset; j < storage.m_data.size(); j++) {
        if (!data_ptr[j].consistent()) {
          data_ptr[j].print();
          color_log::mpi_error("transfer_classifier_grid", "This interacion is illformed");
        }
        if (_pid != data_ptr[j].owner().id_) {
          break;
        }
        _size++;
      }
      info_offset += _size;
    }
  }
}
}  // namespace exaDEM
