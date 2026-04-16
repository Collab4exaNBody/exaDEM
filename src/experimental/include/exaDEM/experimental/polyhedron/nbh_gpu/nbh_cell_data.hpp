#pragma once
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_driver.hpp>

namespace exaDEM {
// Stores information about non-empty cells (GPU/CPU friendly)
struct CellInteractionInformation {

  // Vector type using unified memory (GPU/CPU compatible)
  template<typename T> using VectorT = onika::memory::CudaMMVector<T>;

  VectorT<size_t> start_cell;           // start index of interactions for each cell
  VectorT<size_t> number_of_pair_cells; // number of interaction pairs in each cell
  VectorT<uint8_t> update_ghost;        // flag indicating if ghost update is needed

  // Resize all vectors to a given size
  void resize(size_t size) {
    start_cell.resize(size);
    number_of_pair_cells.resize(size);
    update_ghost.resize(size);
  }

  // Prefetch all vectors to CPU memory asynchronously
  void prefetch_cpu(onikaStream_t& st) {
#ifdef ONIKA_CUDA_VERSION
    ONIKA_CU_MEM_PREFETCH(start_cell.data(),
                          start_cell.size() * sizeof(size_t),
                          cudaCpuDeviceId, st);

    ONIKA_CU_MEM_PREFETCH(number_of_pair_cells.data(),
                          number_of_pair_cells.size() * sizeof(size_t),
                          cudaCpuDeviceId, st);

    ONIKA_CU_MEM_PREFETCH(update_ghost.data(),
                          update_ghost.size() * sizeof(uint8_t),
                          cudaCpuDeviceId, st);
#endif
  }
};


struct CopierFunc {
  template<InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(
      InteractionWrapper<IT>& wrapper,
      PlaceholderInteraction* __restrict__ data_ptr,
      size_t& shift, int start, int size) const {
    for (int j = start; j < start+size; j++) {
      // printf("CopierFunc::shift %lu\n", shift);
      data_ptr[shift++] = wrapper(j);
      // printf("CopierFunc -> shift %lu, id %lu id wrapper %lu\n", shift, data_ptr[shift-1].pair.pi.id, wrapper(j).pair.pi.id);
    }
  }
};

struct CountActiveInteractionFunc {
  template<InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(
      InteractionWrapper<IT>& wrapper,
      size_t& count, int start, int size) const {
    for (int j = start; j < start+size; j++) {
      if (wrapper(j).active()) {
        // printf("count %lu\n", count);
        count++;
      }
    }
  }
};

struct CopierActiveInteractionFunc {
  template<InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(
      InteractionWrapper<IT>& wrapper,
      PlaceholderInteraction* __restrict__ data_ptr,
      size_t& shift, int start, int size) const {
    for (int j = start; j < start+size; j++) {
      // printf("shift %lu\n", shift);
      if (wrapper(j).active()) {
        data_ptr[shift++] = wrapper(j);
        // printf("shift %lu, id %lu id wrapper %lu\n", shift, data_ptr[shift-1].pair.pi.id, wrapper(j).pair.pi.id);
      }
    }
  }
};

/**
 * @brief Transfer ghost interactions from classifier to grid storage.
 * @param info CellInteractionInformation storing start indices, number of pair cells, and ghost flags.
 * @param classifier_helper Helper structure providing offsets and owner cell mapping.
 * @param classifier Classifier providing access to interactions per type.
 * @param ges GridCellParticleInteraction storage for the grid.
 * @param ghost_only If true, only transfers interactions flagged as ghost.
 */
template<bool ghost_only, bool active_interaction, bool append = false>
void transfer_classifier_grid(size_t* cell_ptr,
                              CellInteractionInformation& info,
                              NbhCellStorage& classifier_helper,
                              CellDriverStorage& classifier_helper_driver,
                              InteractionWrapperAccessor& iaccessor,
                              GridCellParticleInteraction& ges,
                              const int typeID_start = 0,
                              const int typeID_end = InteractionTypeId::NTypes - 1) {
  // Number of non-empty cells to process
  size_t ncells = info.start_cell.size();

  // Parallel loop over non-empty cells
#pragma omp parallel for
  for (size_t cell_idx = 0; cell_idx < ncells; cell_idx++) {
    // Skip if we only want ghost cells and this cell is not flagged
    if constexpr (ghost_only) {
      if (info.update_ghost[cell_idx] == 0) {
        continue;
      }
    }

    // Grid cell that owns this non-empty cell
    size_t owner_cell = cell_ptr[cell_idx];

    // Compute particle-particle contribution
    InteractionTypePerCellCounter particle_pair_start;
    InteractionTypePerCellCounter particle_pair_end;
    for (int k = 0; k < InteractionTypeId::NTypes; k++) {
        particle_pair_start[k] = 0;
        particle_pair_end[k] = 0;
    }

    if (info.number_of_pair_cells[cell_idx] > 0) {
        size_t first_interaction = info.start_cell[cell_idx];
        size_t last_interaction  = first_interaction + info.number_of_pair_cells[cell_idx] - 1;
        particle_pair_start = classifier_helper.offset[first_interaction];
        particle_pair_end   = classifier_helper.offset[last_interaction] 
                            + classifier_helper.size[last_interaction];
    }

    auto first_elem_per_type = particle_pair_start + classifier_helper_driver.offset[cell_idx];
    auto n_elem_per_type = particle_pair_end - particle_pair_start + classifier_helper_driver.size[cell_idx];

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
        IDispatcher::dispatch(typeID, iaccessor, counter, number_of_interactions, start, size);
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
      if (size>0) {
        if constexpr (!active_interaction) {
          CopierFunc copier;
          IDispatcher::dispatch(typeID, iaccessor, copier, data_ptr, shift, start, size);
        } else {
          CopierActiveInteractionFunc copier;
          IDispatcher::dispatch(typeID, iaccessor, copier, data_ptr, shift, start, size);
        }
      }
    }

    // Sanity check: copied all interactions
    assert(shift == old_size + number_of_interactions);

    // sorted according the particle position in the cell
    std::stable_sort(storage.m_data.begin(), storage.m_data.end(),
                     [](const PlaceholderInteraction& a, const PlaceholderInteraction& b) {
                     return a.sort_by_owner_p(b);
                     });

    // reindex info
    int info_offset = 0;
    for (size_t i = 0 ; i < info_particles.size() ; i++) {
      auto& [_offset, _size, _pid] = info_particles[i];
      _offset = info_offset;
      _size = 0;
      for (size_t j = info_offset; j < storage.m_data.size() ; j++) {
        if (!data_ptr[j].consistent()) {
          data_ptr[j].print();
          color_log::mpi_error("transfer_classifier_grid", "This interacion is illformed");
        }
        if (_pid != data_ptr[j].owner().id) {
          break;
        }
        _size++;
      }
      info_offset += _size;
    }
  }
}
}  // namespace exaDEM
