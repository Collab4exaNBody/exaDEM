#pragma once
#include <exaDEM/polyhedron/nbh_gpu/nbh_storage.hpp>

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
      data_ptr[shift++] = wrapper(j);
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
      if (wrapper(j).active()) {
        data_ptr[shift++] = wrapper(j);
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
template<bool ghost_only, bool active_interaction>
void transfer_classifier_grid(size_t* cell_ptr,
                              CellInteractionInformation& info,
                              NbhCellStorage& classifier_helper,
                              InteractionWrapperAccessor& iaccessor,
                              GridCellParticleInteraction& ges) {
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

    if (info.number_of_pair_cells[cell_idx] == 0) {
      continue;
    }

    // Indices of interactions for this cell in the classifier
    size_t first_interaction = info.start_cell[cell_idx];
    size_t last_interaction  = first_interaction + info.number_of_pair_cells[cell_idx] - 1;

    // Grid cell that owns this non-empty cell
    size_t owner_cell = cell_ptr[cell_idx];

    // Compute number of interactions per type for this cell
    auto& first_elem_per_type = classifier_helper.offset[first_interaction];
    auto n_elem_per_type = classifier_helper.offset[last_interaction] 
        + classifier_helper.size[last_interaction] 
        - first_elem_per_type;

    // Total number of interactions in this cell
    size_t number_of_interactions = 0;
    if constexpr (!active_interaction) {
      for (size_t type_id = 0; type_id < InteractionTypeId::NTypes; type_id++) {
        number_of_interactions += n_elem_per_type[type_id];
      }
    } else {
      CountActiveInteractionFunc counter;
      for (size_t type_id = 0; type_id < InteractionTypeId::NTypes; type_id++) {
        int start = first_elem_per_type[type_id];
        int size = n_elem_per_type[type_id];
        IDispatcher::dispatch(type_id, iaccessor, counter, number_of_interactions, start, size);
      }
    }

    // Reference to storage for this grid cell
    auto& storage = ges.m_data[owner_cell];
    auto& info_particles = storage.m_info;

    // Resize storage to fit all interactions
    storage.m_data.resize(number_of_interactions);

    if (number_of_interactions == 0) {
      continue;
    }

    PlaceholderInteraction* __restrict__ data_ptr = storage.m_data.data();

    // Copy classified interactions into storage
    size_t shift = 0;
    for (size_t type_id = 0; type_id < InteractionTypeId::NTypes; type_id++) {
      int start = first_elem_per_type[type_id];
      int size = n_elem_per_type[type_id];
      if (size>0) {
        lout << "type " << type_id << std::endl;
        lout << "size " << size << std::endl;
        lout << "shift " << shift << std::endl;
        if constexpr (!active_interaction) {
          CopierFunc copier;
          IDispatcher::dispatch(type_id, iaccessor, copier, data_ptr, shift, start, size);
        } else {
          CopierActiveInteractionFunc copier;
          IDispatcher::dispatch(type_id, iaccessor, copier, data_ptr, shift, start, size);
        }
      }
    }

    // Sanity check: copied all interactions
    assert(shift == number_of_interactions);

    // sorted according the particle position in the cell
    std::stable_sort(storage.m_data.begin(), storage.m_data.end(),
                     [](const PlaceholderInteraction& a, const PlaceholderInteraction& b) {
                     return a.sort_by_owner_p(b);
                     });

    // reindex info
    int info_offset = 0;

    lout << "Particle Info size: " << info_particles.size() << std::endl;
    for(size_t i = 0 ; i < info_particles.size() ; i++) {
      auto& [_offset, _size, _pid] = info_particles[i];
      _offset = info_offset;
      _size = 0;
      for(size_t j = info_offset; j < storage.m_data.size() ; j++) {
        if (_pid != data_ptr[j].owner().id) {
          break;
        }
        _size++;
      }
      info_offset += _size;
      lout << "Particle Info[" << _pid << "]: " << _offset << " " << _size << std::endl;
    }
    lout << "end" << std::endl;
  }
}
}  // namespace exaDEM
