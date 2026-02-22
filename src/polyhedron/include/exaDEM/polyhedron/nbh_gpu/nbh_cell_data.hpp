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

/**
 * @brief Transfer ghost interactions from classifier to grid storage.
 * @tparam TMPLC Type of the cell container.
 * @param cells Reference to the container of cells.
 * @param info CellInteractionInformation storing start indices, number of pair cells, and ghost flags.
 * @param classifier_helper Helper structure providing offsets and owner cell mapping.
 * @param classifier Classifier providing access to interactions per type.
 * @param ges GridCellParticleInteraction storage for the grid.
 * @param ghost_only If true, only transfers interactions flagged as ghost.
 */
template<typename TMPLC>
void transfer_classifier_ghosts(TMPLC& cells,
                                CellInteractionInformation& info,
                                NbhCellStorage& classifier_helper,
                                Classifier& classifier,
                                GridCellParticleInteraction& ges,
                                bool ghost_only) 
{
  // Number of non-empty cells to process
  size_t ncells = info.start_cell.size();

  // Accessor to fetch interactions by type from the classifier
  InteractionAccessor iaccessor;
  for (size_t type_id = 0; type_id < iaccessor.size(); type_id++) {
    iaccessor[type_id] = InteractionWrapper(
        classifier.get_data<ParticleParticle>(type_id));
  }

  // Parallel loop over non-empty cells
#pragma omp parallel for
  for (size_t cell_idx = 0; cell_idx < ncells; cell_idx++) {

    // Skip if we only want ghost cells and this cell is not flagged
    if (ghost_only && info.update_ghost[cell_idx] == 0) {
      continue;
    }

    // Indices of interactions for this cell in the classifier
    size_t first_interaction = info.start_cell[cell_idx];
    size_t last_interaction  = first_interaction + info.number_of_pair_cells[cell_idx] - 1;

    // Grid cell that owns this non-empty cell
    size_t owner_cell = classifier_helper.owner_cell[cell_idx];

    // Compute number of interactions per type for this cell
    auto& first_elem_per_type = classifier_helper.offset[first_interaction];
    auto n_elem_per_type = classifier_helper.offset[last_interaction] 
        + classifier_helper.size[last_interaction] 
        - first_elem_per_type;

    // Total number of interactions in this cell
    size_t number_of_interactions = 0;
    for (size_t type_id = 0; type_id < iaccessor.size(); type_id++) {
      number_of_interactions += n_elem_per_type[type_id];
    }

    // Reference to storage for this grid cell
    auto& storage = ges.m_data[owner_cell];
    auto& info_particles = storage.m_info;

    // Resize storage to fit all interactions
    storage.m_data.resize(number_of_interactions);
    PlaceholderInteraction* __restrict__ data_ptr = storage.m_data.data();

    // Copy classified interactions into storage
    size_t shift = 0;
    for (size_t type_id = 0; type_id < iaccessor.size(); type_id++) {
      auto& classified_interactions = iaccessor[type_id];
      for (size_t j = 0; j < n_elem_per_type[type_id]; j++) {
        data_ptr[shift++] = classified_interactions(j);
      }
    }

    // Sanity check: copied all interactions
    //std::assert(shift == number_of_interactions);
  }
}
}  // namespace exaDEM
