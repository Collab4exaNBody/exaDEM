#pragma once
#include <exaDEM/polyhedron/nbh_gpu/nbh_storage.hpp>

namespace exaDEM {
struct InteractionHistory {
  template<typename T> using VectorT = onika::memory::CudaMMVector<T>;
  VectorT<size_t> start;
  VectorT<size_t> size;
  VectorT<PlaceholderInteraction> data;

  void prefetch_gpu(onikaStream_t& st) {
    ONIKA_CU_MEM_PREFETCH(start.data(),
                          start.size() * sizeof(size_t),
                          0, st);
    ONIKA_CU_MEM_PREFETCH(size.data(),
                          size.size() * sizeof(size_t),
                          0, st);
    ONIKA_CU_MEM_PREFETCH(data.data(),
                          data.size() * sizeof(PlaceholderInteraction),
                          0, st);

  }
};

template<typename TMPLC>
void setup_history_clean_ges(TMPLC& cells,
                             size_t* idxs,
                             size_t ncells,
                             GridCellParticleInteraction& ges,
                             InteractionHistory& history,
                             onikaStream_t& st) {
  size_t all_active_interactions = 0;
  history.start.resize(ncells);
  history.size.resize(ncells);


  size_t sum = 0;
/*
#pragma omp parallel for reduction(inscan:sum)
  for(size_t i = 0 ; i < ncells ; i++) {
    // should contains only active interactions
    auto& storage = ges.m_data[idxs[i]];
    assert(interaction_test::check_extra_interaction_storage_consistency(
            storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));

    history.size[i] = storage.m_data.size();
    sum += storage.m_data.size();
    #pragma omp scan inclusive(sum)
    history.start[i] = sum;
  }
  */
#pragma omp parallel for reduction(+: sum)
  for(size_t i = 0; i < ncells; ++i)
  {
    sum += sum;
    history.size[i] = ges.m_data[idxs[i]].m_data.size();
  }

  history.start[0] = 0;
  for(size_t i = 1; i < ncells; ++i)
  {
    history.start[i] = history.start[i-1] + history.size[i-1];
  }

  history.data.resize(sum);

  // reset storage for new data;
# pragma omp parallel for
  for(size_t i = 0 ; i < ncells ; i++) {
    size_t cell_idx = idxs[i];
    auto& storage = ges.m_data[cell_idx];
    const size_t data_size = storage.m_data.size();
    PlaceholderInteraction* __restrict__ data_ptr = storage.m_data.data();

    std::memcpy(history.size.data() + history.size[cell_idx],
                data_ptr, data_size * sizeof(PlaceholderInteraction));

    size_t n_particles = cells[cell_idx].size();
    storage.initialize(n_particles);

    const uint64_t* __restrict__ id_a = cells[cell_idx][field::id];
    ONIKA_ASSUME_ALIGNED(id_a);
    auto& info_particles = storage.m_info;
    // Fill particle ids in the interaction storage
    for (size_t it = 0; it < n_particles; it++) {
      info_particles[it].pid = id_a[it];
    }
  }
  history.prefetch_gpu(st);
}
}  // namespace exaDEM
