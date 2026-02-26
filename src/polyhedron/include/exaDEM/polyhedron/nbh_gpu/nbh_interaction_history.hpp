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
  for (size_t i = 0; i < ncells; ++i) {
    size_t cell_idx = idxs[i];
    sum += ges.m_data[cell_idx].m_data.size();
    history.size[i] = ges.m_data[cell_idx].m_data.size();
    /*
    for (size_t j = 0 ; j < ges.m_data[idxs[i]].m_data.size() ; j++) {
      ges.m_data[idxs[i]].m_data[j].print();
    }*/
  }

  history.start[0] = 0;
  for(size_t i = 1; i < ncells; ++i) {
    history.start[i] = history.start[i-1] + history.size[i-1];
  }

  history.data.resize(sum);

  PlaceholderInteraction* __restrict__ history_data_ptr = history.data.data();
  // reset storage for new data;
# pragma omp parallel for
  for(size_t i = 0 ; i < ncells ; i++) {
    size_t cell_idx = idxs[i];
    auto& storage = ges.m_data[cell_idx];
    const size_t data_size = storage.m_data.size();

    if (data_size > 0) {
      PlaceholderInteraction* __restrict__ data_ptr = storage.m_data.data();
      std::memcpy(history_data_ptr + history.start[i],
                  data_ptr, data_size * sizeof(PlaceholderInteraction));
    }

    size_t n_particles = cells[cell_idx].size();
    storage.initialize(n_particles);  // clean storage.m_data
    const uint64_t* id_a = cells[cell_idx][field::id];
    // ONIKA_ASSUME_ALIGNED(id_a);
    auto& info_particles = storage.m_info;

    // Fill particle ids in the interaction storage
    for (size_t it = 0; it < n_particles; it++) {
      info_particles[it].pid = id_a[it];
    }
  }
  //history.prefetch_gpu(st);
}

struct UpdateHistoryImplFunc {

  template<InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(
      InteractionWrapper<IT>& wrapper,
      const PlaceholderInteraction& I,
      int begin,
      int end) const {
    for (int j = begin ; j < end ; j++) {
      if (wrapper.same(j, I)) {
        wrapper.update(j,I);
      }
    }
  }
};

struct UpdateHistoryFunc {
  size_t* __restrict__ start;
  size_t* __restrict__ size;
  PlaceholderInteraction* __restrict__ data;
  size_t* __restrict__ start_cell;
  size_t* __restrict__ number_of_pair_cells;
  NbhCellAccessor accessor_shift;
  InteractionWrapperAccessor classifier_accessor;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(long idx) const {
    const UpdateHistoryImplFunc func;
    size_t begin = start[idx];
    size_t end = begin + size[idx];
    size_t first_block_id = start_cell[idx];
    size_t last_block_id = first_block_id + number_of_pair_cells[idx] -1;

    auto& c_begin = accessor_shift.offset[first_block_id];
    auto c_end = accessor_shift.offset[last_block_id] + accessor_shift.size[last_block_id];

    for (size_t i = begin ; i < end ; i++) {
      const PlaceholderInteraction& I = data[i];
      auto type = I.type();
      int a = c_begin[type];
      int b = c_end[type];
      IDispatcher::dispatch(type, classifier_accessor, func, I, a, b);
    }
  }
};

}  // namespace exaDEM
