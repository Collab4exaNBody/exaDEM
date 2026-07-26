#pragma once
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_driver.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>

namespace exaDEM {
struct InteractionHistory {
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;
  VectorT<size_t> start_;
  VectorT<size_t> size_;
  VectorT<PlaceholderInteraction> data_;

  void prefetch_gpu(onikaStream_t& st) {
    ONIKA_PREFETCH(start_.data(), start_.size() * sizeof(size_t), 0, st);
    ONIKA_PREFETCH(size_.data(), size_.size() * sizeof(size_t), 0, st);
    ONIKA_PREFETCH(data_.data(), data_.size() * sizeof(PlaceholderInteraction), 0, st);
  }
};

template <typename TMPLC>
void setup_history_clean_ges(TMPLC& cells, size_t* idxs, size_t ncells, GridCellParticleInteraction& ges,
                             InteractionHistory& history, onikaStream_t& st) {
  history.start_.resize(ncells);
  history.size_.resize(ncells);

  size_t sum = 0;
/*
#pragma omp parallel for reduction(inscan:sum)
  for(size_t i = 0 ; i < ncells ; i++) {
    // should contains only active interactions
    auto& storage = ges.m_data[idxs[i]];
    assert(interaction_test::check_extra_interaction_storage_consistency(
            storage.number_of_particles(), storage.m_info.data(), storage.m_data.data()));

    history.size_[i] = storage.m_data.size();
    sum += storage.m_data.size();
    #pragma omp scan inclusive(sum)
    history.start_[i] = sum;
  }
  */
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < ncells; ++i) {
    size_t cell_idx = idxs[i];
    sum += ges.m_data[cell_idx].m_data.size();
    history.size_[i] = ges.m_data[cell_idx].m_data.size();
  }

  history.start_[0] = 0;
  for (size_t i = 1; i < ncells; ++i) {
    history.start_[i] = history.start_[i - 1] + history.size_[i - 1];
  }

  history.data_.resize(sum);

  PlaceholderInteraction* __restrict__ history_data_ptr = history.data_.data();
  // reset storage for new data;
#pragma omp parallel for
  for (size_t i = 0; i < ncells; i++) {
    size_t cell_idx = idxs[i];
    auto& storage = ges.m_data[cell_idx];
    const size_t data_size = storage.m_data.size();

    if (data_size > 0) {
      PlaceholderInteraction* __restrict__ data_ptr = storage.m_data.data();
      std::memcpy(history_data_ptr + history.start_[i], data_ptr, data_size * sizeof(PlaceholderInteraction));
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
  // history.prefetch_gpu(st);
}

struct UpdateHistoryImplFunc {
  template <InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(InteractionWrapper<IT>& wrapper, const PlaceholderInteraction& I,
                                                int begin, int end) const {
    for (int j = begin; j < end; j++) {
      if (wrapper.same(j, I)) {
        wrapper.update(j, I);
      }
    }
  }
};

struct UpdateHistoryFunc {
  size_t* __restrict__ start_;
  size_t* __restrict__ size_;
  PlaceholderInteraction* __restrict__ data_;
  size_t* __restrict__ start_cell_;
  size_t* __restrict__ number_of_pair_cells_;
  NbhCellAccessor accessor_shift_;
  CellDriverGPUAcessor driver_accessor_;
  InteractionWrapperAccessor classifier_accessor_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(long idx) const {
    const UpdateHistoryImplFunc func;
    size_t begin = start_[idx];
    size_t end = begin + size_[idx];

    for (size_t i = begin; i < end; i++) {
      const PlaceholderInteraction& I = data_[i];
      auto type = I.type();
      int a, b;
      if (type >= get_first_id<InteractionType::ParticleDriver>() &&
          type <= get_last_id<InteractionType::ParticleDriver>()) {
        a = driver_accessor_.offset_[idx][type];
        b = a + driver_accessor_.size_[idx][type];
      } else {
        if (number_of_pair_cells_[idx] > 0) {
          size_t first_block_id = start_cell_[idx];
          size_t last_block_id = first_block_id + number_of_pair_cells_[idx] - 1;
          a = accessor_shift_.offset_[first_block_id][type];
          b = accessor_shift_.offset_[last_block_id][type] + accessor_shift_.size_[last_block_id][type];
        } else {
          a = 0;
          b = 0;
        }
      }
      IDispatcher::dispatch(type, classifier_accessor_, func, I, a, b);
    }
  }
};

}  // namespace exaDEM
