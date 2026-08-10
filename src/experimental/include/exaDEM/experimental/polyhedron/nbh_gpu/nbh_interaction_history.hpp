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

  struct View {
    onika::cuda::span<size_t> start_;
    onika::cuda::span<size_t> size_;
    onika::cuda::span<PlaceholderInteraction> data_;
  };

  inline View view() {
    return View{onika::cuda::make_span(start_), onika::cuda::make_span(size_), onika::cuda::make_span(data_)};
  }

  void prefetch_gpu(onikaStream_t& st) {
    ONIKA_PREFETCH(start_.data(), start_.size() * sizeof(size_t), 0, st);
    ONIKA_PREFETCH(size_.data(), size_.size() * sizeof(size_t), 0, st);
    ONIKA_PREFETCH(data_.data(), data_.size() * sizeof(PlaceholderInteraction), 0, st);
  }
};

template <typename CellsT>
void setup_history_clean_ges(CellsT& cells, size_t* idxs, size_t ncells, GridCellParticleInteraction& ges,
                             InteractionHistory& history, onikaStream_t& st) {
  ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::resize_start_size_history");
  history.start_.resize(ncells);
  history.size_.resize(ncells);
  ONIKA_CU_PROF_RANGE_POP();

  size_t sum = 0;

  ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::fill_start_size_history");
#pragma omp parallel for reduction(+ : sum)
  for (size_t i = 0; i < ncells; i++) {
    size_t cell_idx = idxs[i];
    size_t size = 0;
    for (size_t j = 0; j < ges.m_data[cell_idx].m_data.size(); j++) {
      PlaceholderInteraction& I = ges.m_data[cell_idx].m_data[j];
      // persistent interactions are not stored in the history because they are trivially copier
      if (!I.persistent() && I.active()) {
        size++;
      }
    }
    sum += size;
    history.size_[i] = size;
  }

  history.start_[0] = 0;
  for (size_t i = 1; i < ncells; ++i) {
    history.start_[i] = history.start_[i - 1] + history.size_[i - 1];
  }
  ONIKA_CU_PROF_RANGE_POP();

  ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::resize_history_data");
  history.data_.resize(sum);
  ONIKA_CU_PROF_RANGE_POP();

  PlaceholderInteraction* __restrict__ history_data_ptr = history.data_.data();

  ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::fill_history_data");

  // reset storage for new data;
#pragma omp parallel for
  for (size_t i = 0; i < ncells; i++) {
    size_t cell_idx = idxs[i];
    auto& storage = ges.m_data[cell_idx];
    const size_t data_size = storage.m_data.size();

    size_t shift = history.start_[i];
    for (size_t j = 0; j < data_size; j++) {
      PlaceholderInteraction& I = storage.m_data[j];
      if (!I.persistent() && I.active()) {
        history_data_ptr[shift++] = I;
      }
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
  ONIKA_CU_PROF_RANGE_POP();

  // history.prefetch_gpu(st);
}

struct UpdateHistoryImplFunc {
  template <InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(typename ClassifierContainer<IT>::View& interactions_view,
                                                const PlaceholderInteraction& I, int begin, int end) const {
    for (int j = begin; j < end; j++) {
      if (interactions_view.same(j, I)) {
        interactions_view.update(j, I);
      }
    }
  }
};

struct UpdateHistoryFunc {
  InteractionHistory::View history_;
  CellStorage::View cell_storage_accessor_;
  ClassifierViewAccessor interaction_classifier_accessor_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(long idx) const {
    const UpdateHistoryImplFunc func;
    const size_t* __restrict__ start = history_.start_.data();
    const size_t* __restrict__ size = history_.size_.data();
    const PlaceholderInteraction* __restrict__ data = history_.data_.data();

    size_t begin = start[idx];
    size_t end = begin + size[idx];

    for (size_t i = begin; i < end; i++) {
      const PlaceholderInteraction& I = data[i];
      auto type = I.type();
      const int a = cell_storage_accessor_.offset_[idx][type];
      const int b = a + cell_storage_accessor_.size_[idx][type];
      IDispatcher::dispatch(type, interaction_classifier_accessor_, func, I, a, b);
    }
  }
};

}  // namespace exaDEM

namespace onika {
namespace parallel {
template <>
struct ParallelForFunctorTraits<exaDEM::UpdateHistoryFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
