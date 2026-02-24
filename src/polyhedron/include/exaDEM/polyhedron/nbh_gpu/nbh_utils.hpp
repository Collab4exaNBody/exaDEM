#pragma once

#include <onika/oarray.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>
#include <exaDEM/shapes.hpp>

namespace exaDEM {
static constexpr int ParticleParticleSize = 4;
typedef onika::oarray_t<int,
        InteractionTypeId::NTypes> InteractionTypePerCellCounter;
typedef onika::oarray_t<InteractionWrapper<ParticleParticle>,
        InteractionTypeId::NTypes> InteractionAccessor;
typedef onika::oarray_t<InteractionWrapper<ParticleParticle>,
        ParticleParticleSize> InteractionParticleAccessor;

inline InteractionTypePerCellCounter operator+(const InteractionTypePerCellCounter& a,
                                               const InteractionTypePerCellCounter& b) {
  InteractionTypePerCellCounter res;
  for (size_t i = 0; i < InteractionTypeId::NTypes ; i++) {
    res[i] = a[i] + b[i];
  }
  return res;
}

inline InteractionTypePerCellCounter operator-(const InteractionTypePerCellCounter& a,
                                               const InteractionTypePerCellCounter& b) {
  InteractionTypePerCellCounter res;
  for (size_t i = 0; i < InteractionTypeId::NTypes ; i++) {
    res[i] = a[i] - b[i];
  }
  return res;
}

inline void debug_print(InteractionTypePerCellCounter& in) {
  std::cout << "VertexVertex = "
      << in[InteractionTypeId::VertexVertex] << std::endl;
  std::cout << "VertexEdge   = "
      << in[InteractionTypeId::VertexEdge] << std::endl;
  std::cout << "VertexFace   = "
      << in[InteractionTypeId::VertexFace] << std::endl;
  std::cout << "EdgeEdge     = "
      << in[InteractionTypeId::EdgeEdge] << std::endl;
}

inline void debug_print(InteractionTypePerCellCounter& in1,
                        InteractionTypePerCellCounter& in2) {
  InteractionTypePerCellCounter sum;
  for(size_t i=0 ; i<InteractionTypeId::NTypes ; i++) {
    sum[i] = in1[i] + in2[i];
  }
  debug_print(sum);
}

struct PrefixSumInteractionTypePerCellCounter {
  InteractionTypePerCellCounter* const offset;
  InteractionTypePerCellCounter* const size;
  size_t n_elem;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t id) const {
    offset[0][id] = 0;
    for (size_t i=1 ; i<n_elem ; i++) {
      offset[i][id] = offset[i-1][id] + size[i-1][id];
    }
  }
};

/**
 * @brief Swap two values on host or device.
 * @tparam T Type of the variables to swap
 * @param a First variable
 * @param b Second variable
 */
template<typename T>
ONIKA_HOST_DEVICE_FUNC
inline void gpu_swap(T& a, T& b) {
  // Temporary storage for the swap
  T tmp = a;
  a = b;
  b = tmp;
}
}  // namespace exaDEM

namespace onika {
namespace parallel {
template<>
struct ParallelForFunctorTraits<exaDEM::PrefixSumInteractionTypePerCellCounter> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
