#pragma once

#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>

namespace exaDEM {
struct ApplyInterfaceFractureCriterionFunc {
  Interface* const interface;
  uint8_t* const break_interface;
  InteractionWrapper<InteractionType::InnerBond> interaction;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t i) const {
    auto& [offset, size] = interface[i];
    double En = 0.0;  // reset
    double Et = 0.0;
    double criterion = interaction.Criterion(offset);
    for (size_t j = offset; j < offset + size; j++) {
      En += interaction.En(j);
      Et += interaction.Et(j);
    }

    if ((En + Et) > criterion) {  //(2 * area * g))
      break_interface[i] = true;
    }
  }
};
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <>
struct ParallelForFunctorTraits<exaDEM::ApplyInterfaceFractureCriterionFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = true;
  static inline constexpr bool CudaCompatible = false;  // true;
};
}  // namespace parallel
}  // namespace onika
