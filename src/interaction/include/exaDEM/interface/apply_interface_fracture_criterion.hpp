#pragma once

#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_for.h>

#include <exaDEM/interface/interface.hpp>

namespace exaDEM {
/** @brief Function to apply the fracture criterion to each interface */
struct ApplyInterfaceFractureCriterionFunc {
  Interface* const interface;      // list of interfaces
  uint8_t* const break_interface;  // list of booleans that indicate if the interface is broken or not. 1 if the
                                   // interface is broken, 0 otherwise.
  InteractionWrapper<InteractionType::InnerBond> interaction;  // interactions that compose the interfaces

  /** @brief Apply the fracture criterion to the i-th interface
   * @param i The index of the interface to apply the fracture criterion to.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t i) const {
    auto& [offset, size] = interface[i];
    double En = 0.0;
    double Et = 0.0;
    // Criterion is stored in the interaction that compose the interface.
    // We can take the criterion of the first interaction
    // since all interactions that compose the interface have the same criterion.
    double criterion = interaction.Criterion(offset);
    // Sum of the normal and tangential energy of the interactions that compose the interface
    for (size_t j = offset; j < offset + size; j++) {
      En += interaction.En(j);
      Et += interaction.Et(j);
    }

    // Criterion formula: En + Et > 2 * area * g
    // g is defined by the input parameters.
    if ((En + Et) > criterion) {
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
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
