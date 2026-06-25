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
  exanb::Vec3d* const fn;  // list of normal forces for each interaction that compose the interfaces
  double* const dn;        // list of normal displacements for each interaction that compose the interfaces

  /** @brief Apply the fracture criterion to the i-th interface
   * @param i The index of the interface to apply the fracture criterion to.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(size_t i) const {
    auto& [offset, size] = interface[i];
    double En = 0.0;
    double Et = 0.0;
    double S = 0.0;

    // Sum of the normal and tangential energy of the interactions that compose the interface
    for (size_t j = offset; j < offset + size; j++) {
      En += interaction.En(j);
      Et += interaction.Et(j);
      if (dn[j] > 0.0) {
        S += norm(fn[j]);
      }
    }
    S = S / size;  // average of the normal forces of the interactions that compose the interface

    // Criterion is stored in the interaction that compose the interface.
    // We can take the criterion of the first interaction
    // since all interactions that compose the interface have the same criterion.

    // Criterion formula: En or Et > 2 * area * g_{n or t}
    // g is defined by the input parameters.
    RuptureCriteria& criterion = interaction.criteria(offset);
    RuptureMode mode = criterion.mode;
    if (mode == RuptureMode::EnergyMixedMode) {
      double threshold = criterion.energy_criterion();
      if (En + Et > threshold) {  // cs = sum
        break_interface[i] = true;
      }
    } else if (mode == RuptureMode::EnergySeparateMode) {
      double cn = criterion.energy_normal_criterion();
      double ct = criterion.energy_tangential_criterion();
      if (En > cn) {
        break_interface[i] = true;
      } else if (Et > ct) {
        break_interface[i] = true;
      }
    } else if (mode == RuptureMode::StressEnergySeparateMode) {
      double energy_threshold = criterion.energy_criterion();
      double stress_threshold = criterion.stress_criterion();
      if ((En > energy_threshold) && (S > stress_threshold)) {
        break_interface[i] = true;
      }
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
