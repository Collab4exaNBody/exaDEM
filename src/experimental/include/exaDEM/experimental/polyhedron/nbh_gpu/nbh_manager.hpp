#pragma once

#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_driver.hpp>
#include <exaDEM/traversal.hpp>

namespace exaDEM {
struct NBHManager {
  CellInteractionInformation cell_interaction_info_;
  CellPairStorage cell_pair_storage_;
  CellStorage cell_storage_;  // per-cell counts/offsets, all interaction types (PP + driver)
};

inline void classify_interaction_grid(Classifier& classifier, Traversal& traversal, NBHManager& nbh_manager,
                                      GridCellParticleInteraction& ges) {
  InteractionWrapperStorage wrappers(classifier);
  InteractionWrapperAccessor interaction_classifier_accessor = wrappers.accessor();
  auto [cell_ptr, cell_size] = traversal.info();

  constexpr bool do_ghost_only = false;
  constexpr bool do_active_interaction_only = true;

  transfer_classifier_grid<do_ghost_only, do_active_interaction_only>(
      cell_ptr, nbh_manager.cell_interaction_info_, nbh_manager.cell_storage_, interaction_classifier_accessor, ges);
}
}  // namespace exaDEM
