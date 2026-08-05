#pragma once

#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_driver.hpp>
#include <exaDEM/traversal.hpp>

namespace exaDEM {
struct NBHManager {
  CellInteractionInformation info_cell_;
  CellPairStorage info_pair_cell_;
  CellStorage info_cell_storage_;  // per-cell counts/offsets, all interaction types (PP + driver)
};

inline void classify_interaction_grid(Classifier& classifier, Traversal& traversal, NBHManager& nbh_manager,
                                      GridCellParticleInteraction& ges) {
  InteractionWrapperStorage wrappers(classifier);
  InteractionWrapperAccessor wrapper_accessor = wrappers.accessor();
  auto [cell_ptr, cell_size] = traversal.info();

  constexpr bool do_ghost_only = false;
  constexpr bool do_active_interaction_only = true;

  transfer_classifier_grid<do_ghost_only, do_active_interaction_only>(
      cell_ptr, nbh_manager.info_cell_, nbh_manager.info_cell_storage_, wrapper_accessor, ges);
}
}  // namespace exaDEM
