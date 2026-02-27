#pragma once

namespace exaDEM {
struct NBHManager {
  CellInteractionInformation info_cell;
  NbhCellStorage info_pair_cell;
};

inline void classify_interaction_grid(
    Classifier& classifier,
    Traversal& traversal,
    NBHManager& nbh_manager,
    GridCellParticleInteraction& ges) {
  InteractionWrapperStorage wrappers(classifier);
  InteractionWrapperAccessor wrapper_accessor = wrappers.accessor();
  auto [cell_ptr, cell_size] = traversal.info();

  constexpr bool do_ghost_only = false;
  constexpr bool do_active_interaction_only = false;//true;

  transfer_classifier_grid<do_ghost_only, do_active_interaction_only>(
      cell_ptr, nbh_manager.info_cell, nbh_manager.info_pair_cell,
      wrapper_accessor, ges);

}
}  // namespace exaDEM
