#pragma once

#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_cell_data.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_driver.hpp>
#include <exaDEM/traversal.hpp>

namespace exaDEM {
struct InteractionListBuildLayout {
  CellInteractionInformation cell_interaction_info_;
  CellPairStorage cell_pair_storage_;
  CellStorage cell_storage_;  // per-cell counts/offsets, all interaction types (PP + driver)
};

/**
 * @brief Copies the active interactions from the Classifier back into the grid interaction storage.
 *
 * Despite its name, this is the inverse of a classification: the interactions are read from the
 * Classifier (grouped by type) and written into the per-cell storage `ges`, following the layout
 * described by `interaction_list_layout`. Only active interactions are transferred (ghost-only
 * filtering is disabled).
 *
 * @param classifier Source: interactions classified by type.
 * @param traversal Non-empty cells of the current grid.
 * @param interaction_list_layout Per-cell counts and offsets of the packed interactions.
 * @param ges Destination: grid cell interaction storage.
 */
inline void classify_interaction_grid(Classifier& classifier, Traversal& traversal, InteractionListBuildLayout& interaction_list_layout,
                                      GridCellParticleInteraction& ges) {
  ClassifierViewStorage wrappers(classifier);
  ClassifierViewAccessor interaction_classifier_accessor = wrappers.accessor();
  auto [cell_ptr, cell_size] = traversal.info();

  constexpr bool do_ghost_only = false;
  constexpr bool do_active_interaction_only = true;

  transfer_classifier_grid<do_ghost_only, do_active_interaction_only>(
      cell_ptr, interaction_list_layout.cell_interaction_info_, interaction_list_layout.cell_storage_, interaction_classifier_accessor, ges);
}
}  // namespace exaDEM
