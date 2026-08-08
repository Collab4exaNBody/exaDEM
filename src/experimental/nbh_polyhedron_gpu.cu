/*
         Licensed to the Apache Software Foundation (ASF) under one
         or more contributor license agreements.  See the NOTICE file
         distributed with this work for additional information
         regarding copyright ownership.  The ASF licenses this file
         to you under the Apache License, Version 2.0 (the
         "License"); you may not use this file except in compliance
         with the License.  You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
*/

#ifdef ONIKA_CUDA_VERSION
#define DEBUG_NBH_GPU 1

#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <cassert>
#include <cub/cub.cuh>
#include <exaDEM/experimental/polyhedron/nbh_gpu/interaction_list_layout.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_cell_data.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_driver.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_pccp.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_interaction_history.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_utils.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/interaction_enum.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/polyhedron/nbh_polyhedron_driver.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/traversal.hpp>

namespace exaDEM {

// temporary storage for GPU computations. Avoid allocating and deallocating memory on GPU every time the operator is
// called. It will be removed in the future and replaced by a more generic scratch space for GPU computations
struct DataNeighborGPUScratch {
  ParticlePairStorage pp_storage_;
  onika::memory::CudaMMVector<int> pp_counts_;   // only required for PCCP
  onika::memory::CudaMMVector<int> pp_offsets_;  // only required for PCCP
  onika::memory::CudaMMVector<InteractionTypePerCellCounter> interaction_counts_;
  onika::memory::CudaMMVector<InteractionTypePerCellCounter> interaction_prefix_;
  onika::memory::CudaMMVector<int> type_counts_[InteractionTypeId::NTypesPP];  // only required for PCCP
  onika::memory::CudaMMVector<int> type_prefix_[InteractionTypeId::NTypesPP];  // only required for PCCP
  onika::memory::CudaMMVector<int>
      cp_type_counts_[InteractionTypeId::NTypesPP];  // reconstruct_cell_pair_offsets scratch
  onika::memory::CudaMMVector<int>
      cp_type_prefix_[InteractionTypeId::NTypesPP];  // reconstruct_cell_pair_offsets scratch
  InteractionHistory history_;
};

// Pilot constants for GPU kernels.
// These values are not expected to change
constexpr int kNeighborOffsetCount = 27;
constexpr int kNeighborGridSize = 3;
// but these one can be tuned for performance.
constexpr int kParticlePairBlockX = 8;
constexpr int kParticlePairBlockY = 8;

// helper functions
template <typename T>
void reset(onika::memory::CudaMMVector<T>& vec, onikaStream_t st = 0) {
  if (vec.size() > 0) {
    ONIKA_CU_MEMSET(vec.data(), 0, vec.size() * sizeof(T), st);
  }
}

inline IJK convert_offset_ijk(int offset) {
  assert(offset < kNeighborOffsetCount);
  IJK res;
  res.i = offset % kNeighborGridSize - 1;
  res.j = (offset / kNeighborGridSize) % kNeighborGridSize - 1;
  res.k = offset / (kNeighborGridSize * kNeighborGridSize) - 1;
  return res;
}
//! helper functions

/* Build the host-side metadata describing cell-to-cell neighbor pairs and ghost status.
   The grid provides access to the particle field data used to resolve neighbor cells. */
template <typename GridT>
inline void build_cell_neighbor_metadata(const GridT& grid, const IJK& dims, const size_t* cell_ptr, size_t cell_size,
                                         NbhCellHostStorage& host_storage, CellInteractionInformation& info_cell) {
  const auto& cells = grid.cells();

  info_cell.resize(cell_size);
  std::memset(info_cell.update_ghost_.data(), 0, cell_size * sizeof(uint8_t));

  size_t shift = 0;
  for (size_t i = 0; i < cell_size; ++i) {
    info_cell.start_cell_[i] = shift;
    size_t pair_count = 0;
    const size_t cell_a = cell_ptr[i];
    const IJK loc_a = grid_index_to_ijk(dims, cell_a);

    for (int offset = 0; offset < kNeighborOffsetCount; ++offset) {
      const size_t cell_b = grid_ijk_to_index(dims, loc_a + convert_offset_ijk(offset));
      if (cells[cell_b].size() > 0) {
        host_storage.owner_cell_.push_back(cell_a);
        host_storage.partner_cell_.push_back(cell_b);
        host_storage.ghost_.push_back(grid.is_ghost_cell(cell_b) ? InteractionPair::OwnerGhost
                                                                 : InteractionPair::NotGhost);
        if (grid.is_ghost_cell(cell_b)) {
          info_cell.update_ghost_[i] = 1;
        }
        ++pair_count;
      }
    }

    info_cell.number_of_pair_cells_[i] = pair_count;
    shift += pair_count;
  }
}

/* Initialize the temporary buffers used to store particle-pair counts and offsets. */
template <typename ScratchT>
inline void initialize_particle_pair_scratch(ScratchT& scratch, size_t cell_pair_size, onikaStream_t st = 0) {
  auto& pp_counts = scratch.pp_counts_;
  auto& pp_offsets = scratch.pp_offsets_;
  pp_offsets.resize(cell_pair_size);
  reset(pp_offsets, st);
  pp_counts.resize(cell_pair_size);
  reset(pp_counts, st);
}

/* Initialize the temporary buffers used to store interaction counts and prefix offsets. */
template <typename ScratchT>
inline void initialize_interaction_scratch(ScratchT& scratch, size_t total_pp, onikaStream_t st = 0) {
  auto& interaction_counts = scratch.interaction_counts_;
  auto& interaction_prefix = scratch.interaction_prefix_;
  interaction_counts.resize(total_pp);
  reset(interaction_counts, st);
  interaction_prefix.resize(total_pp);
  reset(interaction_prefix, st);
}

/* Add persistent driver interactions that were not found in the current classifier contents.
   wrapper_storage backs interaction_classifier_accessor's spans and must be owned by the caller:
   reassigning it here (rather than to a function-local InteractionWrapperStorage) keeps those
   spans valid after this function returns, for whatever the caller does next with the accessor. */
template <typename ContainerT, typename WrapperAccessorT, typename CellStorageAccessorT>
inline void add_unmatched_persistent_interactions(const InteractionHistory& history, size_t cell_size,
                                                  ContainerT& container,
                                                  WrapperAccessorT& interaction_classifier_accessor,
                                                  CellStorageAccessorT& cell_storage_accessor,
                                                  InteractionWrapperStorage& wrapper_storage) {
  std::vector<PlaceholderInteraction> unmatched_persistent;
  for (size_t ci = 0; ci < cell_size; ++ci) {
    size_t hist_begin = history.start_[ci];
    size_t hist_end = hist_begin + history.size_[ci];
    for (size_t h = hist_begin; h < hist_end; ++h) {
      PlaceholderInteraction interaction = history.data_[h];
      const auto type = interaction.type();
      if (type < get_first_id<InteractionType::ParticleDriver>() ||
          type > get_last_id<InteractionType::ParticleDriver>())
        continue;
      if (!interaction.persistent()) continue;

      auto& wrapper =
          interaction_classifier_accessor.template get_typed_accessor<InteractionType::ParticleDriver>(type);
      const int drv_offset = cell_storage_accessor.offset_[ci][type];
      const int drv_size = cell_storage_accessor.size_[ci][type];
      bool found = false;
      for (int k = drv_offset; k < drv_offset + drv_size; ++k) {
        if (wrapper.same(k, interaction)) {
          found = true;
          break;
        }
      }
      if (!found) {
        unmatched_persistent.push_back(interaction);
      }
    }
  }

  if (unmatched_persistent.empty()) {
    return;
  }

  lout << "[PERSISTENT] adding " << unmatched_persistent.size() << " unmatched persistent driver interactions"
       << std::endl;
  for (auto& interaction : unmatched_persistent) {
    const auto type = interaction.type();
    auto& c = container.template get_data<InteractionType::ParticleDriver>(type);
    const size_t old_size = c.size();
    container.resize(type, old_size + 1);
    InteractionWrapperStorage wrappers_tmp(container);
    InteractionWrapperAccessor tmp_accessor = wrappers_tmp.accessor();
    auto& w = tmp_accessor.get_typed_accessor<InteractionType::ParticleDriver>(type);
    w.set(old_size, interaction);
  }

  wrapper_storage = InteractionWrapperStorage(container);
  interaction_classifier_accessor = wrapper_storage.accessor();
}

template <typename GridT, class = AssertGridHasFields<GridT>>
class UpdateClassifierPolyhedronGPUPCCP : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(exanb::GridChunkNeighbors, chunk_neighbors, INPUT, OPTIONAL, DocString{"Neighbor list"});
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, DocString{"Interaction list"});
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
  ADD_SLOT(double, rcut_inc, INPUT, REQUIRED,
           DocString{"value added to the search distance to update neighbor list less frequently. in physical space"});
  ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(InteractionListBuildLayout, interaction_list_layout, INPUT_OUTPUT,
           DocString{"Data about packed interactions within classifier."});
  ADD_SLOT(DataNeighborGPUScratch, scratch, PRIVATE, DocString{"Scratch space for GPU computations"});
  ADD_SLOT(PersistentInteractionScratch, persistent_interaction_scratch, PRIVATE,
           DocString{"Scratch space for host-side persistent interaction bookkeeping"});
  ADD_SLOT(bool, enable_persistent_interactions, INPUT, false, DocString{"Enable persistent interactions"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
				This function builds the list of interactions per particle (polyhedron). Block-per-particle-pair (PCCP) version.

				YAML example [no option]:

					- nbh_polyhedron_gpu
			 )EOF";
  }

  inline void execute() final {
    using namespace onika::parallel;
    using onika::cuda::make_const_span;
    using onika::cuda::make_span;
    using onika::cuda::span;
    using onika::parallel::flush;
    using onika::parallel::set_lane;
#ifndef ONIKA_CUDA_VERSION
    color_log::error("nbh_polyhedron_gpu",
                     "This operator only work on GPU.\n"
                     "                     Please use nbh_polyhedron.");
#else
    auto& grid_data = *grid;
    const auto grid_cells = grid_data.cells();
    const IJK grid_dimensions = grid_data.dimension();
    shapes& shapes_data = *shapes_collection;
    auto [cell_indices, active_cell_count] = traversal_real->info();
    auto* vertex_field_data = cvf->data();
    auto& interaction_container = *ic;
    const DriversGPUAccessor driver_accessor = *drivers;

    auto get_exec_ctx = [this](const char* sub_tag = nullptr) { return this->parallel_execution_context(sub_tag); };

    constexpr int kLaneParticleDriver = 0;
    constexpr int kLaneParticleParticle = 1;
    constexpr int kLaneHistory = 2;
    constexpr int kLaneInnerBond = 3;
    auto& parallel_queue = parallel_execution_queue();
    onikaStream_t particle_stream = global_cuda_ctx()->getThreadStream(kLaneParticleParticle);
    onikaStream_t history_stream = global_cuda_ctx()->getThreadStream(kLaneHistory);
    onikaStream_t driver_stream = global_cuda_ctx()->getThreadStream(kLaneParticleDriver);

    onikaStream_t st_updateghost;
    ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_updateghost);

    NbhCellHostStorage cell_neighbor_host_storage;
    CellInteractionInformation& cell_interaction_info = interaction_list_layout->cell_interaction_info_;

    IgnorePairsGPU& ignore_pairs_gpu = persistent_interaction_scratch->ignore_pairs_gpu_;
    if (*enable_persistent_interactions) {
      ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::build_ignore_pairs");
      // Persistent interactions are already accounted for elsewhere; skip them here so the neighbor search doesn't
      // rediscover them as plain contacts.
      // Must run before setup_history_clean_ges(), which resets *ges.
      ListOfIgnorePairs& ignore_pairs = persistent_interaction_scratch->ignore_pairs_;
      build_list_of_ignore_pair(ignore_pairs, cell_indices, active_cell_count, *ges);
      ignore_pairs_gpu.build(ignore_pairs, grid_data.number_of_cells());
      ONIKA_CU_PROF_RANGE_POP();
    }

    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::build_cell_neighbor_metadata");
    build_cell_neighbor_metadata(grid_data, grid_dimensions, cell_indices, active_cell_count,
                                 cell_neighbor_host_storage, cell_interaction_info);
    cell_interaction_info.prefetch_cpu(st_updateghost);
    CellStorage& cell_storage = interaction_list_layout->cell_storage_;
    cell_storage.resize(active_cell_count, parallel_queue, kLaneInnerBond, get_exec_ctx);
    auto cell_storage_accessor = cell_storage.view();

    CellPairStorage& cell_pair_storage = interaction_list_layout->cell_pair_storage_;
    cell_pair_storage.reset(cell_neighbor_host_storage, parallel_queue, kLaneParticleParticle, get_exec_ctx);

    auto cell_pair_accessor = cell_pair_storage.view();
    ONIKA_CU_PROF_RANGE_POP();

    ParallelForOptions opts;
    opts.omp_scheduling = OMP_SCHED_GUIDED;
    // BlockParallelForOptions bopts;

    CountDriverInteractionsFunc driver_counter = {grid_cells,         cell_storage_accessor, cell_indices,   *rcut_inc,
                                                  shapes_data.data(), vertex_field_data,     driver_accessor};

    const auto neighbor_cell_pair_count = cell_neighbor_host_storage.owner_cell_.size();
    ONIKA_CU_DEVICE_SYNCHRONIZE();

    PersistentInnerBonds& persistent_inner_bonds = persistent_interaction_scratch->persistent_inner_bonds_;
    if (*enable_persistent_interactions) {
      ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::collect_persistent_inner_bonds");
      collect_persistent_inner_bonds(persistent_inner_bonds, cell_storage, cell_indices, active_cell_count, *ges);
      ONIKA_CU_PROF_RANGE_POP();
    }

    // Used in Build particle pairs (PCCP)
    // Place here to avoid several synchronization calls in the middle of the operator.
    auto& particle_pair_counts = scratch->pp_counts_;
    auto& particle_pair_offsets = scratch->pp_offsets_;
    initialize_particle_pair_scratch(*scratch, neighbor_cell_pair_count, particle_stream);
    // end scratch variables

    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::count_driver_interactions");
    PrefixSumInteractionTypePerCellCounter cell_storage_prefix_sum{cell_storage_accessor.offset_,
                                                                   cell_storage_accessor.size_, active_cell_count};

    ParallelExecutionSpace<1> prefix_sum_type_range = {{get_first_id<InteractionType::ParticleDriver>()},
                                                       {get_last_id<InteractionType::InnerBond>() + 1}};

    parallel_queue << set_lane(kLaneParticleDriver)
                   << parallel_for(active_cell_count, driver_counter,
                                   parallel_execution_context("nbh_gpu::counter_driver"), opts)
                   << parallel_for(prefix_sum_type_range, cell_storage_prefix_sum,
                                   parallel_execution_context("nbh_gpu::cell_storage_prefix_sum"), opts)
                   << flush;
    ONIKA_CU_PROF_RANGE_POP();

    InteractionHistory& history = scratch->history_;

    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::setup_history_clean_ges");
    setup_history_clean_ges(grid_cells, cell_indices, active_cell_count, *ges, history, history_stream);

    ONIKA_CU_DEVICE_SYNCHRONIZE();
    ONIKA_CU_PROF_RANGE_POP();

    // ****** Build particle pairs (PCCP) ******* //
    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::build_particle_pairs");
    dim3 pp_block(kParticlePairBlockX, kParticlePairBlockY, 1);

    CountParticlePairsKernel<kParticlePairBlockX, kParticlePairBlockY, true>
        <<<neighbor_cell_pair_count, pp_block, 0, particle_stream>>>(
            grid_cells, cell_pair_accessor.owner_cell_.data(), cell_pair_accessor.partner_cell_.data(),
            cell_pair_accessor.ghost_.data(), *rcut_inc, shapes_data.data(), vertex_field_data,
            particle_pair_counts.data(), neighbor_cell_pair_count, ignore_pairs_gpu.view());
    ONIKA_CU_STREAM_SYNCHRONIZE(particle_stream);

    exclusive_scan_device(particle_pair_counts.data(), particle_pair_offsets.data(), neighbor_cell_pair_count,
                          particle_stream);
    ONIKA_CU_STREAM_SYNCHRONIZE(particle_stream);
    ONIKA_CU_PROF_RANGE_POP();

    size_t total_pp = 0;
    if (neighbor_cell_pair_count > 0) {
      total_pp =
          particle_pair_counts[neighbor_cell_pair_count - 1] + particle_pair_offsets[neighbor_cell_pair_count - 1];
    }

    auto& particle_pair_storage = scratch->pp_storage_;
    particle_pair_storage.resize(total_pp);

    // Used in Count interactions per particle pair (PCCP)
    // Place here to avoid several synchronization calls in the middle of the operator.
    auto& interaction_counts_per_pair = scratch->interaction_counts_;
    auto& interaction_prefix_per_pair = scratch->interaction_prefix_;
    initialize_interaction_scratch(*scratch, total_pp, particle_stream);
    // end scratch variables

    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::fill_particle_pairs");
    if (total_pp > 0) {
      FillParticlePairsKernel<kParticlePairBlockX, kParticlePairBlockY, true>
          <<<neighbor_cell_pair_count, pp_block, 0, particle_stream>>>(
              grid_cells, cell_pair_accessor.owner_cell_.data(), cell_pair_accessor.partner_cell_.data(),
              cell_pair_accessor.ghost_.data(), *rcut_inc, shapes_data.data(), vertex_field_data,
              particle_pair_offsets.data(), particle_pair_storage.cell_i_.data(), particle_pair_storage.cell_j_.data(),
              particle_pair_storage.p_i_.data(), particle_pair_storage.p_j_.data(), particle_pair_storage.ghost_.data(),
              particle_pair_storage.cell_pair_idx_.data(), neighbor_cell_pair_count, ignore_pairs_gpu.view());
      ONIKA_CU_STREAM_SYNCHRONIZE(particle_stream);
    }
    ONIKA_CU_PROF_RANGE_POP();

    // ****** Count interactions per particle pair (PCCP) ******* //
    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::count_interactions_per_pair");

    InteractionTypePerCellCounter total_interactions_per_type;
    for (int typeID = 0; typeID < InteractionTypeId::NTypes; typeID++) {
      total_interactions_per_type[typeID] = 0;
    }

    if (total_pp > 0) {
      CountInteractionsPPKernel<kParticlePairBlockX, kParticlePairBlockY><<<total_pp, pp_block, 0, particle_stream>>>(
          grid_cells, vertex_field_data, shapes_data.data(), *rcut_inc, particle_pair_storage.cell_i_.data(),
          particle_pair_storage.cell_j_.data(), particle_pair_storage.p_i_.data(), particle_pair_storage.p_j_.data(),
          interaction_counts_per_pair.data(), total_pp);
      ONIKA_CU_STREAM_SYNCHRONIZE(particle_stream);

      // GPU prefix sum per interaction type
      auto& interaction_type_counts = scratch->type_counts_;
      auto& interaction_type_prefix = scratch->type_prefix_;
      for (int typeID = 0; typeID < InteractionTypeId::NTypesPP; typeID++) {
        interaction_type_counts[typeID].resize(total_pp);
        reset(interaction_type_counts[typeID], particle_stream);
        interaction_type_prefix[typeID].resize(total_pp);
        reset(interaction_type_prefix[typeID], particle_stream);
      }

      ExtractInteractionCountsFunc extract_counts{
          make_const_span(interaction_counts_per_pair), make_span(interaction_type_counts[0]),
          make_span(interaction_type_counts[1]), make_span(interaction_type_counts[2]),
          make_span(interaction_type_counts[3])};
      parallel_queue << set_lane(kLaneParticleParticle)
                     << parallel_for(total_pp, extract_counts,
                                     parallel_execution_context("nbh_gpu::extract_interaction_counts"), opts)
                     << flush;

      // iterate over particle-particle interaction types and compute prefix sum for each type
      for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
        exclusive_scan_device(interaction_type_counts[t].data(), interaction_type_prefix[t].data(), total_pp,
                              particle_stream);
      }

      PackInteractionPrefixFunc pack_prefix{make_span(interaction_prefix_per_pair),
                                            make_const_span(interaction_type_prefix[0]),
                                            make_const_span(interaction_type_prefix[1]),
                                            make_const_span(interaction_type_prefix[2]),
                                            make_const_span(interaction_type_prefix[3])};
      parallel_queue << set_lane(kLaneParticleParticle)
                     << parallel_for(total_pp, pack_prefix,
                                     parallel_execution_context("nbh_gpu::pack_interaction_prefix"), opts)
                     << flush;
      parallel_queue.wait(kLaneParticleParticle);

      // compute total interactions per type
      for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
        total_interactions_per_type[t] =
            interaction_type_prefix[t][total_pp - 1] + interaction_type_counts[t][total_pp - 1];
      }
    }
    ONIKA_CU_PROF_RANGE_POP();

    // ****** Resize Classifier for PP ******* //
    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::resize_classifier");
    InteractionParticleAccessor particle_particle_classifier_accessor;
    for (int typeID = get_first_id<InteractionType::ParticleParticle>();
         typeID <= get_last_id<InteractionType::ParticleParticle>(); typeID++) {
      auto& c = interaction_container.get_data<ParticleParticle>(typeID);
      c.resize(total_interactions_per_type[typeID]);
      particle_particle_classifier_accessor[typeID] = InteractionWrapper(c);
    }

    // ****** Resize Classifier for Driver ******* //
    parallel_queue.wait(kLaneParticleDriver);

    for (int typeID = get_first_id<InteractionType::ParticleDriver>();
         typeID <= get_last_id<InteractionType::ParticleDriver>(); typeID++) {
      size_t newsize = cell_storage.offset_.back()[typeID] + cell_storage.size_.back()[typeID];
      interaction_container.resize(typeID, newsize);
    }

    // ****** Resize Classifier for InnerBond ******* //
    for (int typeID = get_first_id<InteractionType::InnerBond>(); typeID <= get_last_id<InteractionType::InnerBond>();
         typeID++) {
      size_t newsize = cell_storage.offset_.back()[typeID] + cell_storage.size_.back()[typeID];
      interaction_container.resize(typeID, newsize);
    }

    InteractionWrapperStorage wrappers(interaction_container);
    InteractionWrapperAccessor interaction_classifier_accessor = wrappers.accessor();
    ONIKA_CU_PROF_RANGE_POP();

    if (*enable_persistent_interactions) {
      ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::fill_classifier_persistent_inner_bonds");
      fill_classifier_persistent_inner_bonds(persistent_inner_bonds, cell_storage_accessor,
                                             interaction_classifier_accessor);
      ONIKA_CU_PROF_RANGE_POP();
    }

    // ****** Fill Classifier PP (PCCP) ******* //
    ONIKA_CU_DEVICE_SYNCHRONIZE();

    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::fill_classifier_pp");
    if (total_pp > 0) {
      FillInteractionsPPKernel<kParticlePairBlockX, kParticlePairBlockY><<<total_pp, pp_block, 0, particle_stream>>>(
          grid_cells, vertex_field_data, shapes_data.data(), *rcut_inc, particle_pair_storage.cell_i_.data(),
          particle_pair_storage.cell_j_.data(), particle_pair_storage.p_i_.data(), particle_pair_storage.p_j_.data(),
          particle_pair_storage.ghost_.data(), interaction_prefix_per_pair.data(),
          particle_particle_classifier_accessor, total_pp);

      reconstruct_cell_pair_offsets(particle_pair_storage, interaction_counts_per_pair.data(), total_pp,
                                    neighbor_cell_pair_count, cell_pair_storage, scratch->cp_type_counts_,
                                    scratch->cp_type_prefix_, parallel_queue, kLaneParticleParticle, particle_stream,
                                    get_exec_ctx, opts);
      parallel_queue.wait(kLaneParticleParticle);
    }
    ONIKA_CU_PROF_RANGE_POP();

    // Fold PP totals (cell_pair_storage) into cell_storage's per-cell table.
    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::fold_pp_totals_and_classify_driver");
    add_particle_particle_totals(cell_storage, cell_interaction_info, cell_pair_storage, parallel_queue,
                                 kLaneParticleDriver, get_exec_ctx, opts);

    ClassifyDriverInteractionsFunc driver_classifier = {
        grid_cells,         cell_storage_accessor, cell_indices,    *rcut_inc,
        shapes_data.data(), vertex_field_data,     driver_accessor, interaction_classifier_accessor};
    parallel_queue << set_lane(kLaneParticleDriver)
                   << parallel_for(active_cell_count, driver_classifier,
                                   parallel_execution_context("nbh_gpu::classify_driver"), opts)
                   << flush;
    parallel_queue.wait(kLaneParticleDriver);
    ONIKA_CU_PROF_RANGE_POP();

    history.prefetch_gpu(history_stream);

    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::update_history");
    UpdateHistoryFunc history_updater = {history.start_.data(), history.size_.data(), history.data_.data(),
                                         cell_storage_accessor, interaction_classifier_accessor};

    ONIKA_CU_STREAM_SYNCHRONIZE(history_stream);
    ONIKA_CU_STREAM_SYNCHRONIZE(driver_stream);
    ONIKA_CU_STREAM_SYNCHRONIZE(particle_stream);

    parallel_queue << set_lane(kLaneHistory)
                   << parallel_for(history.start_.size(), history_updater, parallel_execution_context(), opts)
                   << flush;
    parallel_queue.wait(kLaneHistory);
    ONIKA_CU_PROF_RANGE_POP();

    // === ADD PERSISTENT INTERACTIONS ===
    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::add_unmatched_persistent_interactions");
    add_unmatched_persistent_interactions(history, active_cell_count, interaction_container,
                                          interaction_classifier_accessor, cell_storage_accessor, wrappers);
    ONIKA_CU_PROF_RANGE_POP();

    ONIKA_CU_PROF_RANGE_PUSH("nbh_gpu::transfer_classifier_grid");
    constexpr bool do_ghost_only = true;
    constexpr bool do_active_interaction_only = false;
    transfer_classifier_grid<do_ghost_only, do_active_interaction_only, false>(
        cell_indices, cell_interaction_info, cell_storage, interaction_classifier_accessor, *ges,
        get_first_id<InteractionType::ParticleParticle>(), get_last_id<InteractionType::ParticleParticle>());

    transfer_classifier_grid<do_ghost_only, do_active_interaction_only, true>(
        cell_indices, cell_interaction_info, cell_storage, interaction_classifier_accessor, *ges,
        get_first_id<InteractionType::ParticleDriver>(), get_last_id<InteractionType::ParticleDriver>());

    // Not required for InnerBond interactions, since they are persistent and already accounted for in the classifier.

    ONIKA_CU_PROF_RANGE_POP();
#endif
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(nbh_polyhedron_gpu) {
  OperatorNodeFactory::instance()->register_factory("nbh_polyhedron_gpu",
                                                    make_grid_variant_operator<UpdateClassifierPolyhedronGPUPCCP>);
}
}  // namespace exaDEM
#endif
