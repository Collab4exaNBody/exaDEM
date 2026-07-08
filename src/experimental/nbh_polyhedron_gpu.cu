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
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_cell_data.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_driver.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_pccp.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_interaction_history.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_manager.hpp>
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
  onika::memory::CudaMMVector<int> pp_counts_;
  onika::memory::CudaMMVector<int> pp_offsets_;
  onika::memory::CudaMMVector<InteractionTypePerCellCounter> interaction_counts_;
  onika::memory::CudaMMVector<InteractionTypePerCellCounter> interaction_prefix_;
  onika::memory::CudaMMVector<int> type_counts_[InteractionTypeId::NTypesPP];
  onika::memory::CudaMMVector<int> type_prefix_[InteractionTypeId::NTypesPP];
};

// Pilot constants for GPU kernels. These values are not expected to change, but they can be tuned for performance.
constexpr int kNeighborOffsetCount = 27;
constexpr int kNeighborGridSize = 3;
constexpr int kParticlePairBlockX = 8;
constexpr int kParticlePairBlockY = 8;
constexpr int kScanBlockSize = 256;

// helper functions
template <typename T>
void reset(onika::memory::CudaMMVector<T>& vec) {
  if (vec.size() > 0) {
    ONIKA_CU_MEMSET(vec.data(), 0, vec.size() * sizeof(T));
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

/* Run an exclusive prefix sum on device memory using CUB. */
template <typename T>
inline void exclusive_scan_device(const T* input, T* output, size_t count) {
  void* d_tmp = nullptr;
  size_t tmp_bytes = 0;
  cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, input, output, count);
  cudaMalloc(&d_tmp, tmp_bytes);
  cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, input, output, count);
  cudaFree(d_tmp);
}

/* Initialize the temporary buffers used to store particle-pair counts and offsets. */
template <typename ScratchT>
inline void initialize_particle_pair_scratch(ScratchT& scratch, size_t cell_pair_size) {
  auto& pp_counts = scratch.pp_counts_;
  auto& pp_offsets = scratch.pp_offsets_;
  pp_offsets.resize(cell_pair_size);
  reset(pp_offsets);
  pp_counts.resize(cell_pair_size);
  reset(pp_counts);
}

/* Initialize the temporary buffers used to store interaction counts and prefix offsets. */
template <typename ScratchT>
inline void initialize_interaction_scratch(ScratchT& scratch, size_t total_pp) {
  auto& interaction_counts = scratch.interaction_counts_;
  auto& interaction_prefix = scratch.interaction_prefix_;
  interaction_counts.resize(total_pp);
  reset(interaction_counts);
  interaction_prefix.resize(total_pp);
  reset(interaction_prefix);
}

/* Add persistent driver interactions that were not found in the current classifier contents. */
template <typename ContainerT, typename WrapperAccessorT, typename CellDriverAccessorT>
inline void add_unmatched_persistent_interactions(const InteractionHistory& history, size_t cell_size,
                                                  ContainerT& container,
                                                  WrapperAccessorT& classifier_interaction_accessor,
                                                  CellDriverAccessorT& cell_driver_accessor) {
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
          classifier_interaction_accessor.template get_typed_accessor<InteractionType::ParticleDriver>(type);
      const int drv_offset = cell_driver_accessor.offset_[ci][type];
      const int drv_size = cell_driver_accessor.size_[ci][type];
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

  InteractionWrapperStorage wrappers2(container);
  classifier_interaction_accessor = wrappers2.accessor();
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
  ADD_SLOT(NBHManager, nbh_manager, INPUT_OUTPUT, DocString{"Data about packed interactions within classifier."});
  ADD_SLOT(DataNeighborGPUScratch, scratch, PRIVATE, DocString{"Scratch space for GPU computations"});

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

    auto get_exec_ctx = [this]() { return this->parallel_execution_context(); };

    onikaStream_t st_updateghost, st_history, st_particle, st_driver, st_innerbond;
    ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_updateghost);
    ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_history);
    ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_particle);
    ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_driver);
    ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_innerbond);

    NbhCellHostStorage cell_neighbor_host_storage;
    CellInteractionInformation& cell_interaction_info = nbh_manager->info_cell_;

    build_cell_neighbor_metadata(grid_data, grid_dimensions, cell_indices, active_cell_count,
                                 cell_neighbor_host_storage, cell_interaction_info);
    cell_interaction_info.prefetch_cpu(st_updateghost);
    CellDriverStorage& cell_driver_storage = nbh_manager->info_cell_driver_;
    cell_driver_storage.resize(active_cell_count, get_exec_ctx);
    auto cell_driver_accessor = cell_driver_storage.accessor();

    NbhCellStorage& cell_pair_storage = nbh_manager->info_pair_cell_;
    cell_pair_storage.reset(cell_neighbor_host_storage, get_exec_ctx);

    NbhCellAccessor cell_pair_accessor(cell_pair_storage);

    ParallelForOptions opts;
    opts.omp_scheduling = OMP_SCHED_GUIDED;
    // BlockParallelForOptions bopts;

    CountIPDFunc driver_counter = {grid_cells,         cell_driver_accessor, cell_indices,   *rcut_inc,
                                   shapes_data.data(), vertex_field_data,    driver_accessor};

    const auto neighbor_cell_pair_count = cell_neighbor_host_storage.owner_cell_.size();
    ONIKA_CU_DEVICE_SYNCHRONIZE();

    // Used in Build particle pairs (PCCP)
    // Place here to avoid several synchronization calls in the middle of the operator.
    auto& particle_pair_counts = scratch->pp_counts_;
    auto& particle_pair_offsets = scratch->pp_offsets_;
    initialize_particle_pair_scratch(*scratch, neighbor_cell_pair_count);
    // end scratch variables

    parallel_for(active_cell_count, driver_counter, parallel_execution_context("nbh_gpu::counter_driver,"), opts);
    PrefixSumInteractionTypePerCellCounter driver_prefix_sum{cell_driver_accessor.offset_, cell_driver_accessor.size_,
                                                             active_cell_count};

    ParallelExecutionSpace<1> parallel_range_fpd = {{get_first_id<InteractionType::ParticleDriver>()},
                                                    {get_last_id<InteractionType::ParticleDriver>() + 1}};

    ONIKA_CU_DEVICE_SYNCHRONIZE();
    parallel_for(parallel_range_fpd, driver_prefix_sum, parallel_execution_context("nbh_gpu::func_prefix_driver"),
                 opts);

    InteractionHistory history;

    setup_history_clean_ges(grid_cells, cell_indices, active_cell_count, *ges, history, st_history);

    ONIKA_CU_DEVICE_SYNCHRONIZE();

    // ****** Build particle pairs (PCCP) ******* //
    dim3 pp_block(kParticlePairBlockX, kParticlePairBlockY, 1);

    CountParticlePairsKernel<kParticlePairBlockX, kParticlePairBlockY><<<neighbor_cell_pair_count, pp_block>>>(
        grid_cells, cell_pair_accessor.owner_cell_, cell_pair_accessor.partner_cell_, cell_pair_accessor.ghost_,
        *rcut_inc, shapes_data.data(), vertex_field_data, particle_pair_counts.data(), neighbor_cell_pair_count);
    ONIKA_CU_DEVICE_SYNCHRONIZE();

    exclusive_scan_device(particle_pair_counts.data(), particle_pair_offsets.data(), neighbor_cell_pair_count);
    ONIKA_CU_DEVICE_SYNCHRONIZE();

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
    initialize_interaction_scratch(*scratch, total_pp);
    // end scratch variables

    if (total_pp > 0) {
      FillParticlePairsKernel<kParticlePairBlockX, kParticlePairBlockY><<<neighbor_cell_pair_count, pp_block>>>(
          grid_cells, cell_pair_accessor.owner_cell_, cell_pair_accessor.partner_cell_, cell_pair_accessor.ghost_,
          *rcut_inc, shapes_data.data(), vertex_field_data, particle_pair_offsets.data(),
          particle_pair_storage.cell_i_.data(), particle_pair_storage.cell_j_.data(), particle_pair_storage.p_i_.data(),
          particle_pair_storage.p_j_.data(), particle_pair_storage.ghost_.data(),
          particle_pair_storage.cell_pair_idx_.data(), neighbor_cell_pair_count);
      ONIKA_CU_DEVICE_SYNCHRONIZE();
    }

    // ****** Count interactions per particle pair (PCCP) ******* //

    InteractionTypePerCellCounter total_interactions_per_type;
    for (int typeID = 0; typeID < InteractionTypeId::NTypes; typeID++) {
      total_interactions_per_type[typeID] = 0;
    }

    if (total_pp > 0) {
      CountInteractionsPPKernel<kParticlePairBlockX, kParticlePairBlockY><<<total_pp, pp_block>>>(
          grid_cells, vertex_field_data, shapes_data.data(), *rcut_inc, particle_pair_storage.cell_i_.data(),
          particle_pair_storage.cell_j_.data(), particle_pair_storage.p_i_.data(), particle_pair_storage.p_j_.data(),
          interaction_counts_per_pair.data(), total_pp);
      ONIKA_CU_DEVICE_SYNCHRONIZE();

      // GPU prefix sum per interaction type
      auto& interaction_type_counts = scratch->type_counts_;
      auto& interaction_type_prefix = scratch->type_prefix_;
      for (int typeID = 0; typeID < InteractionTypeId::NTypesPP; typeID++) {
        interaction_type_counts[typeID].resize(total_pp);
        reset(interaction_type_counts[typeID]);
        interaction_type_prefix[typeID].resize(total_pp);
        reset(interaction_type_prefix[typeID]);
      }

      const int grid_1d = (total_pp + kScanBlockSize - 1) / kScanBlockSize;

      ONIKA_CU_DEVICE_SYNCHRONIZE();
      ExtractInteractionCounts<<<grid_1d, kScanBlockSize>>>(
          interaction_counts_per_pair.data(), interaction_type_counts[0].data(), interaction_type_counts[1].data(),
          interaction_type_counts[2].data(), interaction_type_counts[3].data(), total_pp);
      ONIKA_CU_DEVICE_SYNCHRONIZE();

      for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
        exclusive_scan_device(interaction_type_counts[t].data(), interaction_type_prefix[t].data(), total_pp);
      }
      ONIKA_CU_DEVICE_SYNCHRONIZE();

      PackInteractionPrefix<<<grid_1d, kScanBlockSize>>>(
          interaction_prefix_per_pair.data(), interaction_type_prefix[0].data(), interaction_type_prefix[1].data(),
          interaction_type_prefix[2].data(), interaction_type_prefix[3].data(), total_pp);
      ONIKA_CU_DEVICE_SYNCHRONIZE();

      for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
        total_interactions_per_type[t] =
            interaction_type_prefix[t][total_pp - 1] + interaction_type_counts[t][total_pp - 1];
      }
    }

    // ****** Resize Classifier for PP ******* //
    InteractionParticleAccessor particle_particle_classifier_accessor;
    for (int typeID = get_first_id<InteractionType::ParticleParticle>();
         typeID <= get_last_id<InteractionType::ParticleParticle>(); typeID++) {
      auto& c = interaction_container.get_data<ParticleParticle>(typeID);
      c.resize(total_interactions_per_type[typeID]);
      particle_particle_classifier_accessor[typeID] = InteractionWrapper(c);
    }

    // ****** Resize Classifier for Driver ******* //
    for (int typeID = get_first_id<InteractionType::ParticleDriver>();
         typeID <= get_last_id<InteractionType::ParticleDriver>(); typeID++) {
      size_t newsize = cell_driver_storage.offset_.back()[typeID] + cell_driver_storage.size_.back()[typeID];
      interaction_container.resize(typeID, newsize);
    }

    InteractionWrapperStorage wrappers(interaction_container);
    InteractionWrapperAccessor interaction_classifier_accessor = wrappers.accessor();

    // ****** Fill Classifier PP (PCCP) ******* //
    if (total_pp > 0) {
      FillInteractionsPPKernel<kParticlePairBlockX, kParticlePairBlockY><<<total_pp, pp_block>>>(
          grid_cells, vertex_field_data, shapes_data.data(), *rcut_inc, particle_pair_storage.cell_i_.data(),
          particle_pair_storage.cell_j_.data(), particle_pair_storage.p_i_.data(), particle_pair_storage.p_j_.data(),
          particle_pair_storage.ghost_.data(), interaction_prefix_per_pair.data(),
          particle_particle_classifier_accessor, total_pp);
      ONIKA_CU_DEVICE_SYNCHRONIZE();

      reconstruct_cell_pair_offsets(particle_pair_storage, interaction_counts_per_pair.data(), total_pp,
                                    neighbor_cell_pair_count, cell_pair_storage);
    }

    ClassifyIPDFunc driver_classifier = {
        grid_cells,         cell_driver_accessor, cell_indices,    *rcut_inc,
        shapes_data.data(), vertex_field_data,    driver_accessor, interaction_classifier_accessor};
    parallel_for(active_cell_count, driver_classifier, parallel_execution_context("nbh_gpu::classify_driver"), opts);

    ONIKA_CU_DEVICE_SYNCHRONIZE();

    UpdateHistoryFunc history_updater = {history.start_.data(),
                                         history.size_.data(),
                                         history.data_.data(),
                                         cell_interaction_info.start_cell_.data(),
                                         cell_interaction_info.number_of_pair_cells_.data(),
                                         cell_pair_accessor,
                                         cell_driver_accessor,
                                         interaction_classifier_accessor};

    parallel_for(history.start_.size(), history_updater, parallel_execution_context(), opts);

    // === ADD PERSISTENT INTERACTIONS ===
    add_unmatched_persistent_interactions(history, active_cell_count, interaction_container,
                                          interaction_classifier_accessor, cell_driver_accessor);

    constexpr bool do_ghost_only = true;
    constexpr bool do_active_interaction_only = false;
    transfer_classifier_grid<do_ghost_only, do_active_interaction_only, false>(
        cell_indices, cell_interaction_info, cell_pair_storage, cell_driver_storage, interaction_classifier_accessor,
        *ges, get_first_id<InteractionType::ParticleParticle>(), get_last_id<InteractionType::ParticleParticle>());

    transfer_classifier_grid<do_ghost_only, do_active_interaction_only, true>(
        cell_indices, cell_interaction_info, cell_pair_storage, cell_driver_storage, interaction_classifier_accessor,
        *ges, get_first_id<InteractionType::InnerBond>(), get_last_id<InteractionType::InnerBond>());

    transfer_classifier_grid<do_ghost_only, do_active_interaction_only, true>(
        cell_indices, cell_interaction_info, cell_pair_storage, cell_driver_storage, interaction_classifier_accessor,
        *ges, get_first_id<InteractionType::ParticleDriver>(), get_last_id<InteractionType::ParticleDriver>());

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
