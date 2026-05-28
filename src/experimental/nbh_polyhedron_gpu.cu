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

#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>

#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/particle_neighbors/chunk_neighbors.h>
#include <exanb/particle_neighbors/chunk_neighbors_apply.h>

#include <cassert>

#include <exaDEM/traversal.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/interaction/interaction_manager.hpp>
#include <exaDEM/interaction/interaction_enum.hpp>
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/polyhedron/nbh_polyhedron_driver.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_utils.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_storage.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_cell_data.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_interaction_history.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_manager.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_driver.hpp>
#include <cub/cub.cuh>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_gpu_pccp.hpp>

namespace exaDEM {
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

 public:
	inline std::string documentation() const final {
		return R"EOF(
				This function builds the list of interactions per particle (polyhedron). Block-per-particle-pair (PCCP) version.

				YAML example [no option]:

					- nbh_polyhedron_gpu_pccp
			 )EOF";
	}

	inline void execute() final {
		using namespace onika::parallel;
#ifndef ONIKA_CUDA_VERSION
		color_log::error("nbh_polyhedron_gpu_pccp", "This operator only work on GPU.\n"
										 "                     Please use nbh_polyhedron.");
#else
		constexpr int block_size_x = 8;
		constexpr int block_size_y = 8;
		auto& g = *grid;
		const auto cells = g.cells();
		const IJK dims = g.dimension();
		const size_t n_cells = g.number_of_cells();
		shapes& shps = *shapes_collection;
		auto [cell_ptr, cell_size] = traversal_real->info();
		auto* vertex_fields = cvf->data();
		auto& container = *ic;
		const DriversGPUAccessor drvs = *drivers;

		auto get_exec_ctx = [this] () {
			return this->parallel_execution_context();
		};

		onikaStream_t st_updateghost, st_history, st_particle, st_driver, st_innerbond;
		ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_updateghost);
		ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_history);
		ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_particle);
		ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_driver);
		ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_innerbond);

		NbhCellHostStorage CellInfoStorageHost;
		CellInteractionInformation& info_cell = nbh_manager->info_cell;

		auto convert_offset_ijk = [] (int offset) {
			assert(offset < 27);
			IJK res;
			res.i = offset % 3 - 1;
			res.j = (offset / 3) % 3 - 1;
			res.k = offset / 9 - 1;
			return res;
		};

		info_cell.resize(cell_size);
		std::memset(info_cell.update_ghost.data(), 0, cell_size * sizeof(uint8_t));
		size_t shift = 0;
		for(size_t i = 0 ; i < cell_size ; i++) {
			info_cell.start_cell[i] = shift;
			size_t incr = 0;
			size_t cell_a = cell_ptr[i];
			IJK loc_a = grid_index_to_ijk(dims, cell_a);

			for (size_t j = 0; j < 27 ; j++) {
				size_t cell_b = grid_ijk_to_index(
						dims, loc_a + convert_offset_ijk(j));
				if (cells[cell_b].size() > 0) {
					CellInfoStorageHost.owner_cell.push_back(cell_a);
					CellInfoStorageHost.partner_cell.push_back(cell_b);
					if (g.is_ghost_cell(cell_b)) {
						info_cell.update_ghost[i] = 1;
						CellInfoStorageHost.ghost.push_back(InteractionPair::OwnerGhost);
					} else {
						CellInfoStorageHost.ghost.push_back(InteractionPair::NotGhost);
					}
					incr++;
				}
			}
			info_cell.number_of_pair_cells[i] = incr;
			shift += incr;
		}
		info_cell.prefetch_cpu(st_updateghost);
		CellDriverStorage& info_cell_driver = nbh_manager->info_cell_driver;
		info_cell_driver.resize(cell_size, get_exec_ctx);
		auto cell_driver_accessor = info_cell_driver.accessor();

		NbhCellStorage& info_cell_pair = nbh_manager->info_pair_cell;
		info_cell_pair.reset(CellInfoStorageHost, get_exec_ctx);

		NbhCellAccessor accessor(info_cell_pair);

		ParallelForOptions opts;
		opts.omp_scheduling = OMP_SCHED_GUIDED;
		BlockParallelForOptions bopts;

		CountIPDFunc counter_driver = {
			cells, cell_driver_accessor,
			cell_ptr, *rcut_inc, shps.data(),
			vertex_fields, drvs};

		auto cell_pair_size = CellInfoStorageHost.owner_cell.size();
		ONIKA_CU_DEVICE_SYNCHRONIZE();

		parallel_for(cell_size, counter_driver, parallel_execution_context("nbh_gpu::counter_driver,"), opts);
		PrefixSumInteractionTypePerCellCounter func_prefix_driver {
			cell_driver_accessor.offset, cell_driver_accessor.size, cell_size};

		ParallelExecutionSpace<1> parallel_range_fpd = {
			 {get_first_id<InteractionType::ParticleDriver>()},
			 { get_last_id<InteractionType::ParticleDriver>() + 1}};

		ONIKA_CU_DEVICE_SYNCHRONIZE();
		parallel_for(parallel_range_fpd, func_prefix_driver, parallel_execution_context("nbh_gpu::func_prefix_driver"), opts);

		InteractionHistory history;

		setup_history_clean_ges(cells, cell_ptr, cell_size,
														*ges, history, st_history);

		ONIKA_CU_DEVICE_SYNCHRONIZE();

		// ****** Build particle pairs (PCCP) ******* //
		constexpr int pp_block_x = 8;
		constexpr int pp_block_y = 8;
		dim3 pp_block(pp_block_x, pp_block_y, 1);

		onika::memory::CudaMMVector<int> pp_counts;
		pp_counts.resize(cell_pair_size);

		CountParticlePairsKernel<pp_block_x, pp_block_y>
		    <<<cell_pair_size, pp_block>>>(
		    cells, accessor.owner_cell, accessor.partner_cell,
		    accessor.ghost, *rcut_inc, shps.data(), vertex_fields,
		    pp_counts.data(), cell_pair_size);
		ONIKA_CU_DEVICE_SYNCHRONIZE();

		onika::memory::CudaMMVector<int> pp_offsets;
		pp_offsets.resize(cell_pair_size);
		{
		  void* d_tmp = nullptr;
		  size_t tmp_bytes = 0;
		  cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes,
		      pp_counts.data(), pp_offsets.data(), cell_pair_size);
		  cudaMalloc(&d_tmp, tmp_bytes);
		  cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes,
		      pp_counts.data(), pp_offsets.data(), cell_pair_size);
		  cudaFree(d_tmp);
		}
		ONIKA_CU_DEVICE_SYNCHRONIZE();

		size_t total_pp = 0;
		if (cell_pair_size > 0)
		  total_pp = pp_counts[cell_pair_size-1] + pp_offsets[cell_pair_size-1];

		ParticlePairStorage pp_storage;
		pp_storage.resize(total_pp);

		if (total_pp > 0) {
		  FillParticlePairsKernel<pp_block_x, pp_block_y>
		      <<<cell_pair_size, pp_block>>>(
		      cells, accessor.owner_cell, accessor.partner_cell,
		      accessor.ghost, *rcut_inc, shps.data(), vertex_fields,
		      pp_offsets.data(),
		      pp_storage.cell_i.data(), pp_storage.cell_j.data(),
		      pp_storage.p_i.data(), pp_storage.p_j.data(),
		      pp_storage.ghost.data(), pp_storage.cell_pair_idx.data(),
		      cell_pair_size);
		  ONIKA_CU_DEVICE_SYNCHRONIZE();
		}

		// ****** Count interactions per particle pair (PCCP) ******* //
		onika::memory::CudaMMVector<InteractionTypePerCellCounter> interaction_counts;
		interaction_counts.resize(total_pp);
		onika::memory::CudaMMVector<InteractionTypePerCellCounter> interaction_prefix;
		interaction_prefix.resize(total_pp);
		InteractionTypePerCellCounter total_interactions;
		for (int typeID = 0; typeID < InteractionTypeId::NTypes; typeID++) 
		{
			total_interactions[typeID] = 0;
		}

		if (total_pp > 0) {
		  CountInteractionsPPKernel<pp_block_x, pp_block_y>
		      <<<total_pp, pp_block>>>(
		      cells, vertex_fields, shps.data(), *rcut_inc,
		      pp_storage.cell_i.data(), pp_storage.cell_j.data(),
		      pp_storage.p_i.data(), pp_storage.p_j.data(),
		      interaction_counts.data(), total_pp);
		  ONIKA_CU_DEVICE_SYNCHRONIZE();

		  // GPU prefix sum per interaction type
		  onika::memory::CudaMMVector<int> type_counts[InteractionTypeId::NTypesPP];
		  onika::memory::CudaMMVector<int> type_prefix[InteractionTypeId::NTypesPP];
		  for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
		    type_counts[t].resize(total_pp);
		    type_prefix[t].resize(total_pp);
		  }

		  int block_1d = 256;
		  int grid_1d = (total_pp + block_1d - 1) / block_1d;

		  ExtractInteractionCounts<<<grid_1d, block_1d>>>(
		      interaction_counts.data(),
		      type_counts[0].data(), type_counts[1].data(),
		      type_counts[2].data(), type_counts[3].data(),
		      total_pp);
		  ONIKA_CU_DEVICE_SYNCHRONIZE();

		  for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
		    void* d_tmp = nullptr;
		    size_t tmp_bytes = 0;
		    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes,
		        type_counts[t].data(), type_prefix[t].data(), total_pp);
		    cudaMalloc(&d_tmp, tmp_bytes);
		    cub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes,
		        type_counts[t].data(), type_prefix[t].data(), total_pp);
		    cudaFree(d_tmp);
		  }
		  ONIKA_CU_DEVICE_SYNCHRONIZE();

		  PackInteractionPrefix<<<grid_1d, block_1d>>>(
		      interaction_prefix.data(),
		      type_prefix[0].data(), type_prefix[1].data(),
		      type_prefix[2].data(), type_prefix[3].data(),
		      total_pp);
		  ONIKA_CU_DEVICE_SYNCHRONIZE();

		  for (int t = 0; t < InteractionTypeId::NTypesPP; t++) {
		    total_interactions[t] = type_prefix[t][total_pp - 1]
		                         + type_counts[t][total_pp - 1];
		  }
		}

		// ****** Resize Classifier for PP ******* //
		InteractionParticleAccessor classifier_accessor;
		for (int typeID = get_first_id<InteractionType::ParticleParticle>();
		     typeID <= get_last_id<InteractionType::ParticleParticle>(); typeID++) {
		  auto& c = container.get_data<ParticleParticle>(typeID);
		  c.resize(total_interactions[typeID]);
		  classifier_accessor[typeID] = InteractionWrapper(c);
		}

		// ****** Resize Classifier for Driver ******* //
		for (int typeID = get_first_id<InteractionType::ParticleDriver>();
		     typeID <= get_last_id<InteractionType::ParticleDriver>(); typeID++) {
		  size_t newsize = info_cell_driver.offset.back()[typeID]
		                 + info_cell_driver.size.back()[typeID];
		  container.resize(typeID, newsize);
		}

		InteractionWrapperStorage wrappers(container);
		InteractionWrapperAccessor classifier_interaction_accessor = wrappers.accessor();

		// ****** Fill Classifier PP (PCCP) ******* //
		if (total_pp > 0) {
		  FillInteractionsPPKernel<pp_block_x, pp_block_y>
		      <<<total_pp, pp_block>>>(
		      cells, vertex_fields, shps.data(), *rcut_inc,
		      pp_storage.cell_i.data(), pp_storage.cell_j.data(),
		      pp_storage.p_i.data(), pp_storage.p_j.data(),
		      pp_storage.ghost.data(),
		      interaction_prefix.data(),
		      classifier_accessor, total_pp);
		  ONIKA_CU_DEVICE_SYNCHRONIZE();

		  reconstruct_cell_pair_offsets(pp_storage, interaction_counts.data(),
		      total_pp, cell_pair_size, info_cell_pair);
		}

		ClassifyIPDFunc classify_driver = {
		  cells, cell_driver_accessor,
		  cell_ptr, *rcut_inc, shps.data(),
		  vertex_fields, drvs, classifier_interaction_accessor};
		parallel_for(cell_size, classify_driver,
		  parallel_execution_context("nbh_gpu::classify_driver"), opts);

		ONIKA_CU_DEVICE_SYNCHRONIZE();
		
		UpdateHistoryFunc update_history = {
      history.start.data(),
      history.size.data(),
      history.data.data(),
      info_cell.start_cell.data(),
      info_cell.number_of_pair_cells.data(),
      accessor,
      cell_driver_accessor,
      classifier_interaction_accessor};

		parallel_for(history.start.size(), update_history, parallel_execution_context(), opts);

		// === ADD PERSISTENT INTERACTIONS ===
    {
      std::vector<PlaceholderInteraction> unmatched_persistent;
      for (size_t ci = 0; ci < cell_size; ci++) {
        size_t hist_begin = history.start[ci];
        size_t hist_end = hist_begin + history.size[ci];
        for (size_t h = hist_begin; h < hist_end; h++) {
          PlaceholderInteraction I = history.data[h];
          auto type = I.type();
          if (type < get_first_id<InteractionType::ParticleDriver>() ||
              type > get_last_id<InteractionType::ParticleDriver>())
            continue;
          if (!I.persistent()) continue;
          auto& wrapper = classifier_interaction_accessor.get_typed_accessor<InteractionType::ParticleDriver>(type);
          int drv_offset = cell_driver_accessor.offset[ci][type];
          int drv_size = cell_driver_accessor.size[ci][type];
          bool found = false;
          for (int k = drv_offset; k < drv_offset + drv_size; k++) {
            if (wrapper.same(k, I)) {
              found = true;
              break;
            }
          }
          if (!found) {
            unmatched_persistent.push_back(I);
          }
        }
      }
      if (!unmatched_persistent.empty()) {
        lout << "[PERSISTENT] adding " << unmatched_persistent.size()
             << " unmatched persistent driver interactions" << std::endl;
        for (auto& I : unmatched_persistent) {
          auto type = I.type();
          auto& c = container.get_data<InteractionType::ParticleDriver>(type);
          size_t old_size = c.size();
          container.resize(type, old_size + 1);
          InteractionWrapperStorage wrappers_tmp(container);
          InteractionWrapperAccessor tmp_accessor = wrappers_tmp.accessor();
          auto& w = tmp_accessor.get_typed_accessor<InteractionType::ParticleDriver>(type);
          w.set(old_size, I);
        }
        InteractionWrapperStorage wrappers2(container);
        classifier_interaction_accessor = wrappers2.accessor();
      }
    }

		constexpr bool do_ghost_only = true;
		constexpr bool do_active_interaction_only = false;
		transfer_classifier_grid<do_ghost_only, do_active_interaction_only, false>(
			cell_ptr, info_cell, info_cell_pair,
			info_cell_driver,
			classifier_interaction_accessor, *ges,
			get_first_id<InteractionType::ParticleParticle>(),
			get_last_id<InteractionType::ParticleParticle>()
		);

		transfer_classifier_grid<do_ghost_only, do_active_interaction_only, true>(
			cell_ptr, info_cell, info_cell_pair,
			info_cell_driver,
			classifier_interaction_accessor, *ges,
			get_first_id<InteractionType::InnerBond>(),
			get_last_id<InteractionType::InnerBond>()
		);

		transfer_classifier_grid<do_ghost_only, do_active_interaction_only, true>(
			cell_ptr, info_cell, info_cell_pair,
			info_cell_driver,
			classifier_interaction_accessor, *ges,
			get_first_id<InteractionType::ParticleDriver>(),
			get_last_id<InteractionType::ParticleDriver>()
		);

#endif
	}
};

// === register factories ===
ONIKA_AUTORUN_INIT(nbh_polyhedron_gpu_pccp) {
	OperatorNodeFactory::instance()->register_factory("nbh_polyhedron_gpu",
																										make_grid_variant_operator<UpdateClassifierPolyhedronGPUPCCP>);
}
}  // namespace exaDEM
#endif
