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
#include <exaDEM/interaction/migration_test.hpp>
#include <exaDEM/classifier/classifier_transfert.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/polyhedron/nbh_polyhedron_driver.hpp>
#include <exaDEM/polyhedron/nbh_gpu/nbh_utils.hpp>
#include <exaDEM/polyhedron/nbh_gpu/nbh_storage.hpp>
#include <exaDEM/polyhedron/nbh_gpu/nbh_gpu.hpp>
#include <exaDEM/polyhedron/nbh_gpu/nbh_cell_data.hpp>
#include <exaDEM/polyhedron/nbh_gpu/nbh_interaction_history.hpp>
#include <exaDEM/polyhedron/nbh_gpu/nbh_manager.hpp>

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT>>
class UpdateClassifierPolyhedronGPU : public OperatorNode {

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
        This function builds the list of interactions per particle (polyhedron). Interactions are between two particles or a particle and a driver. In this function, frictions and moments are updated if the interactions are still actived. Note that, a list of non-empty cells is built during this function.

        YAML example [no option]:

          - nbh_polyhedron_gpu
       )EOF";
  }

  inline void execute() final {
    using namespace onika::parallel;
#ifndef ONIKA_CUDA_VERSION
    color_log::error("nbh_polyhedron_gpu", "This operator only work on GPU.\n"
                     "                     Please use nbh_polyhedron.");
#else
    constexpr int block_size_x = 8;
    constexpr int block_size_y = 8;
    auto& g = *grid;
    const auto cells = g.cells();
    using ACF = ApplyClassifierFunc<block_size_x, block_size_y, decltype(cells)>;
    lout << "Operator in progress" << std::endl;
    const IJK dims = g.dimension();
    const size_t n_cells = g.number_of_cells();
    shapes& shps = *shapes_collection;
    auto [cell_ptr, cell_size] = traversal_real->info();
    auto* vertex_fields = cvf->data();
    auto& container = *ic;

    //onikaStream_t&
    onikaStream_t st_updateghost, st_history, st_particle, st_driver, st_innerbond;
    ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_updateghost);
    ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_history);
    ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_particle);
    ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_driver);
    ONIKA_CU_CREATE_STREAM_NON_BLOCKING(st_innerbond);

    // CPU Stage : get cell a / cell b 
    // Reset other fields (particle counters per type / skip)
    NbhCellHostStorage CellInfoStorageHost;
    CellInteractionInformation& info_cell = nbh_manager->info_cell;
		// Sequential
    auto convert_offset_ijk = [] (int offset) {
      assert(offset < 27);
      IJK res;
      res.i = offset % 3 - 1;
      res.j = (offset / 3) % 3 - 1;
      res.k = offset / 9 - 1;
      return res;
    };

    //
    lout << "Fill CPU data storage" << std::endl;
    info_cell.resize(cell_size);
    size_t shift = 0;
		for(size_t i = 0 ; i < cell_size ; i++) {
      info_cell.start_cell[i] = shift;
      size_t incr = 0;
			size_t cell_a = cell_ptr[i];
      IJK loc_a = grid_index_to_ijk(dims, cell_a);

      for (size_t j = 0; j < 26 ; j++) {
        size_t cell_b = grid_ijk_to_index(
            dims, loc_a + convert_offset_ijk(j));
        if (cells[cell_b].size() > 0) {
          CellInfoStorageHost.owner_cell.push_back(cell_a);
          CellInfoStorageHost.partner_cell.push_back(cell_b);
          incr++;
        }        
      }
      info_cell.number_of_pair_cells[i] = incr;
      shift += incr;
    }
    info_cell.prefetch_cpu(st_updateghost);


    auto get_exec_ctx = [this] () {
      return this->parallel_execution_context();
    };
    lout << "Copy to GPU" << std::endl;
    // copy to gpu here
    NbhCellStorage& info_cell_pair = nbh_manager->info_pair_cell;
    info_cell_pair.reset(CellInfoStorageHost, get_exec_ctx);

    // define a wrapper function
    NbhCellAccessor accessor(info_cell_pair);

		ParallelForOptions opts;
    opts.omp_scheduling = OMP_SCHED_GUIDED;
		BlockParallelForOptions bopts;

		// ****** First pass ******* //
    // prepare ApplyNbhFunc
		ApplyNbhFunc apply = {
      cells, accessor,
      *rcut_inc, shps.data(),
      vertex_fields};

    auto cell_pair_size = CellInfoStorageHost.owner_cell.size();
    ONIKA_CU_DEVICE_SYNCHRONIZE();
    lout << "Get the number of interactions" << std::endl;
 
    // run ApplyNbhFunc
		auto pec1 = parallel_execution_context();
		pec1->s_gpu_block_dims = {block_size_x, block_size_y, 1};  // enforce block size 
		ParallelExecutionSpace<3> parallel_range = {
       {0, 0, 0}, {static_cast<long>(cell_pair_size), 1, 1} };
		block_parallel_for(parallel_range, apply, pec1, bopts);


    // Recover it by history update on CPU
    // project ghost / edge interactions
    InteractionHistory history;
    setup_history_clean_ges(cells, cell_ptr, cell_size,
                            *ges, history, st_history);

		// fill offset
    lout << "Prefix sum to get offsets per cell pairs" << std::endl;
    PrefixSumInteractionTypePerCellCounter func_prefix = {
      accessor.offset, accessor.size, cell_pair_size};
    parallel_for(4, func_prefix, parallel_execution_context(), opts);

    ONIKA_CU_DEVICE_SYNCHRONIZE();

#ifdef DEBUG_NBH_GPU
    debug_print(info_cell_pair.offset.back(), info_cell_pair.size.back());
#endif

	  // ****** Second pass ******* //
	  // prepare ApplyClassifierFunc
    auto new_size = info_cell_pair.offset.back() + info_cell_pair.size.back();
    InteractionParticleAccessor classifier_accessor;
    lout << "Prepare containers (classifier)" << std::endl;
		for (size_t type_id = 0; type_id<ParticleParticleSize ; type_id++) {
				auto& c = container.get_data<ParticleParticle>(type_id);
				c.resize(new_size[type_id]);
				classifier_accessor[type_id] = InteractionWrapper(c); 
		}

		ACF filler = {
			cells, accessor,
			*rcut_inc, shps.data(),
			vertex_fields, classifier_accessor};

		// run ApplyClassifierFunc 
		lout << "Copy interaction within the classifier" << std::endl;
		auto pec2 = parallel_execution_context();
		pec2->s_gpu_block_dims = {block_size_x, block_size_y, 1};  // enforce block size 
		onika::parallel::block_parallel_for(parallel_range, filler, pec2, bopts);
		lout << "victory" << std::endl;

    lout << "Prepare containers (classifier)" << std::endl;
    InteractionWrapperStorage wrappers(container);
    InteractionWrapperAccessor classifier_interaction_accessor = wrappers.accessor();

		UpdateHistoryFunc update_history = {
			history.start.data(),
			history.size.data(),
			history.data.data(), 
			info_cell.start_cell.data(),
			info_cell.number_of_pair_cells.data(),
			accessor,
		  classifier_interaction_accessor};

    parallel_for(history.start.size(), update_history, parallel_execution_context(), opts);


    // update ghost area with interactions
    lout << "Project ghost interactions on grid." << std::endl;
    constexpr bool do_ghost_only = false;  // should be true, but false is better for debugging
    constexpr bool do_active_interaction_only = false;
		transfer_classifier_grid<do_ghost_only, do_active_interaction_only>(
      cell_ptr, info_cell,  info_cell_pair,
      classifier_interaction_accessor, *ges);

#ifdef DEBUG_NBH_GPU
		// Debug check
		ONIKA_CU_DEVICE_SYNCHRONIZE();
		for (size_t type_id = 1; type_id<ParticleParticleSize ; type_id++) {
      lout << "type_id " << type_id << std::endl;
			auto& InteractionList = container.get_data<ParticleParticle>(type_id);
			for(size_t i = 0 ; i < InteractionList.size(); i++) {
				auto I = InteractionList[i];
				if( I.pair.type == 0 && I.pair.pj.sub > 8 ) {
				  I.print();
        }
			} 
		}
#endif

#endif
	}
};

// === register factories ===
ONIKA_AUTORUN_INIT(nbh_polyhedron_gpu) {
	OperatorNodeFactory::instance()->register_factory("nbh_polyhedron_gpu",
																										make_grid_variant_operator<UpdateClassifierPolyhedronGPU>);
}
}  // namespace exaDEM
