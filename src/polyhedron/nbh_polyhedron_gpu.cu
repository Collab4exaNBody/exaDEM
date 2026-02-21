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
#include <exaDEM/polyhedron/nbh_dem.hpp>
#include <exaDEM/polyhedron/nbh_gpu.hpp>
#include <exaDEM/polyhedron/nbh_runner.hpp>

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

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This function builds the list of interactions per particle (polyhedron). Interactions are between two particles or a particle and a driver. In this function, frictions and moments are updated if the interactions are still actived. Note that, a list of non-empty cells is built during this function.

        YAML example [no option]:

          - nbh_polyhedron_gpu
       )EOF";
  }

  inline void execute() final {
#ifndef ONIKA_CUDA_VERSION
    color_log::error("nbh_polyhedron_gpu", "This operator only work on GPU.\n"
                     "                     Please use nbh_polyhedron.");
#else
    lout << "Operator in progress" << std::endl;
    auto& g = *grid;
    const auto cells = g.cells();
    const IJK dims = g.dimension();
    const size_t n_cells = g.number_of_cells();
    shapes& shps = *shapes_collection;
    auto [cell_ptr, cell_size] = traversal_real->info();
    auto* vertex_fields = cvf->data();
    auto& container = *ic;

    // CPU Stage : get cell a / cell b 
    // Reset other fields (particle counters per type / skip)
    NbhManagerStrorageHost CellInfoStorageHost;
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
    for(size_t i = 0 ; i < cell_size ; i++) {
      size_t cell_a = cell_ptr[i];
      IJK loc_a = grid_index_to_ijk(dims, cell_a);

      for (size_t j = 0; j < 26 ; j++) {
        size_t cell_b = grid_ijk_to_index(
            dims, loc_a + convert_offset_ijk(j));
        if (cells[cell_b].size() > 0) {
          lout << "add " << cell_a << " - " << cell_b << std::endl;
          CellInfoStorageHost.owner_cell.push_back(cell_a);
          CellInfoStorageHost.partner_cell.push_back(cell_b);
        }        
      }
    }

    auto get_exec_ctx = [this] () {
      return this->parallel_execution_context();
    };
    lout << "Copy to GPU" << std::endl;
    // copy to gpu here
    NbhManagerStrorage cellinfostorage(CellInfoStorageHost, get_exec_ctx);

    lout << "cellinfostorage cella: " << cellinfostorage.owner_cell[0] << std::endl;
    // define a wrapper function
    NbhManagerAcessor accessor(cellinfostorage);

    ParallelForOptions opts;
    opts.omp_scheduling = OMP_SCHED_GUIDED;

		// ****** First pass ******* //
    // prepare ApplyNbhFunc
		ApplyNbhFunc apply = {
      cells, accessor, *rcut_inc, shps.data(), vertex_fields};
    auto cell_pair_size = CellInfoStorageHost.owner_cell.size();
    ONIKA_CU_DEVICE_SYNCHRONIZE();
    lout << "Get the number of interactions" << std::endl;
 
    // run ApplyNbhFunc
		parallel_for(cell_pair_size, apply, parallel_execution_context(), opts);

		// fill offset
    lout << "Prefix sum to get offsets per cell pairs" << std::endl;
    PrefxSumInteractionTypePerCellCounter func_prefix = {
      accessor.offset, accessor.size, cell_pair_size};
    parallel_for(4, func_prefix, parallel_execution_context(), opts);

    ONIKA_CU_DEVICE_SYNCHRONIZE();
    debug_print(cellinfostorage.offset.back(), cellinfostorage.size.back());

	  // ****** Second pass ******* //
	  // prepare ApplyClassifierFunc
    auto new_size = cellinfostorage.offset.back() + cellinfostorage.size.back();
    using array_i = onika::oarray_t<InteractionWrapper<ParticleParticle>, ParticleParticleSize>;
    array_i classifier_accessor;
		for (size_t type_id = 0; type_id<ParticleParticleSize ; type_id++) {
			auto& c = container.get_data<ParticleParticle>(type_id);
			c.resize(new_size[type_id]);
      classifier_accessor[type_id] = InteractionWrapper(c); 
		}
	  ApplyClassifierFunc<8,8, decltype(cells)> filler = {cells, accessor,
      *rcut_inc, shps.data(),
      vertex_fields, classifier_accessor};

		// run ApplyClassifierFunc 
		parallel_for(cell_pair_size, filler, parallel_execution_context(), opts);
#endif
	}
};

// === register factories ===
ONIKA_AUTORUN_INIT(nbh_polyhedron_gpu) {
	OperatorNodeFactory::instance()->register_factory("nbh_polyhedron_gpu",
																										make_grid_variant_operator<UpdateClassifierPolyhedronGPU>);
}
}  // namespace exaDEM
