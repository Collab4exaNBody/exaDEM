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

#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <omp.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_cell_data.hpp>
#include <exaDEM/traversal.hpp>
#include <vector>

namespace exaDEM {

/**
 * @brief Prefetches the whole storage of every cell in cell_indices[0..n_cells) to `device`.
 * `device` may be cudaCpuDeviceId to prefetch back to host.
 */
template <typename GridCellsT>
inline void prefetch_grid_cells(GridCellsT* grid_cells, const size_t* cell_indices, size_t n_cells, int device,
                                onikaStream_t* streams, int n_streams) {
#pragma omp parallel num_threads(n_streams)
  {
    onikaStream_t st = streams[omp_get_thread_num()];
#pragma omp for schedule(static)
    for (size_t i = 0; i < n_cells; i++) {
      auto& cell = grid_cells[cell_indices[i]];
      if (cell.size() <= 0) continue;
      ONIKA_PREFETCH(cell.storage_ptr(), cell.storage_size(), device, st);
    }
  }
}

static constexpr unsigned int kPrefetchLaneBase = 64;

template <typename GridT>
class PrefetchAllFieldsGPU : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        YAML example:

          - prefetch_all_fields_gpu
       )EOF";
  }

  inline void execute() final {
#ifndef ONIKA_CUDA_VERSION
    color_log::error("prefetch_all_fields_gpu", "This operator only works on GPU.");
#else
    auto grid_cells = grid->cells();
    auto [cell_indices, active_cell_count] = traversal_real->info();

    const int n_threads = omp_get_max_threads();
    std::vector<onikaStream_t> streams(n_threads);
    for (int t = 0; t < n_threads; t++) {
      streams[t] = global_cuda_ctx()->getThreadStream(kPrefetchLaneBase + t);  // try
    }

    prefetch_grid_cells(grid_cells, cell_indices, active_cell_count, 0, streams.data(), n_threads);

    for (int t = 0; t < n_threads; t++) {
      ONIKA_CU_STREAM_SYNCHRONIZE(streams[t]);
    }
#endif
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(prefetch_grid_fields_gpu) {
  OperatorNodeFactory::instance()->register_factory("prefetch_all_fields_gpu",
                                                    make_grid_variant_operator<PrefetchAllFieldsGPU>);
}
}  // namespace exaDEM
