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
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// exaNBody
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <mpi.h>

// reduce
#include <exanb/compute/reduce_cell_particles.h>

// exaDEM
#include <exaDEM/forcefield/contact_parameters.hpp>
#include <exaDEM/forcefield/multimat_parameters.hpp>
#include <exaDEM/traversal.hpp>

namespace exaDEM {

/** Reduces the maximum value of the 'group' particle field. */
struct ReduceMaxGroupFunctor {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint32_t& local, uint32_t group, reduce_thread_local_t = {}) const {
    local = std::max(local, group);
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint32_t& global, const uint32_t& local, reduce_thread_block_t) const {
    ONIKA_CU_ATOMIC_MAX(global, local);
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint32_t& global, const uint32_t& local, reduce_global_t) const {
    ONIKA_CU_ATOMIC_MAX(global, local);
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ReduceCellParticlesTraits<exaDEM::ReduceMaxGroupFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = true;
  static inline constexpr bool RequiresCellParticleIndex = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb

namespace exaDEM {
using namespace exanb;

template <typename GridT, class = AssertGridHasFields<GridT, field::_group>>
class CheckGroupCompleteness : public OperatorNode {
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(MultiMatParamsT<ContactParams>, multimat_cp, INPUT_OUTPUT, OPTIONAL,
           DocString{"List of contact parameters for simulations with multiple materials"});
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});

  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
        Checks that the number of groups set up in multimat_cp (via multimat_contact_params and/or
        drivers_contact_params) is large enough to cover the maximum group index actually present
        in the 'group' particle field. Stops the simulation if it is not the case.

        YAML example [no option]:

          - check_group_completeness
      )EOF";
  }

  inline std::string operator_name() { return "check_group_completeness"; }

 public:
  inline void execute() final {
    if (!multimat_cp.has_value()) {
      return;
    }

    uint32_t local = 0;
    ReduceMaxGroupFunctor func;
    const ReduceCellParticlesOptions rcpo = traversal_real->get_reduce_cell_particles_options();
    if (rcpo.m_num_cell_indices > 0) {
      auto user_cb = onika::parallel::ParallelExecutionCallback{};
      reduce_cell_particles(*grid, false, func, local, FieldSet<field::_group>{}, parallel_execution_context(), user_cb,
                            rcpo);
    }

    uint32_t max_group = 0;
    MPI_Allreduce(&local, &max_group, 1, MPI_UNSIGNED, MPI_MAX, *mpi);
    const uint32_t n_groups_field = max_group + 1;

    const int n_groups_cp = multimat_cp->size_groups();
    if (n_groups_field > static_cast<uint32_t>(n_groups_cp)) {
      color_log::error(operator_name(),
                       "The 'group' particle field contains group index " + std::to_string(n_groups_field - 1) +
                           ", but multimat_cp was only set up for " + std::to_string(n_groups_cp) +
                           " group(s). Increase n_groups in multimat_contact_params / drivers_contact_params.");
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(check_group_completeness) {
  OperatorNodeFactory::instance()->register_factory("check_group_completeness",
                                                    make_grid_variant_operator<CheckGroupCompleteness>);
}
}  // namespace exaDEM
