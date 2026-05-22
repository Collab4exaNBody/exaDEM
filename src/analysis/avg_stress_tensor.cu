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
#include <exaDEM/atomic.h>
#include <mpi.h>
// onika
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
// exanb
#include <exanb/core/concurent_add_contributions.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
// reduction
#include <exanb/compute/reduce_cell_particles.h>
// exaDEM
#include <exaDEM/traversal.hpp>

namespace exaDEM {
using exanb::Mat3d;

/** @brief Functor for reducing stress tensors. */
struct ReduceStressTensorFunctor {
  ONIKA_HOST_DEVICE_FUNC inline void operator()(Mat3d& local, const Mat3d& stress, reduce_thread_local_t = {}) const {
    local += stress;
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(Mat3d& global, Mat3d& local, reduce_thread_block_t) const {
    exaDEM::mat3d_atomic_add_block_contribution(global, local);
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(Mat3d& global, Mat3d& local, reduce_global_t) const {
    exaDEM::mat3d_atomic_add_contribution(global, local);
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ReduceCellParticlesTraits<exaDEM::ReduceStressTensorFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool RequiresCellParticleIndex = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT, field::_stress>>
class AverageStressTensor : public OperatorNode {
  // attributes processed during computation
  using ReduceFields = FieldSet<field::_stress>;
  static constexpr ReduceFields reduce_field_set{};

  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});
  ADD_SLOT(double, volume, INPUT, REQUIRED, DocString{"Volume of the domain simulation. >0 "});
  ADD_SLOT(Mat3d, stress_tensor, OUTPUT, DocString{"Write an Output file containing stress tensors."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
    This operator computes the total stress tensor and the stress tensor for each particles.
    
    Parameter:
    - [double] volume: Volume of the domain simulation. >0

    YAML example:

    - operator: avg_stress_tensor
      volume: 1000
    )EOF";
  }

  inline void execute() final {
    // Skip operation if there are no cells in the grid.
    if (grid->number_of_cells() == 0) {
      return;
    }

    assert(*volume > 0);

    // Get options for reduce_cell_particles from the traversal
    const ReduceCellParticlesOptions rcpo = traversal_real->get_reduce_cell_particles_options();

    // Compute the total stress tensor by reducing the stress tensor of each particle
    Mat3d stress = exanb::make_zero_matrix();
    ReduceStressTensorFunctor func;

    // Reduce the stress tensor of each particle to compute the total stress tensor. The reduction is done in parallel
    // on the GPU if available, otherwise it is done on the CPU.
    reduce_cell_particles(*grid, false, func, stress, reduce_field_set, parallel_execution_context(), {}, rcpo);

    double buff[9] = {stress.m11, stress.m12, stress.m13, stress.m21, stress.m22,
                      stress.m23, stress.m31, stress.m32, stress.m33};

    MPI_Allreduce(MPI_IN_PLACE, buff, 9, MPI_DOUBLE, MPI_SUM, *mpi);
    stress = {buff[0], buff[1], buff[2], buff[3], buff[4], buff[5], buff[6], buff[7], buff[8]};

    // Store the average stress tensor in the output slot
    *stress_tensor = stress / (*volume);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(avg_stress_tensor) {
  OperatorNodeFactory::instance()->register_factory("avg_stress_tensor",
                                                    make_grid_variant_operator<AverageStressTensor>);
}
}  // namespace exaDEM
