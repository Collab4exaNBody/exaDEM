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

#pragma once

#include <exanb/compute/reduce_cell_particles.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_for.h>

namespace exaDEM {

template <typename FuncT>
struct ReduceInterfaceTraits {
  static inline constexpr bool RequireBreakInterfaceMember = false;
};

template <typename FuncT, typename ResultT>
struct ReduceInterfaceFunctor {
  Interface* const interface_;      /**< List of interfaces */
  uint8_t* const break_interface_;  // list of booleans that indicate if the interface is broken or not. 1 if the
  FuncT func_;           /**< Functor that defines how reduction is performed. */
  ResultT* reduced_val_; /**< Pointer to the result of the reduction. */

  /**
   * @brief Operator to perform the reduction.
   * @param i The index of the data to reduce.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t i) const {
    ResultT local_val = ResultT();
    if constexpr (!ReduceInterfaceTraits<FuncT>::RequireBreakInterfaceMember) {
      func_(local_val, interface_[i], reduce_thread_local_t{});
    } else {
      func_(local_val, interface_[i], break_interface_[i], reduce_thread_local_t{});
    }

    ONIKA_CU_BLOCK_SHARED onika::cuda::UnitializedPlaceHolder<ResultT> team_val_place_holder;
    ResultT& team_val = team_val_place_holder.get_ref();

    if (ONIKA_CU_THREAD_IDX == 0) {
      team_val = local_val;
    }
    ONIKA_CU_BLOCK_SYNC();

    if (ONIKA_CU_THREAD_IDX != 0) {
      func_(team_val, local_val, reduce_thread_block_t{});
    }
    ONIKA_CU_BLOCK_SYNC();

    if (ONIKA_CU_THREAD_IDX == 0) {
      func_(*reduced_val_, team_val, reduce_global_t{});
    }
  }
};

template <typename FuncT, typename ResultT>
static inline ResultT reduce_interface(
    InterfaceManager& interface, FuncT& func, ResultT& init, onika::parallel::ParallelExecutionContext* exec_ctx) {
  using namespace onika::parallel;
  ParallelForOptions opts;
  opts.omp_scheduling = OMP_SCHED_STATIC;
  onika::memory::CudaMMVector<ResultT> array(1);
  array[0] = init;

  ReduceInterfaceFunctor reduce = {interface.data_.data(), interface.break_interface_.data(), func, array.data()};
  parallel_for(interface.data_.size(), reduce, exec_ctx, opts);
  return array[0];
}
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <class FuncT, class ResultT>
struct ParallelForFunctorTraits<exaDEM::ReduceInterfaceFunctor<FuncT, ResultT>> {
  static inline constexpr bool RequiresBlockSynchronousCall =
      ParallelForFunctorTraits<FuncT>::RequiresBlockSynchronousCall;
  static inline constexpr bool CudaCompatible = ParallelForFunctorTraits<FuncT>::CudaCompatible;
};
}  // namespace parallel
}  // namespace onika
