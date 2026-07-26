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
#include <exanb/core/grid_particle_field_accessor.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_error.h>
#include <onika/cuda/device_storage.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_for.h>
#include <onika/soatl/field_id.h>

#include <exaDEM/classifier/classifier.hpp>

// mini macro here
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)

__host__ __device__ double atomicMin_double(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
  } while (assumed != old);
  return __longlong_as_double(old);
}

#define EXADEM_CU_ATOMIC_MIN(x, a, ...) atomicMin_double(&x, static_cast<std::remove_reference_t<decltype(x)>>(a));

#else
#define EXADEM_CU_ATOMIC_MIN(x, a, ...) \
  ::onika::capture_atomic_min(x, static_cast<std::remove_reference_t<decltype(x)>>(a));
#endif

namespace exaDEM {
/**
 * @namespace itools
 * Contains interaction tools for parallel processing in the exaDEM library.
 */
namespace itools /* interaction tools */
{
template <typename T>
using VectorT = onika::memory::CudaMMVector<T>;

/**
 * @struct ReduceTFunctor
 * @brief Functor for reducing data in a parallel execution.
 *
 * @tparam T The data type.
 * @tparam FuncT The function type to perform the reduction.
 * @tparam ResultT The result type that stores the reduction result.
 */
template <InteractionType IT, class FuncT, class ResultT>
struct ReduceTFunctor {
  InteractionWrapper<IT> data_; /**< Pointer to the data to be reduced. */
  const FuncT func_;            /**< Functor that defines how reduction is performed. */
  ResultT* reduced_val_;        /**< Pointer to the result of the reduction. */

  /**
   * @brief Operator to perform the reduction.
   * @param i The index of the data to reduce.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t i) const {
    ResultT local_val = ResultT();
    func_(local_val, i, data_, reduce_thread_local_t{});

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

/**
 * @struct IOSimInteractionResult
 * @brief Stores the results.
 */
struct IOSimInteractionResult {
  unsigned long long int n_act_interaction_ = 0; /**< Number of active interactions. */
  unsigned long long int n_tot_interaction_ = 0; /**< Total number of interactions. */
  double min_dn_ = 0; /**< Minimum distance between particles during interactions. Or Max overlapping between between
                        two particles. */

  /**
   * @brief Updates the current interaction result with data from another result.
   * @param in The input interaction result to merge.
   */
  void update(IOSimInteractionResult& in) {
    n_act_interaction_ += in.n_act_interaction_;
    n_tot_interaction_ += in.n_tot_interaction_;
    min_dn_ = std::min(min_dn_, in.min_dn_);
  }
};

struct IOSimInteractionFunctor {
  const double* const dnp_; /**< Pointer to the array of overlapping distances (dn). */
  const int coef_;          /**< Coefficient for symmetry (if sym -> 2 else 1) and interaction type. (driver -> 1) */

  /**
   * @brief Operator to get local interaction data.
   *
   * @param local The local result.
   * @param idx The index of the interaction.
   * @param interactions Pointer to the interaction data.
   * @param reduce_thread_local_t Tag for thread-local reduction.
   *
   * This operator processes individual particle interactions, filtering out duplicates
   * and counting active interactions where particles are overlapping (dn < 0).
   */
  template <InteractionType IT>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(IOSimInteractionResult& local, const uint64_t idx,
                                                const InteractionWrapper<IT>& interactions,
                                                reduce_thread_local_t = {}) const {
    auto I = interactions(idx);
    if (I.pair_.ghost_ != InteractionPair::PartnerGhost) {
      const double& dn = dnp_[idx];
      local.n_tot_interaction_ += coef_;
      if (dn < 0.0 || I.active()) {
        local.n_act_interaction_ += coef_;
        local.min_dn_ = std::min(local.min_dn_, dn);
      }
    }
  }

  /**
   * @brief Operator to combine local results at the block level.
   *
   * @param global The global result.
   * @param local The local result.
   * @param reduce_thread_block_t Tag for block-level reduction.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(IOSimInteractionResult& global, IOSimInteractionResult& local,
                                                reduce_thread_block_t) const {
    ONIKA_CU_ATOMIC_ADD(global.n_act_interaction_, local.n_act_interaction_);
    ONIKA_CU_ATOMIC_ADD(global.n_tot_interaction_, local.n_tot_interaction_);
    EXADEM_CU_ATOMIC_MIN(global.min_dn_, local.min_dn_);
  }

  /**
   * @brief Operator to combine results globally.
   *
   * @param global The global result.
   * @param local The local result.
   * @param reduce_global_t Tag for global reduction.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(IOSimInteractionResult& global, IOSimInteractionResult& local,
                                                reduce_global_t) const {
    ONIKA_CU_ATOMIC_ADD(global.n_act_interaction_, local.n_act_interaction_);
    ONIKA_CU_ATOMIC_ADD(global.n_tot_interaction_, local.n_tot_interaction_);
    EXADEM_CU_ATOMIC_MIN(global.min_dn_, local.min_dn_);
  }
};

/**
 * @brief Reduces data in parallel using a functor.
 *
 * @tparam T The data type.
 * @tparam Func The functor type used for the reduction.
 * @tparam ResultT The result type for storing the reduction.
 * @param exec_ctx Execution context for parallel operations.
 * @param data Pointer to the data to reduce.
 * @param func The functor for reduction.
 * @param size The size of the data array.
 * @param result The result of the reduction.
 * @return A wrapper for the parallel execution.
 */
template <InteractionType IT, typename Func, typename ResultT>
static inline onika::parallel::ParallelExecutionWrapper reduce_data(onika::parallel::ParallelExecutionContext* exec_ctx,
                                                                    InteractionWrapper<IT>& data, Func& func,
                                                                    uint64_t size, ResultT& result) {
  using namespace onika::parallel;
  using namespace onika::parallel;
  // #     if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#ifdef ONIKA_CUDA_VERSION
  ParallelForOptions opts;
  opts.omp_scheduling = OMP_SCHED_STATIC;
  ReduceTFunctor<IT, Func, ResultT> kernel = {data, func, &result};
  return parallel_for(size, kernel, exec_ctx, opts);
#else
  // should be generic, later
  int n_act_interaction(0), n_tot_interaction(0);
  double min_dn = 0;
#pragma omp parallel for reduction(+ : n_act_interaction, n_tot_interaction) reduction(min : min_dn)
  for (uint64_t i = 0; i < size; i++) {
    auto I = data(i);
    // filter duplicate (mpi ghost)
    if (I.pair_.ghost_ != InteractionPair::PartnerGhost) {
      const double& dn = func.dnp_[i];
      n_tot_interaction += func.coef_;
      if (dn < 0.0) {
        n_act_interaction += func.coef_;
        min_dn = std::min(min_dn, dn);
      }
    }
  }
  result.n_act_interaction_ = n_act_interaction;
  result.n_tot_interaction_ = n_tot_interaction;
  result.min_dn_ = min_dn;

  // useless but it avoid bugs, TODO LATER
  ParallelForOptions opts;
  opts.omp_scheduling = OMP_SCHED_STATIC;
  ReduceTFunctor<IT, Func, ResultT> kernel = {data, func, &result};
  return parallel_for(0, kernel, exec_ctx, opts);
#endif
}
}  // namespace itools
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <exaDEM::InteractionType IT, class FuncT, class ResultT>
struct ParallelForFunctorTraits<exaDEM::itools::ReduceTFunctor<IT, FuncT, ResultT>> {
  static inline constexpr bool CudaCompatible = ParallelForFunctorTraits<FuncT>::CudaCompatible;
  ;
  static inline constexpr bool RequiresBlockSynchronousCall =
      ParallelForFunctorTraits<FuncT>::RequiresBlockSynchronousCall;
};
}  // namespace parallel
}  // namespace onika
