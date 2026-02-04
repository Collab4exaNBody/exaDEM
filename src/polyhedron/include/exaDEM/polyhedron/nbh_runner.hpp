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

#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>


namespace exaDEM {
using namespace onika::parallel;

/**
 * @brief Namespace for utilities related to tuple manipulation.
 */
namespace tuple_helper {
template <size_t... Is>
struct index {};

template <size_t N, size_t... Is>
struct gen_seq : gen_seq<N - 1, N - 1, Is...> {};

template <size_t... Is>
struct gen_seq<0, Is...> : index<Is...> {};
}  // namespace tuple_helper

template <typename Func, typename... Args>
struct NeighborRunner {
  const size_t* const cell_idx;
  IJK dims;
  Func func;
  std::tuple<Args...> params; /**< Tuple of parameters to be passed to the kernel function. */

  NeighborRunner(const size_t* const cells, const IJK& d, Func& f, Args... args)
      : cell_idx(cells), dims(d), func(f), params(std::tuple<Args...>(args...)) {}


  IJK convert_offset_ijk(int offset) const {
    assert(offset < 27);
    IJK res;
    res.i = offset % 3 - 1;
    res.j = (offset / 3) % 3 - 1;
    res.k = offset / 9 - 1;
    return res;
  }

  std::pair<size_t, size_t> nbh_runner_decode_block(onikaInt3_t& block) const {
    size_t cell_a = cell_idx[block.x];
    IJK loc_a = grid_index_to_ijk(dims, cell_a);
    size_t cell_b = grid_ijk_to_index(dims, loc_a + convert_offset_ijk(block.y));
    return {cell_a, cell_b};
  }

  template <size_t... Is>
  ONIKA_HOST_DEVICE_FUNC inline void apply(onikaInt3_t& block,
                                           tuple_helper::index<Is...> indexes) const {
    assert(block.z == 1);
    auto [cell_a, cell_b] = nbh_runner_decode_block(block);
    func(cell_a, cell_b, std::get<Is>(params)...); 
  }

  /**
   * @brief Functor operator to apply the kernel function to each element in the array.
   * @param i Index of the element in the array.
   */
  ONIKA_HOST_DEVICE_FUNC inline void operator()(onikaInt3_t block) const {
    apply(block, tuple_helper::gen_seq<sizeof...(Args)>{});
  }
};

// Default Trait definition
template <typename Func>
struct NeighborRunnerFunctorTraits {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = false;
};
}  // namespace exaDEM



namespace onika {
namespace parallel {

template <typename Func, typename... Args>
struct ParallelForFunctorTraits<exaDEM::NeighborRunner<Func, Args...>> {
  static inline constexpr bool RequiresBlockSynchronousCall = exaDEM::NeighborRunnerFunctorTraits<Func>::RequiresBlockSynchronousCall;
  static inline constexpr bool CudaCompatible = exaDEM::NeighborRunnerFunctorTraits<Func>::CudaCompatible;
};
}  // namespace parallel
}  // namespace onika

