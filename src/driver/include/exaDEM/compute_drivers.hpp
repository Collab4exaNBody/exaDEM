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

#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_execution_context.h>

#include <exaDEM/drivers.hpp>

namespace exaDEM {

template <typename FuncT>
struct ComputeDriversFunc {
  Cylinder* __restrict__ cylinders_;
  size_t cylinders_size_;
  Surface* __restrict__ surfaces_;
  size_t surfaces_size_;
  Ball* __restrict__ balls_;
  size_t balls_size_;
  RShapeDriver* __restrict__ rshapes_;
  size_t rshapes_size_;
  FuncT func_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(long block_idx) const {
    if (block_idx == DRIVER_TYPE::CYLINDER) {
      ONIKA_CU_BLOCK_SIMD_FOR(size_t, i, 0, cylinders_size_) { func_(cylinders_[i]); }
    } else if (block_idx == DRIVER_TYPE::SURFACE) {
      ONIKA_CU_BLOCK_SIMD_FOR(size_t, i, 0, surfaces_size_) { func_(surfaces_[i]); }
    } else if (block_idx == DRIVER_TYPE::BALL) {
      ONIKA_CU_BLOCK_SIMD_FOR(size_t, i, 0, balls_size_) { func_(balls_[i]); }
    } else if (block_idx == DRIVER_TYPE::RSHAPE) {
      ONIKA_CU_BLOCK_SIMD_FOR(size_t, i, 0, rshapes_size_) { func_(rshapes_[i]); }
    }
  }
};

template <class FuncT>
inline void compute_drivers(Drivers& drivers, FuncT& func, onika::parallel::ParallelExecutionContext* exec_ctx) {
  auto& cylinders = drivers.data_.get_nth<DRIVER_TYPE::CYLINDER>();
  auto& surfaces = drivers.data_.get_nth<DRIVER_TYPE::SURFACE>();
  auto& balls = drivers.data_.get_nth<DRIVER_TYPE::BALL>();
  auto& rshapes = drivers.data_.get_nth<DRIVER_TYPE::RSHAPE>();
  ComputeDriversFunc<FuncT> wrapper{cylinders.data(), cylinders.size(), surfaces.data(),
                                    surfaces.size(),  balls.data(),     balls.size(),
                                    rshapes.data(),   rshapes.size(),   func};
  onika::parallel::block_parallel_for(DRIVER_TYPE_SIZE, wrapper, exec_ctx, onika::parallel::BlockParallelForOptions{});
}
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <typename FuncT>
struct BlockParallelForFunctorTraits<exaDEM::ComputeDriversFunc<FuncT>> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
