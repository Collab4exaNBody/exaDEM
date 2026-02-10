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

#include <onika/parallel/parallel_for.h>

namespace exaDEM {
using namespace onika::parallel;
using namespace exanb;
template <typename T>
using VectorT = onika::memory::CudaMMVector<T>;

struct reduce_thread_block_t {};
struct reduce_thread_local_t {};
struct reduce_global_t {};
struct RShapeDriverDisplacementFunctor {
  const double m_threshold_sqr = 0.0;
  const Vec3d* vertices;
  const Vec3d c_a;
  const Quaternion q_a;
  const Vec3d c_b;
  const Quaternion q_b;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int& count_over_dist2, uint64_t i, reduce_thread_local_t = {}) const {
    const Vec3d v_a = c_a + q_a * vertices[i];
    const Vec3d v_b = c_b + q_b * vertices[i];
    const Vec3d dv = v_a - v_b;
    if (exanb::dot(dv, dv) >= m_threshold_sqr) {
      ++count_over_dist2;
    }
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int& count_over_dist2, int value, reduce_thread_block_t) const {
    ONIKA_CU_ATOMIC_ADD(count_over_dist2, value);
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(int& count_over_dist2, int value, reduce_global_t) const {
    ONIKA_CU_ATOMIC_ADD(count_over_dist2, value);
  }
};

struct ReduceMaxRShapeDriverDisplacementFunctor {
  const RShapeDriverDisplacementFunctor m_func;
  int* m_reduced_val;
  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t i) const {
    int local_val = int();
    m_func(local_val, i, reduce_thread_local_t{});

    ONIKA_CU_BLOCK_SHARED onika::cuda::UnitializedPlaceHolder<int> team_val_place_holder;
    int& team_val = team_val_place_holder.get_ref();

    if (ONIKA_CU_THREAD_IDX == 0) {
      team_val = local_val;
    }
    ONIKA_CU_BLOCK_SYNC();

    if (ONIKA_CU_THREAD_IDX != 0 && local_val > 0) {
      m_func(team_val, local_val, reduce_thread_block_t{});
    }
    ONIKA_CU_BLOCK_SYNC();

    if (ONIKA_CU_THREAD_IDX == 0 && team_val > 0) {
      m_func(*m_reduced_val, team_val, reduce_global_t{});
    }
  }
};
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <>
struct ParallelForFunctorTraits<exaDEM::ReduceMaxRShapeDriverDisplacementFunctor> {
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika
