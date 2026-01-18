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

#include <exanb/compute/compute_cell_particles.h>
#include <exanb/compute/reduce_cell_particles.h>

#include <tuple>

namespace exaDEM {
enum REORDER {
  NONE,
  MORTON,
  HILBERT
};

struct Traversal {
  static constexpr size_t EXADEM_BLOCK_UNIT = 32;
  template <typename T>
  using VectorT = onika::memory::CudaMMVector<T>;
  VectorT<size_t> m_data;
  size_t m_max_block_size = 0;  // used for gpu

  bool iterator = false;

  size_t* __restrict__ data() {
    return onika::cuda::vector_data(m_data);
  }

  size_t size() {
    return onika::cuda::vector_size(m_data);
  }

  size_t get_max_block_size() {
    return (m_max_block_size / EXADEM_BLOCK_UNIT + 1) * EXADEM_BLOCK_UNIT;
  }

  exanb::ComputeCellParticlesOptions get_compute_cell_particles_options() {
    exanb::ComputeCellParticlesOptions ccpo;
    ccpo.m_max_block_size = this->get_max_block_size();
    ccpo.m_num_cell_indices = this->size();
    ccpo.m_cell_indices = this->data();
    return ccpo;
  }

  exanb::ReduceCellParticlesOptions get_reduce_cell_particles_options() {
    exanb::ReduceCellParticlesOptions rcpo;
    rcpo.m_max_block_size = this->get_max_block_size();
    rcpo.m_num_cell_indices = this->size();
    rcpo.m_cell_indices = this->data();
    return rcpo;
  }

  std::tuple<size_t*, size_t> info() {
    const size_t s = this->size();
    if (s == 0) {
      return {nullptr, 0};
    } else {
      return {this->data(), this->size()};
    }
  }

  void reorder(const REORDER reorder_type) {
    if (reorder_type == NONE) {
      /** do nothing */
    } else if (reorder_type == MORTON) {
      /** Not implemented */
      onika::lout << "reoder with MORTON indexes is not implemented" << std::endl;
    } else if (reorder_type == HILBERT) {
      /** Not implemented */
      onika::lout << "reoder with HILBER indexes is not implemented" << std::endl;
    }
  }
};
}  // namespace exaDEM
