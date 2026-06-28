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

#include <mpi.h>

#include <exaDEM/classifier/interaction_wrapper.hpp>
#include <exaDEM/interaction/placeholder_interaction.hpp>

namespace exaDEM {

/** @brief A struct representing an interface composed of multiple interactions */
struct Interface {
  // Important assumption: interactions are stored contiguously
  size_t loc_;   // Location in the classifier
  size_t size_;  // Number of interactions composed this interface
};

// This struct is used to build interfaces on the CPU.
// The data is then copied to the InterfaceManager that is used on the GPU.
/** @brief Struct used to build interfaces on the CPU */
struct InterfaceBuildManager {
  std::vector<Interface> data_;  // list of interfaces.
};

/** @brief Struct used to handle interfaces on the CPU/GPU */
struct InterfaceManager {
  template <typename T>
  using vector_t = onika::memory::CudaMMVector<T>;
  vector_t<Interface> data_;  // list of interfaces. Each interface is defined by its location in the classifier and its
                              // size (number of interactions that compose the interface)
  vector_t<uint8_t> break_interface_;  // list of booleans that indicate if the interface is broken or not. 1 if the
                                       // interface is broken, 0 otherwise.

  /** @brief Resize the interface manager
   * @param new_size The new size of the interface manager.
   */
  void resize(size_t new_size) {
    assert(new_size < 1e8);  // 1e8 is an arbitrary value to avoid resizing the interface manager with a too large size.
                             // This can be a sign of a bug.
    data_.clear();
    data_.resize(new_size);
    break_interface_.resize(new_size);
    std::fill(break_interface_.begin(), break_interface_.end(), false);
  }

  /** @brief Get the size of the interface manager
   * @return The size of the interface manager.
   */
  size_t size() { return data_.size(); }
};

/** @brief Check the consistency of the interfaces
 * @param interfaces The interface build manager
 * @param interactions The interaction classifier
 * @return True if the interfaces are consistent, false otherwise.
 * warning: this function is not efficient, it is only used for debugging purposes.
 */
inline bool check_interface_consistency(InterfaceBuildManager& interfaces,
                                        ClassifierContainer<InteractionType::InnerBond>& interactions) {
  int res = 0;  // number of inconsistent interfaces;

  // disable openmp, this function is only used for debugging purposes.
  // #pragma omp parallel for reduction(+ : res)
  for (size_t i = 0; i < interfaces.data_.size(); i++) {
    auto [loc, size] = interfaces.data_[i];

    uint64_t id_i = interactions.particle_id_i(loc);
    uint64_t id_j = interactions.particle_id_j(loc);

    assert(loc + size <= interactions.size());
    for (size_t next = loc + 1; next < loc + size; next++) {
      if (id_i != interactions.particle_id_i(next) || id_j != interactions.particle_id_j(next)) {
        res += 1;
      }
    }
  }

  if (res == 0) {
    return true;
  }

  color_log::warning("check_interface_consistency",
                     std::to_string(res) + " interface are not defined correctly.\n" +
                         "The interactions that compose the interface are not all defined between the same particles.");
  assert(res == 0);
  return false;
}
}  // namespace exaDEM
