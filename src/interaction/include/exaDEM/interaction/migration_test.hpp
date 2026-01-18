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

#include <cassert>
#include <exanb/extra_storage/extra_storage_info.hpp>

namespace exaDEM {

namespace interaction_test {
using UIntType = uint64_t;
using InfoType = ExtraStorageInfo;
inline bool check_extra_interaction_storage_consistency(int n_particles, InfoType* info_ptr,
                                                        PlaceholderInteraction* data_ptr) {
  for (int p = 0; p < n_particles; p++) {
    auto [offset, size, id] = info_ptr[p];
    for (size_t i = offset; i < offset + size; i++) {
      auto& item = data_ptr[i];
      if (item.pair.pi.id != id && item.pair.pj.id != id) {
        std::cout << "info says particle id = " << id << " and the interaction is between the particle id "
                  << item.pair.pi.id << " and the particle id " << item.pair.pj.id << std::endl;
        return false;
      }
    }
  }

  return true;
}
}  // namespace interaction_test
}  // namespace exaDEM
