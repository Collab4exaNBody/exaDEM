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

#include <exaDEM/interaction/placeholder_interaction.hpp>
#include <exaDEM/classifier/interaction_wrapper.hpp>
#include <mpi.h>

namespace exaDEM {
template <typename T> using vector_t = onika::memory::CudaMMVector<T>;

struct Interface {
  // Important assumption: interactions are stored contiguously
  size_t loc; // Location in the classifier
  size_t size; // Number of interactions composed this interface
};

// Thread Local Storage
struct InterfaceBuildManager {
  std::vector<Interface> data;
};

struct InterfaceManager {
  vector_t<Interface> data;
  vector_t<uint8_t> break_interface; // warning on gpu 
  void resize(size_t new_size) {
    assert(new_size < 1e7);
    data.clear();
    data.resize(new_size);
    break_interface.resize(new_size);
    std::fill(break_interface.begin(), break_interface.end(), false);
  }
  size_t size() {
    return data.size();
  }
};


inline bool check_interface_consistency(
    InterfaceBuildManager& interfaces, 
    ClassifierContainer<InteractionType::InnerBond>& interactions) {
  int res = 0;

#pragma omp parallel for reduction(+: res)
  for(size_t i=0 ; i<interfaces.data.size() ; i++) {
    auto [loc, size] = interfaces.data[i];

    uint64_t id_i = interactions.particle_id_i(loc);
    uint64_t id_j = interactions.particle_id_j(loc);

    assert(loc+size <= interactions.size());
    for(size_t next=loc+1; next<loc+size ; next++) {
      if(id_i != interactions.particle_id_i(next) 
         || id_j != interactions.particle_id_j(next)) {
        res += 1;
      }
    }
  }

  if(res == 0) {
    return true;
  }
  color_log::warning("check_interface_consistency", 
                     std::to_string(res) + " interface are not defined correctly.\n" 
                     + "The interactions that compose the interface are not all defined between the same particles.");
  assert(res == 0);
  return false;
}

// CPU only
inline void rebuild_interface_Manager(
    InterfaceBuildManager& interfaces, 
    ClassifierContainer<InteractionType::InnerBond>& interactions) {
  interfaces.data.clear();
  size_t n_interactions = interactions.size();

  size_t loc = 0;
  while( loc<n_interactions ) {
    // Here, we do not build interfaces that are managed by another MPI process (partner).
    if( interactions.ghost[loc] == InteractionPair::PartnerGhost ) { loc++ ; continue ; }

    // Information about the particles managed by the first interaction is retrieved. 
    // The interactions that compose an interface are stored contiguously.
    uint64_t idloci = interactions.particle_id_i(loc);
    uint64_t idlocj = interactions.particle_id_j(loc);
    size_t n = 1;
    uint64_t idni = interactions.particle_id_i(loc + n);
    uint64_t idnj = interactions.particle_id_j(loc + n);
    n++;

    // We locate the range of all interactions that make up the interface.
    while(loc + n  < n_interactions
          && idloci == idni 
          && idlocj == idnj) {
      idni = interactions.particle_id_i(loc + n);
      idnj = interactions.particle_id_j(loc + n);
      n++;
    }
    if( loc + n != n_interactions ) {
      n--; // exclude the last element that failed the test 
    }
    Interface interface = {loc, n};
    interfaces.data.push_back(interface);
    loc += n;
  }
  assert(loc == n_interactions);
}
}
