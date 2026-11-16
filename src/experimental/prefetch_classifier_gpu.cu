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

#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/experimental/polyhedron/nbh_gpu/nbh_cell_data.hpp>

namespace exaDEM {
using namespace exanb;

/// @brief Prefetches one classifier field vector (bk_/fm_/contact state entry) to `device`.
template <typename VecT>
inline void prefetch_classifier_vector(VecT& vec, int device, onikaStream_t& st) {
  if (!vec.empty()) {
    ONIKA_PREFETCH(vec.data(), vec.size() * sizeof(typename VecT::value_type), device, st);
  }
}

/// @brief Applies prefetch_classifier_vector to every field vector of a ClassifierContainer<IT>.
struct PrefetchClassifierFieldFunctor {
  int device_;
  onikaStream_t stream_;
  template <typename VecT>
  inline void operator()(VecT& vec) {
    prefetch_classifier_vector(vec, device_, stream_);
  }
};

/// @brief Prefetches every buffer (particle-particle, particle-driver, inner-bond, contact
/// state) of a Classifier to `device`. `device` may be cudaCpuDeviceId to prefetch back to host.
inline void prefetch_classifier(Classifier& classifier, int device, onikaStream_t& st) {
  PrefetchClassifierFieldFunctor func{device, st};

  for (auto& container : classifier.particles_) container.apply_on_fields(func);
  for (auto& container : classifier.drivers_) container.apply_on_fields(func);
  for (auto& container : classifier.innerbonds_) container.apply_on_fields(func);

  for (auto& state : classifier.contact_state_) {
    prefetch_classifier_vector(state.dn_, device, st);
    prefetch_classifier_vector(state.cp_, device, st);
    prefetch_classifier_vector(state.fn_, device, st);
    prefetch_classifier_vector(state.ft_, device, st);
  }
}

class PrefetchClassifierGPU : public OperatorNode {
  ADD_SLOT(Classifier, ic, INPUT, REQUIRED, DocString{"Interaction lists classified according to their types"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Prefetches all classifier interaction buffers (particle-particle, particle-driver,
        inner-bond, and contact state) to the GPU.

        YAML example:

          - prefetch_classifier_gpu
       )EOF";
  }

  inline void execute() final {
#ifndef ONIKA_CUDA_VERSION
    color_log::error("prefetch_classifier_gpu", "This operator only works on GPU.");
#else
    onikaStream_t st = global_cuda_ctx()->getThreadStream(0);
    prefetch_classifier(*ic, 0, st);
    ONIKA_CU_STREAM_SYNCHRONIZE(st);
#endif
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(prefetch_classifier_gpu) {
  OperatorNodeFactory::instance()->register_factory("prefetch_classifier_gpu",
                                                    make_simple_operator<PrefetchClassifierGPU>);
}
}  // namespace exaDEM
