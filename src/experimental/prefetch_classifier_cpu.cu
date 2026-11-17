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
#include <exaDEM/experimental/prefetch_classifier.hpp>

namespace exaDEM {
using namespace exanb;

class PrefetchClassifierCPU : public OperatorNode {
  ADD_SLOT(Classifier, ic, INPUT, REQUIRED, DocString{"Interaction lists classified according to their types"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Prefetches all classifier interaction buffers (particle-particle, particle-driver,
        inner-bond, and contact state) back to the CPU. Useful before a CPU-only operator
        (e.g. unclassify_interactions) reads classifier data that was last written on GPU, to
        avoid paying for on-demand page migration one page at a time during that operator.

        YAML example:

          - prefetch_classifier_cpu
       )EOF";
  }

  inline void execute() final {
#ifndef ONIKA_CUDA_VERSION
    color_log::error("prefetch_classifier_cpu", "This operator only works on GPU.");
#else
    onikaStream_t st = global_cuda_ctx()->getThreadStream(0);
    prefetch_classifier(*ic, cudaCpuDeviceId, st);
    ONIKA_CU_STREAM_SYNCHRONIZE(st);
#endif
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(prefetch_classifier_cpu) {
  OperatorNodeFactory::instance()->register_factory("prefetch_classifier_cpu",
                                                    make_simple_operator<PrefetchClassifierCPU>);
}
}  // namespace exaDEM
