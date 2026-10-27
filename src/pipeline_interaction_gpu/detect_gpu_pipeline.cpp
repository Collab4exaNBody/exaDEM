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

#include <onika/cuda/cuda_context.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <onika/log.h>

#include <mpi.h>

namespace exaDEM {
using namespace exanb;

class DetectGPUPipeline : public OperatorNode {
  ADD_SLOT(bool, disable_gpu_pipeline, INPUT, false,
           DocString{"Global flag: if true, the GPU pipeline is never used, even if a GPU is available."});
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(bool, use_gpu_pipeline, OUTPUT, DocString{"true if the GPU pipeline (interaction lists) must be used."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator sets the boolean 'use_gpu_pipeline'. It is true only if the code has been compiled with CUDA,
        a GPU device is available for every MPI process, and the global flag 'disable_gpu_pipeline' is false.
        It is meant to be used in 'condition:' fields of msp blocks.

        YAML example [no option]:

          - detect_gpu_pipeline
       )EOF";
  }

  inline void execute() final {
    const bool local_gpu_available = (global_cuda_ctx() != nullptr) && global_cuda_ctx()->has_devices();
    // All MPI processes must use the same pipeline (e.g. with mpi_even_only, only some processes own a GPU).
    int local_flag = (local_gpu_available && !(*disable_gpu_pipeline)) ? 1 : 0;
    int global_flag = 0;
    MPI_Allreduce(&local_flag, &global_flag, 1, MPI_INT, MPI_MIN, *mpi);
    *use_gpu_pipeline = (global_flag == 1);
    if (*disable_gpu_pipeline) {
      lout << "GPU pipeline disabled by the global flag disable_gpu_pipeline" << std::endl;
    } else {
      lout << "GPU pipeline: "
           << (*use_gpu_pipeline ? "enabled" : "disabled (no GPU device on at least one MPI process)") << std::endl;
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(detect_gpu_pipeline) {
  OperatorNodeFactory::instance()->register_factory("detect_gpu_pipeline", make_simple_operator<DetectGPUPipeline>);
}
}  // namespace exaDEM
