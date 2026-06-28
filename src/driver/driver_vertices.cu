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

#include <mpi.h>
#include <onika/math/basic_types.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/parallel_for.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/drivers.hpp>
#include <exaDEM/shapes.hpp>

namespace exaDEM {
using namespace exanb;
struct WrapperRShapeDriverComputeVertices {
  Vec3d center_;
  Quaternion quat_;
  Vec3d* rshape_vertices_;
  const Vec3d* shp_vertices_;
  ONIKA_HOST_DEVICE_FUNC
  void operator()(int idx) const { rshape_vertices_[idx] = center_ + quat_ * shp_vertices_[idx]; }
};
}  // namespace exaDEM

namespace onika {
namespace parallel {
template <>
struct ParallelForFunctorTraits<exaDEM::WrapperRShapeDriverComputeVertices> {
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace parallel
}  // namespace onika

namespace exaDEM {

using namespace onika::parallel;
class DriverVertices : public OperatorNode {
  template <class ParallelExcutionContextFuncT>
  struct DriverComputeVerticesFunc {
    ParallelExcutionContextFuncT parallel_execution_context_;
    inline void operator()(exaDEM::RShapeDriver& mesh) const {
      const size_t size = mesh.shp_.get_number_of_vertices();
      if (mesh.stationary()) {
        return;
      }

      ParallelForOptions opts;
      opts.omp_scheduling = OMP_SCHED_STATIC;

      Vec3d rshape_center = mesh.fields_.center_;
      Quaternion rshape_quat = mesh.fields_.quat_;
      Vec3d* ptr_rshape_vertices = onika::cuda::vector_data(mesh.vertices_);
      Vec3d* ptr_shp_vertices = onika::cuda::vector_data(mesh.shp_.vertices_);
      WrapperRShapeDriverComputeVertices func = {rshape_center, rshape_quat, ptr_rshape_vertices, ptr_shp_vertices};
      parallel_for(size, func, parallel_execution_context_(), opts);
    }

    template <class OtherDriverType>
    inline void operator()(const OtherDriverType&) const {
      static_assert(get_type<OtherDriverType>() != DRIVER_TYPE::UNDEFINED);
    }
  };

  // -----------------------------------------------
  // -----------------------------------------------
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers"});
  ADD_SLOT(bool, force_host, INPUT, REQUIRED, DocString{"Force computations on the host"});

 public:
  // -----------------------------------------------
  // -----------------------------------------------
  inline std::string documentation() const final {
    return R"EOF(
           This operator calculates new vertex positions. 
           If rshape mesh velocity and angular velocity are equal to [0,0,0], vertices are not calculated.
        )EOF";
  }

  // -----------------------------------------------
  // -----------------------------------------------
  inline void execute() final {
    Drivers& drvs = *drivers;
    size_t size = drvs.get_size();
    set_gpu_enabled(!(*force_host));
    auto pec_func = [this]() { return this->parallel_execution_context(); };
    for (size_t i = 0; i < size; i++) {
      DriverComputeVerticesFunc<decltype(pec_func)> func = {pec_func};
      drivers->apply(i, func);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(driver_vertices) {
  OperatorNodeFactory::instance()->register_factory("driver_vertices", make_simple_operator<DriverVertices>);
}
}  // namespace exaDEM
