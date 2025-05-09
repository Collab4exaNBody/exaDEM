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
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>

#include <onika/math/basic_types.h>
#include <mpi.h>
#include <exaDEM/shapes.hpp>
#include <exaDEM/drivers.h>

namespace exaDEM
{
  using namespace exanb;
  struct WrapperSTLMeshComputeVertices
  {
    Vec3d center;
    Quaternion quat;
    Vec3d * stl_vertices;
    const Vec3d * shp_vertices;
    ONIKA_HOST_DEVICE_FUNC  
    void operator() (int idx) const
    {
      stl_vertices[idx] = center + quat * shp_vertices[idx];
    }
  };
}

namespace onika
{
  namespace parallel
  {
    template<> struct ParallelForFunctorTraits<exaDEM::WrapperSTLMeshComputeVertices>
    {
      static inline constexpr bool CudaCompatible = true;
    };
  } // namespace parallel
} // namespace onika


namespace exaDEM
{

  using namespace onika::parallel;
  class DriverVertices : public OperatorNode
  {
  
    template<class ParallelExcutionContextFuncT>
    struct driver_compute_vertices
    {
      ParallelExcutionContextFuncT m_parallel_execution_context;
      inline void operator()(exaDEM::Stl_mesh &mesh) const
      {
        const size_t size = mesh.shp.get_number_of_vertices();

        if(mesh.stationary()) 
        {
          return;
        }

        ParallelForOptions opts;
        opts.omp_scheduling = OMP_SCHED_STATIC;

        Vec3d stl_center = mesh.center;
        Quaternion stl_quat = mesh.quat;
        Vec3d* ptr_stl_vertices = onika::cuda::vector_data(mesh.vertices);
        Vec3d* ptr_shp_vertices = onika::cuda::vector_data(mesh.shp.m_vertices);
        WrapperSTLMeshComputeVertices func = {stl_center, stl_quat, ptr_stl_vertices, ptr_shp_vertices};
        parallel_for(size, func, m_parallel_execution_context(), opts);
      }
      /**
        Add another driver type here 
       */

      template<class OtherDriverType>
      inline void operator()(const OtherDriverType &) const
      {
        static_assert( get_type<OtherDriverType>() != DRIVER_TYPE::UNDEFINED );
        //lout << "WARNING: driver_compute_vertices is not defined for this driver. " << std::endl;
        //return 0;
      }
    };

    // -----------------------------------------------
    // -----------------------------------------------
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(bool, force_host , INPUT, REQUIRED, DocString{"Force computations on the host"});

    public:
    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
           This operator calculates new vertex positions. 
           If stl mesh velocity and angular velocity are equal to [0,0,0], vertices are not calculated.
        )EOF";
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute() override final
    {
      Drivers &drvs = *drivers;
      size_t size = drvs.get_size();
      set_gpu_enabled(!(*force_host));
      auto pec_func = [this]() { return this->parallel_execution_context(); } ;
      for (size_t i = 0; i <size; i++)
      {
        driver_compute_vertices<decltype(pec_func)> func = {pec_func}; // now, you do know how to make it
        drivers->apply( i , func );
      }
      // 
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(driver_vertices) { OperatorNodeFactory::instance()->register_factory("driver_vertices", make_simple_operator<DriverVertices>); }

} // namespace exaDEM
