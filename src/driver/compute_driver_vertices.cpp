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
//#pragma xstamp_cuda_enable

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <onika/parallel/parallel_execution_context.h>
#include <onika/parallel/block_parallel_for.h>
#include <onika/parallel/parallel_for.h>

#include <exanb/core/basic_types.h>
#include <mpi.h>
#include <exaDEM/shape/shapes.hpp>
#include <exaDEM/drivers.h>

namespace exaDEM
{
  using namespace exanb;
  using namespace onika::parallel;
  class DriverComputeVertices : public OperatorNode
  {

    struct WrapperSTLMeshComputeVertices
    {
      Stl_mesh* mesh;
      ONIKA_HOST_DEVICE_FUNC  
      void operator() (int idx) const
      {
        mesh->update_vertex(idx);
      }
    };

    template<typename Operator>
      struct driver_compute_vertices
      {
        Operator* op; // I don't know how to do it properly
        void operator()(exaDEM::Stl_mesh &mesh)
        {
          const size_t size = mesh.shp.get_number_of_vertices();

          if(mesh.stationary()) 
          {
            return;
          }

          ParallelForOptions opts;
          opts.omp_scheduling = OMP_SCHED_STATIC;

          WrapperSTLMeshComputeVertices func = {&mesh};
          parallel_for(size, func, op->parallel_execution_context(), opts);
        }
        /**
          Add another driver type here 
         */

        void operator()(auto &&a) 
        {
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
      set_gpu_enabled(*force_host);
      for (size_t i = 0; i <size; i++)
      {
        driver_compute_vertices<DriverComputeVertices> func = {this}; // I don't know how to do it properly
        auto & drv = drvs.data(i);
        std::visit(func, drv);
      }
      // 
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("compute_driver_vertices", make_simple_operator<DriverComputeVertices>); }

} // namespace exaDEM
