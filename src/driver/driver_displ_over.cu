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
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <onika/math/basic_types.h>
#include <exaDEM/drivers.h>
#include <exaDEM/reduce_stl_mesh.hpp>

namespace exaDEM
{
  using namespace exanb;
  template <typename T> using VectorT = onika::memory::CudaMMVector<T>;

  template<typename Operator> struct driver_displ_over
  {
    Operator* op;
    double r2; // square distance
    MPI_Comm &comm;
    int * storage;
    /* Ignored */   
    int operator()(exaDEM::UndefinedDriver &a, exaDEM::UndefinedDriver &b) { return 0; }
    int operator()(exaDEM::Cylinder &a, exaDEM::Cylinder &b) { return 0; } // WARNING should not move

    /* Active */
    int operator()(exaDEM::Surface &a, exaDEM::Surface &b)
    {
      Vec3d d = a.center_proj - b.center_proj;
      if (exanb::dot(d, d) >= r2)
        return 1;
      else
        return 0;
    }

    int operator()(exaDEM::Ball &a, exaDEM::Ball &b)
    {
      Vec3d d = a.center - b.center;
      if (exanb::dot(d, d) >= r2)
        return 1;
      else
        return 0;
    }

    int operator()(exaDEM::Stl_mesh &a, exaDEM::Stl_mesh &b)
    {
      if ((a.center == b.center) && (a.quat == b.quat))
        return 0;

#ifdef ONIKA_CUDA_VERSION
      size_t size = a.shp.get_number_of_vertices();
      const Vec3d * __restrict__ ptr_shp_vertices = onika::cuda::vector_data(a.shp.m_vertices);
      STLVertexDisplacementFunctor SVDFunc = {r2, ptr_shp_vertices, a.center, a.quat, b.center, b.quat};
      ReduceMaxSTLVertexDisplacementFunctor func = {SVDFunc, storage};

      ParallelForOptions opts;
      opts.omp_scheduling = OMP_SCHED_STATIC;
      parallel_for(size, func, op->parallel_execution_context(), opts);   
      return *storage;
#else
      int sum = 0;
      // check each vertex
      auto check_dist = [](Vec3d &v1, Vec3d &v2, double r2) -> bool
      {
        Vec3d d = v1 - v2;
        if (exanb::dot(d, d) >= r2)
          return true;
        return false;
      };
      auto &shp_a = a.shp;
      size_t size = shp_a.get_number_of_vertices();

      size_t start(0), end(size);
      // optimization for big shapes
      if (size > 100)
      {
        int mpi_rank, mpi_size;
        MPI_Comm_rank(comm, &mpi_rank);
        MPI_Comm_size(comm, &mpi_size);
        start = (size * mpi_rank) / mpi_size;
        end = (size * (mpi_rank + 1)) / mpi_size;
      }

#     pragma omp parallel for reduction(+: sum)
      for (size_t i = start; i < end; i++)
      {
        Vec3d va = shp_a.get_vertex(i, a.center, a.quat);
        Vec3d vb = shp_a.get_vertex(i, b.center, b.quat);
        if (check_dist(va, vb, r2))
          sum++;
      }
      return sum;
#endif
    }

    int operator()(auto &&a, auto &&b)
    {
      lout << "WARNING: driver_displ_over is not defined for this driver. " << std::endl;
      return 0;
    }
  };

  struct Accumulator
  {
    VectorT<int> data;
  };

  class DriverDisplacementOver : public OperatorNode
  {
    // -----------------------------------------------
    // -----------------------------------------------
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(double, threshold, INPUT, 0.0);
    ADD_SLOT(bool, result, OUTPUT);
    ADD_SLOT(Drivers, drivers, INPUT, REQUIRED, DocString{"List of Drivers"});
    ADD_SLOT(Drivers, backup_drvs, INPUT, REQUIRED, DocString{"List of backup Drivers"});
    ADD_SLOT(Accumulator, displ_driver_count, PRIVATE);
    public:
    // -----------------------------------------------
    // -----------------------------------------------
    inline std::string documentation() const override final
    {
      return R"EOF(
compute the distance between each particle in grid input and it's backup position in backup_dem input.
sets result output to true if at least one particle has moved further than threshold.
)EOF";
    }

    // -----------------------------------------------
    // -----------------------------------------------
    inline void execute() override final
    {
      MPI_Comm comm = *mpi;

      auto& ddc = *displ_driver_count;

      const double max_dist = *threshold;
      const double max_dist2 = max_dist * max_dist;

      int local_drivers_displ(0), global_drivers_displ(0);

      Drivers &drvs = *drivers;
      size_t size = drvs.get_size();
      if (size == 0)
      {
        *result = false;
        return;
      }

      ddc.data.resize(1);
      int * pddc = onika::cuda::vector_data(ddc.data);
      // reset value (accumulator)
      *pddc = 0;
      // get backup
      Drivers &bcpd = *backup_drvs;
      size_t bcpd_size = bcpd.get_size();
      assert(bcpd_size == size);
      driver_displ_over<DriverDisplacementOver> func = {this, max_dist2, comm, pddc};

      for (size_t i = 0; i < bcpd_size && local_drivers_displ == 0 ; i++)
      {
        auto &drv1 = drvs.data(i);
        auto &drv2 = bcpd.data(i);
        local_drivers_displ += std::visit(func, drv1, drv2);
      }

      MPI_Allreduce(&local_drivers_displ, &(global_drivers_displ), 1, MPI_INT, MPI_SUM, comm);
      *result = (global_drivers_displ > 0);
    }
  };

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("driver_displ_over", make_simple_operator<DriverDisplacementOver>); }

} // namespace exaDEM
