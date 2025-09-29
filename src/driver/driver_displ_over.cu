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

  /**
   * @brief Integer accumulator.
   */
  struct Accumulator
  {
    private:
      VectorT<int> data; ///< Storage buffer

    public:

      Accumulator() { reset(); } 
      /// Reset accumulator to zero
      void reset()  
      {
        data.resize(1);
        data[0] = 0;
      }
      /// Get raw pointer to data
      int* __restrict__ get_ptr() { return data.data(); }
      /// Get current value
      const int get() { return data[0]; }
  };

  /**
   * @brief Driver displacement overlap handler.
   */
  template<class ParallelExecutionContextFunctor>
    struct driver_displ_over
    {
      ParallelExecutionContextFunctor m_parallel_execution_context; ///< Parallel execution context
      const double r2;                                             ///< Squared distance
      MPI_Comm& comm;                                              ///< MPI communicator
      Drivers& bcpd;                                               ///< Backup drivers
      const size_t bd_idx;                                         ///< Backup driver index
      Accumulator storage;                                         ///< Result accumulator

      inline int operator()(exaDEM::Cylinder &a) { return 0; } // WARNING should not move

      inline int operator()(exaDEM::Surface &a) 
      {
        exaDEM::Surface &b = bcpd.get_typed_driver<exaDEM::Surface>(bd_idx);
        Vec3d d = a.center_proj - b.center_proj;

        if( b.motion_type == PENDULUM_MOTION )
        {
          // Utility function: compute the intersection between a line and a plane
          auto intersect_line_plane = [] (
              const Vec3d& plane_point,    // A point lying on the plane (pendulum anchor)
              const Vec3d& plane_normal,   // Plane normal vector
              const Vec3d& line_point,     // Starting point of the line (initial pendulum position)
              const Vec3d& line_dir        // Direction vector of the line
              ) -> Vec3d
          {
            // Ensure the line is not parallel to the plane
            assert(std::abs(exanb::dot(plane_normal, line_dir)) >= 1e-12);
            // Vector from line_point to plane_point
            Vec3d delta = plane_point - line_point;
            // Scalar parameter t of the intersection
            double t = exanb::dot(delta, plane_normal) / exanb::dot(line_dir, plane_normal);
            // Intersection point
            return line_point + t * line_dir;
          };

          // Utility function: check if a point lies on a given plane
          auto is_point_on_plane = [] (
              const Vec3d& point,          // Point to check
              const Vec3d& plane_normal,   // Plane normal
              const double plane_offset    // Plane offset (ax + by + cz = offset)
              ) -> bool
          {
            double projected_value = exanb::dot(point, plane_normal);
            return std::abs(projected_value - plane_offset) < 1e-14;
          };

          // Project the initial pendulum positions onto their respective planes
          Vec3d proj_a = intersect_line_plane(a.pendulum_anchor_point, a.normal, a.pendulum_initial_position, a.pendulum_direction());
          Vec3d proj_b = intersect_line_plane(b.pendulum_anchor_point, b.normal, b.pendulum_initial_position, b.pendulum_direction());

          // Sanity check
          if (!is_point_on_plane(proj_a, a.normal, a.offset) ||
              !is_point_on_plane(proj_b, b.normal, b.offset))
          {
            color_log::error("driver_displ_over::surface", "proj_a or proj_b are invalid");
          }

          d = proj_b - proj_a; 
        }

        if (exanb::dot(d, d) >= r2)
          return 1;
        else
          return 0;
      }

      inline int operator()(exaDEM::Ball &a) 
      {
        exaDEM::Ball &b = bcpd.get_typed_driver<exaDEM::Ball>(bd_idx);
        Vec3d d = a.center - b.center;
        if( b.is_compressive() )
        {
          if( exanb::dot(d, d) >= 1e-12 )
          {
            color_log::error("driver_displ_over", "Ball with compressive motion type should not move");
          }
          double r_diff = a.radius - b.radius;
          if( r_diff * r_diff >= r2 )
          {
            return 1;
          } 
          else
          {
            return 0;
          }
        }

        if (exanb::dot(d, d) >= r2)
          return 1;
        else
          return 0;

      }

      inline int operator()(exaDEM::Stl_mesh &a)
      {
        exaDEM::Stl_mesh &b = bcpd.get_typed_driver<exaDEM::Stl_mesh>(bd_idx);

        if ((a.center == b.center) && (a.quat == b.quat))
          return 0;

#ifdef ONIKA_CUDA_VERSION
        storage.reset();
        size_t size = a.shp.get_number_of_vertices();
        const Vec3d * __restrict__ ptr_shp_vertices = onika::cuda::vector_data(a.shp.m_vertices);
        STLVertexDisplacementFunctor SVDFunc = {r2, ptr_shp_vertices, a.center, a.quat, b.center, b.quat};
        ReduceMaxSTLVertexDisplacementFunctor func = {SVDFunc, storage.get_ptr()};

        ParallelForOptions opts;
        opts.omp_scheduling = OMP_SCHED_STATIC;
        parallel_for(size, func, m_parallel_execution_context(), opts);   
        return storage.get();
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

      // get backup
      Drivers &bcpd = *backup_drvs;
      size_t bcpd_size = bcpd.get_size();
      assert(bcpd_size == size);

      auto pec_func = [this]() { return this->parallel_execution_context(); };
      for (size_t i = 0; i < bcpd_size && local_drivers_displ == 0 ; i++)
      {
        driver_displ_over<decltype(pec_func)> func = {pec_func, max_dist2, comm, bcpd, i };
        local_drivers_displ += drvs.apply( i , func );
      }

      MPI_Allreduce(&local_drivers_displ, &(global_drivers_displ), 1, MPI_INT, MPI_SUM, comm);
      *result = (global_drivers_displ > 0);
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(driver_displ_over) { OperatorNodeFactory::instance()->register_factory("driver_displ_over", make_simple_operator<DriverDisplacementOver>); }

} // namespace exaDEM
