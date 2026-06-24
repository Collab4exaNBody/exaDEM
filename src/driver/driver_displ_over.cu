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
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/drivers.hpp>
#include <exaDEM/reduce_rshape_driver.hpp>

namespace exaDEM {
using namespace onika::scg;
template <typename T>
using VectorT = onika::memory::CudaMMVector<T>;

/**
 * @brief Integer accumulator.
 */
struct Accumulator {
 private:
  VectorT<int> data_;  ///< Storage buffer

 public:
  Accumulator() { reset(); }

  /// Reset accumulator to zero
  void reset() {
    data_.resize(1);
    data_[0] = 0;
  }

  /// Get raw pointer to data
  int* __restrict__ get_ptr() { return data_.data(); }

  /// Get current value
  int get() { return data_[0]; }
};

/**
 * @brief Driver displacement overlap handler.
 */
template <class ParallelExecutionContextFunctor>
struct DriverDisplOver {
  ParallelExecutionContextFunctor parallel_execution_context_;  ///< Parallel execution context
  const double r2_;                                               ///< Squared distance
  MPI_Comm& comm_;                                                ///< MPI communicator
  Drivers& bcpd_;                                                 ///< Backup drivers
  const size_t bd_idx_;                                           ///< Backup driver index
  Driver_params& motion_;                                         ///< Driver Motion reference
  Accumulator storage_;                                           ///< Result accumulator

  inline int operator()(exaDEM::Cylinder& a) {
    return 0;  // WARNING should not move
  }

  inline int operator()(exaDEM::Surface& a) {
    exaDEM::Surface& b = bcpd_.get_typed_driver<exaDEM::Surface>(bd_idx_);
    exanb::Vec3d d = a.fields_.center_proj_ - b.fields_.center_proj_;

    if (b.motion_type_ == PENDULUM_MOTION) {
      // Utility function: compute the intersection between a line and a plane
      auto intersect_line_plane = [](const exanb::Vec3d& plane_point,   // A point lying on the plane (pendulum anchor)
                                     const exanb::Vec3d& plane_normal,  // Plane normal vector
                                     const exanb::Vec3d& line_point,    // Starting point of the line (initial
                                                                        // pendulum position)
                                     const exanb::Vec3d& line_dir       // Direction vector of the line
                                     ) -> exanb::Vec3d {
        // Ensure the line is not parallel to the plane
        assert(std::abs(exanb::dot(plane_normal, line_dir)) >= 1e-12);
        // Vector from line_point to plane_point
        exanb::Vec3d delta = plane_point - line_point;
        // Scalar parameter t of the intersection
        double t = exanb::dot(delta, plane_normal) / exanb::dot(line_dir, plane_normal);
        // Intersection point
        return line_point + t * line_dir;
      };

      // Project the initial pendulum positions onto their respective planes
      exanb::Vec3d proj_a = intersect_line_plane(motion_.pendulum_anchor_point_, a.fields_.normal_,
                                                 motion_.pendulum_initial_position_, motion_.pendulum_direction());
      exanb::Vec3d proj_b = intersect_line_plane(motion_.pendulum_anchor_point_, b.fields_.normal_,
                                                 motion_.pendulum_initial_position_, motion_.pendulum_direction());
      d = proj_b - proj_a;
    }

    if (exanb::dot(d, d) >= r2_) {
      return 1;
    } else {
      return 0;
    }
  }

  inline int operator()(exaDEM::Ball& a) {
    exaDEM::Ball& b = bcpd_.get_typed_driver<exaDEM::Ball>(bd_idx_);
    exanb::Vec3d d = a.fields_.center_ - b.fields_.center_;
    if (is_compressive(a.motion_type_)) {
      if (exanb::dot(d, d) >= 1e-12) {
        color_log::error("driver_displ_over", "Ball with compressive motion type should not move");
      }
      double r_diff = a.fields_.radius_ - b.fields_.radius_;
      if (r_diff * r_diff >= r2_) {
        return 1;
      } else {
        return 0;
      }
    }

    if (exanb::dot(d, d) >= r2_) {
      return 1;
    } else {
      return 0;
    }
  }

  inline int operator()(exaDEM::RShapeDriver& a) {
    using onika::cuda::vector_data;
    exaDEM::RShapeDriver& b = bcpd_.get_typed_driver<exaDEM::RShapeDriver>(bd_idx_);

    if ((a.fields_.center_ == b.fields_.center_) && (a.fields_.quat_ == b.fields_.quat_)) {
      return 0;
    }

#ifdef ONIKA_CUDA_VERSION
    storage_.reset();
    size_t size = a.shp_.get_number_of_vertices();
    const exanb::Vec3d* __restrict__ ptr_shp_vertices = vector_data(a.shp_.vertices_);
    RShapeDriverDisplacementFunctor SVDFunc = {
        r2_, ptr_shp_vertices, a.fields_.center_, a.fields_.quat_, b.fields_.center_, b.fields_.quat_};
    ReduceMaxRShapeDriverDisplacementFunctor func = {SVDFunc, storage_.get_ptr()};

    onika::parallel::ParallelForOptions opts;
    opts.omp_scheduling = onika::parallel::OMP_SCHED_STATIC;
    parallel_for(size, func, parallel_execution_context_(), opts);
    return storage_.get();
#else
    int sum = 0;
    // check each vertex
    auto check_dist = [](exanb::Vec3d& v1, exanb::Vec3d& v2, double r2) -> bool {
      exanb::Vec3d d = v1 - v2;
      if (exanb::dot(d, d) >= r2) return true;
      return false;
    };
    auto& shp_a = a.shp_;
    size_t size = shp_a.get_number_of_vertices();

    size_t start(0), end(size);
    // optimization for big shapes
    if (size > 100) {
      int mpi_rank, mpi_size;
      MPI_Comm_rank(comm_, &mpi_rank);
      MPI_Comm_size(comm_, &mpi_size);
      start = (size * mpi_rank) / mpi_size;
      end = (size * (mpi_rank + 1)) / mpi_size;
    }

    constexpr double homothety = 1.0;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = start; i < end; i++) {
      exanb::Vec3d va = shp_a.get_vertex(i, a.fields_.center_, homothety, a.fields_.quat_);
      exanb::Vec3d vb = shp_a.get_vertex(i, b.fields_.center_, homothety, b.fields_.quat_);
      if (check_dist(va, vb, r2_)) {
        sum++;
      }
    }
    return sum;
#endif
  }
};

class DriverDisplacementOver : public OperatorNode {
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
  inline std::string documentation() const final {
    return R"EOF(
         Compute the distance between each particle in grid input and it's backup position in backup_dem input.
         sets result output to true if at least one particle has moved further than threshold.
      )EOF";
  }

  // -----------------------------------------------
  // -----------------------------------------------
  inline void execute() final {
    MPI_Comm comm = *mpi;
    const double max_dist = *threshold;
    const double max_dist2 = max_dist * max_dist;

    int local_drivers_displ(0), global_drivers_displ(0);

    Drivers& drvs = *drivers;
    size_t size = drvs.get_size();
    if (size == 0) {
      *result = false;
      return;
    }

    // get backup
    Drivers& bcpd_ = *backup_drvs;
    size_t bcpd_size = bcpd_.get_size();

    auto pec_func = [this]() { return this->parallel_execution_context(); };

    for (size_t i = 0; i < bcpd_size && local_drivers_displ == 0; i++) {
      DriverDisplOver<decltype(pec_func)> func = {pec_func, max_dist2, comm, bcpd_, i, drvs.get_motion(i)};
      local_drivers_displ += drvs.apply(i, func);
    }

    MPI_Allreduce(&local_drivers_displ, &(global_drivers_displ), 1, MPI_INT, MPI_SUM, comm);
    *result = (global_drivers_displ > 0);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(driver_displ_over) {
  OperatorNodeFactory::instance()->register_factory("driver_displ_over", make_simple_operator<DriverDisplacementOver>);
}
}  // namespace exaDEM
