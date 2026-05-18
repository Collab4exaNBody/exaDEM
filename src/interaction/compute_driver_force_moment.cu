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

#include <exaDEM/atomic.h>
#include <exanb/core/grid.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/classifier/classifier_for_all.hpp>
#include <exaDEM/driver_extractor.hpp>
#include <exaDEM/drivers.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/traversal.hpp>

// Functor to accumulate forces and moments from particle-driver interactions
struct ComputeForceMomentDriverFunc {
  exanb::Vec3d* centers;  // Driver centers for moment computation
  exanb::Vec3d* cp;       // Contact points pointer
  exanb::Vec3d* fn;       // Normal forces pointer
  exanb::Vec3d* ft;       // Tangential forces pointer
  exanb::Vec3d* forces;   // Output: accumulated forces per driver (thread-safe)
  exanb::Vec3d* moments;  // Output: accumulated moments per driver (thread-safe)

  // Process each particle-driver interaction
  template <typename InteractionT>
  void operator()(size_t i, InteractionT& I) const {
    auto id = I.partner().id;  // Driver ID
    // Thread-safe addition of normal + tangential forces to the driver
    const exanb::Vec3d f = fn[i] + ft[i];  // Total force from this interaction
    exaDEM::lockAndAdd(forces[id], -f);
    // Compute moment contribution from this interaction and add to driver (negative sign for Newton's 3rd law)
    const exanb::Vec3d Cd = (cp[i] - centers[id]);
    const exanb::Vec3d mom = exanb::cross(Cd, -f) + -I.moment;
    exaDEM::lockAndAdd(moments[id], mom);
  }
};

// Traits specialization to indicate that ComputeForceMomentDriverFunc is CUDA compatible.
template <>
struct onika::parallel::ParallelForFunctorTraits<ComputeForceMomentDriverFunc> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = true;
};

// Functor to assign accumulated forces and moments to drivers
struct SetForceMomentFunc {
  exanb::Vec3d f;  // Force to set
  exanb::Vec3d m;  // Moment to set

  // Apply forces and moments to a driver (Newton's 3rd law: negative sign)
  template <typename DriverT>
  void operator()(DriverT& drv) const {
    drv.forces() = f;  // Set force
    drv.moment() = m;  // Set moment
  }
};

namespace exaDEM {
template <typename T>
using vector_t = onika::memory::CudaMMVector<T>;
class ComputeForceMomentDriverOp : public OperatorNode {
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(Drivers, drivers, INPUT, OPTIONAL, DocString{"List of Drivers"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator accumulates forces and moments from particle-driver interactions 
        and applies them to the corresponding drivers.

        The operator iterates through all particle-driver interactions, accumulates the 
        normal and tangential forces for each driver, and updates the driver's force and 
        moment vectors.

        YAML example [no option]:

          - compute_driver_force_moment
      )EOF";
  }

  inline void execute() final {
    constexpr InteractionType IT = InteractionType::ParticleDriver;
    if (drivers.has_value()) {
      auto& drvs = *drivers;
      // Allocate temporary buffers (CPU/GPU agnostic) for accumulated forces and moments
      vector_t<exanb::Vec3d> forces(drvs.get_size());
      vector_t<exanb::Vec3d> moments(drvs.get_size());
      vector_t<exanb::Vec3d> centers(drvs.get_size());
      // Get interaction classifier containing classified interactions
      auto& classifier = *ic;

      for (size_t i = 0; i < drvs.get_size(); i++) {
        exanb::Vec3d center;
        auto get_center = [&center] (auto& drv) {
          center = drv.position();
        };
        // center will be set by the functor, which captures it by reference
        drvs.apply(i, get_center);  // Apply functor to driver i
        centers[i] = center;
      }

      // Iterate through all driver interaction types
      for (int typeID = InteractionTypeId::FirstIdDriver; typeID <= InteractionTypeId::LastIdDriver; typeID++) {
        if (classifier.get_size(typeID) != 0) {
          // Extract force/moment data for this interaction type
          auto [dn, contact_position, fn, ft] = classifier.buffer_p(typeID);
          ComputeForceMomentDriverFunc func = {centers.data(), contact_position, fn, ft, forces.data(), moments.data()};
          // Get interactions of this type
          auto [data, size] = classifier.get_info<IT>(typeID);
          InteractionWrapper<IT> interactions(data);
          // Configure parallel execution
          ParallelForOptions opts;
          opts.omp_scheduling = OMP_SCHED_STATIC;
          // Setup Wrapper
          WrapperForAll wrapper = {interactions, func};
          // Execute parallel accumulation of forces and moments
          parallel_for(size, wrapper, parallel_execution_context(), opts);
        }
      }

      ONIKA_CU_DEVICE_SYNCHRONIZE();

      // Apply accumulated forces and moments to each driver
      for (size_t i = 0; i < drvs.get_size(); i++) {
        SetForceMomentFunc func = {forces[i], moments[i]};
        drvs.apply(i, func);  // Apply functor to driver i
      }
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(compute_driver_force_moment) {
  OperatorNodeFactory::instance()->register_factory("compute_driver_force_moment",
                                                    make_simple_operator<ComputeForceMomentDriverOp>);
}
}  // namespace exaDEM