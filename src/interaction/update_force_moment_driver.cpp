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

#include <exanb/core/grid.h>
#include <exaDEM/traversal.hpp>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/classifier/classifier_for_all.hpp>

struct UpdateForceMomentDriverFunc {
  Vec3d* forces;
  Vec3d* moments;
  Vec3d* fn;
  Vec3d* ft;

  ONIKA_HOST_DEVICE_FUNC
      inline void lockAndAdd(Vec3d& val, Vec3d&& add) {
        ONIKA_CU_ATOMIC_ADD(val.x, add.x);
        ONIKA_CU_ATOMIC_ADD(val.y, add.y);
        ONIKA_CU_ATOMIC_ADD(val.z, add.z);
      }


  template<typename InteractionT>
  void operator() (size_t i, InteractionT& I) {
    auto id = I.partner().id;
    lockAndAdd(forces[id], fn[i], ft[i]);
  }
};

namespace exaDEM {
class UpdateForceMomentDriverOp : public OperatorNode {

  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(DriverExtractor, driver_extractor, INPUT, OPTIONAL, DocString{"Extract specific data about drivers."});
  ADD_SLOT(Drivers, drivers, INPUT, OPTIONAL, DocString{"List of Drivers"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator copies interactions from GridCellParticleInteraction to the Interaction Classifier.

        YAML example [no option]:

          - update_force_moment_driver
      )EOF";
  }

  inline void execute() final {
    constexpr InteractionType IT = InteractionType::ParticleDriver;
    using vector_t = onika::memory::CudaMMVector<T>;
    if (driver_extractor.has_value()) {
      bool need_interaction = extractor.require_interaction();
      if (need_interaction) {
        // used to store data on CPU/GPU
        vector_t<Vec3d> forces(drvs.get_size(), 0);
        vector_t<Vec3d> moments(drvs.get_size(), 0);
        // data are stored in data buffer within the (i)nteraction (c)lassifier
        auto& classifier = *ic;

        for (int typeID = get_first_id<InteractionType::FirstIdDriver>() ;
             typeID <= get_last_id<InteractionType::LastIdDriver>() ; typeID++) {
          if (classifier.get_size(typeID) != 0) {
            // Setup Functor
            auto [dn, contact_position, fn, ft] = classifier.buffer_p(typeID);
            UpdateForceMomentDriverFunc func = {forces.data(), moments.data(), fn, ft};
            // Setup Interaciton
            auto [data, size] = ic.get_info<IT>(type);
            InteractionWrapper<IT> interactions(data);
            // Setup Options
            ParallelForOptions opts;
            opts.omp_scheduling = OMP_SCHED_STATIC;
            // Setup Wrapper
            WrapperForAll wrapper = {interactions, func};
            // Launch Kernel
            parallel_for(size, wrapper, parallel_execution_context, opts);
          }
        }
      }
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(update_force_moment_driver) {
  OperatorNodeFactory::instance()->register_factory("update_force_moment_driver",
                                                    make_simple_operator<UpdateForceMomentDriverOp>);
}
}  // namespace exaDEM
