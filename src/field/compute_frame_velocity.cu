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

// compute_frame_velocity: computes the velocity of a rigid frame made of frozen
// particles. Each component is either imposed ('velocity') or obtained from the
// motion of a virtual rigid body of mass M driven by a target force ('force').
//
// It does NOT modify the particles: the result 'frame_velocity' is then applied
// with the existing operators freeze_particles + set_velocity (value rebound
// to frame_velocity).
// Place it in compute_force AFTER the contact / inner bond operators and
// BEFORE freeze_particles, so that the particle forces are complete.

#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>
#include <exanb/compute/reduce_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exaDEM/traversal.hpp>

#include <vector>
#include <string>
#include <algorithm>

namespace exaDEM {

// ---------------------------------------------------------------------------
// Reduction: sum of the forces of the particles located in the region
// ---------------------------------------------------------------------------
struct FrameForceValue {
  Vec3d f;          // sum of the forces
  double count;     // number of particles (check)
};

ONIKA_HOST_DEVICE_FUNC inline void FRAME_ATOMIC_ADD(Vec3d& a, const Vec3d& b) {
  ONIKA_CU_ATOMIC_ADD(a.x, b.x);
  ONIKA_CU_ATOMIC_ADD(a.y, b.y);
  ONIKA_CU_ATOMIC_ADD(a.z, b.z);
}

struct ReduceFrameForceFunctor {
  const ParticleRegionCSGShallowCopy region_;

  ONIKA_HOST_DEVICE_FUNC inline void operator()(FrameForceValue& local, const double rx, const double ry,
                                                const double rz, const uint64_t id, const double fx,
                                                const double fy, const double fz,
                                                reduce_thread_local_t = {}) const {
    Vec3d r = {rx, ry, rz};
    if (region_.contains(r, id)) {
      local.f += Vec3d{fx, fy, fz};
      local.count += 1.0;
    }
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(FrameForceValue& global, const FrameForceValue& local,
                                                reduce_thread_block_t) const {
    FRAME_ATOMIC_ADD(global.f, local.f);
    ONIKA_CU_ATOMIC_ADD(global.count, local.count);
  }

  ONIKA_HOST_DEVICE_FUNC inline void operator()(FrameForceValue& global, const FrameForceValue& local,
                                                reduce_global_t) const {
    FRAME_ATOMIC_ADD(global.f, local.f);
    ONIKA_CU_ATOMIC_ADD(global.count, local.count);
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ReduceCellParticlesTraits<exaDEM::ReduceFrameForceFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool RequiresCellParticleIndex = false;
  static inline constexpr bool CudaCompatible = true;
};
}  // namespace exanb

namespace exaDEM {
using namespace exanb;

template <typename GridT, class = AssertGridHasFields<GridT, field::_fx, field::_fy, field::_fz>>
class ComputeFrameVelocityNode : public OperatorNode {
  using ReduceFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_fx, field::_fy, field::_fz>;
  static constexpr ReduceFields reduce_field_set{};

  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});
  ADD_SLOT(ParticleRegions, particle_regions, INPUT, REQUIRED);
  ADD_SLOT(ParticleRegionCSG, region, INPUT, REQUIRED, DocString{"Frame region, e.g. 'TOP or LEFT_UP'"});
  ADD_SLOT(double, dt, INPUT, REQUIRED);
  ADD_SLOT(long, timestep, INPUT, REQUIRED);

  // control parameters
  ADD_SLOT(std::vector<std::string>, control_mode, INPUT, std::vector<std::string>{"velocity", "velocity", "velocity"},
           DocString{"Control mode per component: 'velocity' or 'force'"});
  ADD_SLOT(Vec3d, imposed_velocity, INPUT, Vec3d{0, 0, 0}, DocString{"Imposed velocity (velocity-controlled components)"});
  ADD_SLOT(Vec3d, target_force, INPUT, Vec3d{0, 0, 0},
           DocString{"Target force applied BY THE MACHINE ON THE FRAME (force-controlled components)"});
  ADD_SLOT(long, force_ramp_start, INPUT, 0, DocString{"Timestep at which the force ramp starts"});
  ADD_SLOT(long, force_ramp_steps, INPUT, 0, DocString{"Number of timesteps of the linear force ramp (0: no ramp)"});
  ADD_SLOT(double, frame_mass, INPUT, REQUIRED, DocString{"Virtual mass of the frame (force control)"});
  ADD_SLOT(double, frame_damping, INPUT, 0.0, DocString{"Viscous damping coefficient c (force control)"});
  ADD_SLOT(double, frame_max_velocity, INPUT, -1.0, DocString{"Cap on |v| per component (<=0: no cap)"});
  ADD_SLOT(long, frame_log_frequency, INPUT, 1000, DocString{"Print the frame state every n timesteps (<=0: never)"});

  // state, shared between the instances of the operator (stage 1 / stage 2)
  ADD_SLOT(Vec3d, frame_velocity, INPUT_OUTPUT, Vec3d{0, 0, 0}, DocString{"Current frame velocity (state)"});
  ADD_SLOT(Vec3d, frame_force, OUTPUT, DocString{"Force exerted by the specimen on the frame"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Velocity of a rigid frame made of frozen particles. Each component is either
        imposed ('velocity') or obtained from a virtual rigid body of mass M ('force'):

          M dv/dt = F_target(t) + F_specimen - c v

        where F_specimen is the sum of the forces of the particles of the region.
        The particles are not modified: apply the result with freeze_particles and
        set_velocity (value rebound to frame_velocity).

        YAML example:
          - compute_frame_velocity:
              region: TOP or LEFT_UP
              control_mode: [ force, velocity, velocity ]
              target_force: [ 1000. , 0. , 0. ]
              imposed_velocity: [ 0. , 0.002 , 0. ]
              force_ramp_start: 0
              force_ramp_steps: 100000
              frame_mass: 60.
              frame_damping: 7000.
        )EOF";
  }

  inline void execute() final {
    if (control_mode->size() != 3) {
      fatal_error() << "compute_frame_velocity: 'control_mode' must have 3 entries" << std::endl;
    }
    bool force_ctrl[3];
    for (int i = 0; i < 3; i++) {
      const std::string& m = (*control_mode)[i];
      if (m == "force") {
        force_ctrl[i] = true;
      } else if (m == "velocity") {
        force_ctrl[i] = false;
      } else {
        fatal_error() << "compute_frame_velocity: unknown mode '" << m << "' (use 'force' or 'velocity')" << std::endl;
      }
    }

    // region
    if (region->m_nb_operands == 0) {
      region->build_from_expression_string(particle_regions->data(), particle_regions->size());
    }
    ParticleRegionCSGShallowCopy prcsg = *region;

    // 1) sum of the forces on the frame (real particles only), then MPI reduction
    const ReduceCellParticlesOptions rcpo = traversal_real->get_reduce_cell_particles_options();
    FrameForceValue value = {Vec3d{0.0, 0.0, 0.0}, 0.0};
    ReduceFrameForceFunctor rfunc = {prcsg};
    reduce_cell_particles(*grid, false, rfunc, value, reduce_field_set, parallel_execution_context(), {}, rcpo);

    double local[4] = {value.f.x, value.f.y, value.f.z, value.count};
    double global[4] = {0.0, 0.0, 0.0, 0.0};
    MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_SUM, *mpi);
    const double Fs[3] = {global[0], global[1], global[2]};

    // 2) target force with linear ramp
    double ramp = 1.0;
    if (*force_ramp_steps > 0) {
      ramp = double(*timestep - *force_ramp_start) / double(*force_ramp_steps);
      ramp = std::clamp(ramp, 0.0, 1.0);
    }
    const double Ft[3] = {target_force->x * ramp, target_force->y * ramp, target_force->z * ramp};
    const double Vi[3] = {imposed_velocity->x, imposed_velocity->y, imposed_velocity->z};

    // 3) update of the frame velocity
    double v[3] = {frame_velocity->x, frame_velocity->y, frame_velocity->z};
    const double h = *dt;
    for (int i = 0; i < 3; i++) {
      if (force_ctrl[i]) {
        const double acc = (Ft[i] + Fs[i] - (*frame_damping) * v[i]) / (*frame_mass);
        v[i] += acc * h;
        if (*frame_max_velocity > 0.0) {
          v[i] = std::clamp(v[i], -(*frame_max_velocity), *frame_max_velocity);
        }
      } else {
        v[i] = Vi[i];
      }
    }
    *frame_velocity = Vec3d{v[0], v[1], v[2]};
    *frame_force = Vec3d{Fs[0], Fs[1], Fs[2]};

    // 4) log
    if (*frame_log_frequency > 0 && (*timestep % *frame_log_frequency) == 0) {
      lout << "compute_frame_velocity step " << *timestep << " n=" << long(global[3])
           << " F_target=(" << Ft[0] << "," << Ft[1] << "," << Ft[2] << ")"
           << " F_specimen=(" << Fs[0] << "," << Fs[1] << "," << Fs[2] << ")"
           << " v=(" << v[0] << "," << v[1] << "," << v[2] << ")" << std::endl;
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(compute_frame_velocity) {
  OperatorNodeFactory::instance()->register_factory("compute_frame_velocity",
                                                    make_grid_variant_operator<ComputeFrameVelocityNode>);
}
}  // namespace exaDEM
