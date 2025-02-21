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
#include <onika/math/quaternion_operators.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/core/domain.h>
#include <exanb/compute/compute_cell_particles.h>
#include <memory>

#include <exaDEM/angular_acceleration.h>
#include <exaDEM/angular_velocity.h>
#include <exaDEM/traversal.hpp>
#include <exanb/defbox/push_vec3_1st_order.h>
//#include <exanb/defbox/push_vec3_1st_order_xform.h>

namespace exaDEM
{
  using namespace exanb;

  struct CombinedEpilogFunctor
  {
    PushToAngularAccelerationFunctor angular_accel;
    PushToAngularVelocityFunctor angular_vel;
    PushVec3FirstOrderFunctor push_f_v;
    ONIKA_HOST_DEVICE_FUNC inline void operator()(const Quaternion &Q, const Vec3d &mom, Vec3d &vrot, Vec3d &arot, const Vec3d &inertia, double &vx, double &vy, double &vz, double fx, double fy, double fz) const
    {
      angular_accel(Q, mom, vrot, arot, inertia);
      angular_vel(vrot, arot);
      push_f_v(vx, vy, vz, fx, fy, fz);
    }
  };

  struct CombinedEpilogXFormFunctor
  {
    PushToAngularAccelerationFunctor angular_accel;
    PushToAngularVelocityFunctor angular_vel;
    PushVec3FirstOrderXFormFunctor push_f_v;
    ONIKA_HOST_DEVICE_FUNC inline void operator()(const Quaternion &Q, const Vec3d &mom, Vec3d &vrot, Vec3d &arot, const Vec3d &inertia, double &vx, double &vy, double &vz, double fx, double fy, double fz) const
    {
      angular_accel(Q, mom, vrot, arot, inertia);
      angular_vel(vrot, arot);
      push_f_v(vx, vy, vz, fx, fy, fz);
    }
  };
} // namespace exaDEM

namespace exanb
{
  template <> struct ComputeCellParticlesTraits<exaDEM::CombinedEpilogFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

  template <> struct ComputeCellParticlesTraits<exaDEM::CombinedEpilogXFormFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

} // namespace exanb

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_orient, field::_mom, field::_vrot, field::_arot, field::_inertia, field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz>> class CombinedComputeEpilog : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_orient, field::_mom, field::_vrot, field::_arot, field::_inertia, field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(Domain, domain, INPUT, REQUIRED);
    ADD_SLOT(double, dt, INPUT);
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});

  public:
    inline void execute() override final
    {
      const double delta_t = *dt;
      const double half_delta_t = delta_t * 0.5;
      auto [cell_ptr, cell_size] = traversal_real->info();

      if (domain->xform_is_identity())
      {
        PushToAngularAccelerationFunctor func1{};
        PushToAngularVelocityFunctor func2{half_delta_t};
        PushVec3FirstOrderFunctor func3{half_delta_t};
        CombinedEpilogFunctor func{func1, func2, func3};
        compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context(), cell_ptr, cell_size);
      }
      else
      {
        const Mat3d inv_xform = domain->inv_xform();
        PushToAngularAccelerationFunctor func1{};
        PushToAngularVelocityFunctor func2{half_delta_t};
        PushVec3FirstOrderXFormFunctor func3{inv_xform, half_delta_t};
        CombinedEpilogXFormFunctor func{func1, func2, func3};
        compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context(), cell_ptr, cell_size);
      }
    }
  };

  template <class GridT> using CombinedComputeEpilogTmpl = CombinedComputeEpilog<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("combined_compute_epilog", make_grid_variant_operator<CombinedComputeEpilogTmpl>); }
} // namespace exaDEM
