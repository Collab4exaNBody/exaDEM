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

#include <exaDEM/push_to_quaternion.h>
#include <exaDEM/traversal.h>
#include <exanb/defbox/push_vec3_2nd_order.h>
#include <exanb/defbox/push_vec3_1st_order.h>

namespace exaDEM
{
  using namespace exanb;

  struct CombinedPrologFunctor
  {
    PushVec3SecondOrderFunctor push_f_v_r;
    PushVec3FirstOrderFunctor push_f_v;
    PushToQuaternionFunctor push_to_quaternion;

    ONIKA_HOST_DEVICE_FUNC inline void operator()(double &rx, double &ry, double &rz, double &vx, double &vy, double &vz, double &fx, double &fy, double &fz, Quaternion &Q, Vec3d &vrot, const Vec3d &arot) const
    {
      push_f_v_r(rx, ry, rz, vx, vy, vz, fx, fy, fz);
      push_f_v(vx, vy, vz, fx, fy, fz);
      push_to_quaternion(Q, vrot, arot);
    }
  };

  struct CombinedPrologXFormFunctor
  {
    PushVec3SecondOrderXFormFunctor push_f_v_r;
    PushVec3FirstOrderXFormFunctor push_f_v;
    PushToQuaternionFunctor push_to_quaternion;

    ONIKA_HOST_DEVICE_FUNC inline void operator()(double &rx, double &ry, double &rz, double &vx, double &vy, double &vz, double &fx, double &fy, double &fz, Quaternion &Q, Vec3d &vrot, const Vec3d &arot) const
    {
      push_f_v_r(rx, ry, rz, vx, vy, vz, fx, fy, fz);
      push_f_v(vx, vy, vz, fx, fy, fz);
      push_to_quaternion(Q, vrot, arot);
    }
  };
} // namespace exaDEM

namespace exanb
{
  template <> struct ComputeCellParticlesTraits<exaDEM::CombinedPrologFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

  template <> struct ComputeCellParticlesTraits<exaDEM::CombinedPrologXFormFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };
} // namespace exanb

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz, field::_orient, field::_vrot, field::_arot>> class CombinedComputeProlog : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz, field::_orient, field::_vrot, field::_arot>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(Domain, domain, INPUT, REQUIRED);
    ADD_SLOT(double, dt, INPUT);
    ADD_SLOT(Traversal, traversal_real, INPUT, DocString{"list of non empty cells within the current grid"});

  public:
    inline void execute() override final
    {
      // compute coefficients
      const double half_delta_t = (*dt) * 0.5;
      const double delta_t = *dt;
      const double delta_t2_2 = delta_t * delta_t * 0.5;
      // get non-empty cells
      auto [cell_ptr, cell_size] = traversal_real->info();

      if (domain->xform_is_identity())
      {
        PushVec3SecondOrderFunctor func1{delta_t, delta_t2_2};
        PushVec3FirstOrderFunctor func2{half_delta_t};
        PushToQuaternionFunctor func3{delta_t, half_delta_t, delta_t2_2};
        CombinedPrologFunctor func{func1, func2, func3};
        compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context(), cell_ptr, cell_size);
      }
      else
      {
        const Mat3d inv_xform = domain->inv_xform();
        PushVec3SecondOrderXFormFunctor func1{inv_xform, delta_t, delta_t2_2};
        PushVec3FirstOrderXFormFunctor func2{inv_xform, half_delta_t};
        PushToQuaternionFunctor func3{delta_t, half_delta_t, delta_t2_2};
        CombinedPrologXFormFunctor func{func1, func2, func3};
        compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context(), cell_ptr, cell_size);
      }
    }
  };

  template <class GridT> using CombinedComputePrologTmpl = CombinedComputeProlog<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(combined_compute_prolog) { OperatorNodeFactory::instance()->register_factory("combined_compute_prolog", make_grid_variant_operator<CombinedComputePrologTmpl>); }
} // namespace exaDEM
