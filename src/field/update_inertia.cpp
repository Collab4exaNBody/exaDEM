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
//#pragma xstamp_cuda_enable // DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <memory>

namespace exaDEM
{
  using namespace exanb;
  struct UpdateIntertiaSphereFunctor
  {
    const ParticleRegionCSGShallowCopy region; /**< Shallow copy of a particle region. */

    ONIKA_HOST_DEVICE_FUNC inline void operator()(const double mass, const double radius, Vec3d &inertia) const
    {
      const double inertia_value = 0.4 * mass * radius * radius;
      inertia = {inertia_value, inertia_value, inertia_value};
    }
    // If the region feature is activated
    ONIKA_HOST_DEVICE_FUNC inline void operator()(const double rx, const double ry, const double rz, const uint64_t id, const double mass, const double radius, Vec3d &inertia) const
    {
      Vec3d r = {rx, ry, rz};
      if (region.contains(r, id))
      {
        const double inertia_value = 0.4 * mass * radius * radius;
        inertia = {inertia_value, inertia_value, inertia_value};
      }
    }
  };
} // namespace exaDEM

namespace exanb
{
  template <> struct ComputeCellParticlesTraits<exaDEM::UpdateIntertiaSphereFunctor>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };
} // namespace exanb

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_inertia, field::_radius, field::_mass>> class UpdateInertia : public OperatorNode
  {

    using ComputeFields = FieldSet<field::_mass, field::_radius, field::_inertia>;
    using ComputeRegionFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_mass, field::_radius, field::_inertia>;
    static constexpr ComputeFields compute_field_set{};
    static constexpr ComputeRegionFields compute_region_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
    ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);

  public:
    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator updates the inertia field (0.4*mass*radius*radius).
        )EOF";
    }

    inline void execute() override final
    {
      if (region.has_value())
      {
        if (!particle_regions.has_value())
        {
          fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
        }

        if (region->m_nb_operands == 0)
        {
          ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
          region->build_from_expression_string(particle_regions->data(), particle_regions->size());
        }
        ParticleRegionCSGShallowCopy prcsg = *region;
        UpdateIntertiaSphereFunctor func = {prcsg};
        compute_cell_particles(*grid, false, func, compute_region_field_set, parallel_execution_context());
      }
      else
      {
        ParticleRegionCSGShallowCopy prcsg; // no region
        UpdateIntertiaSphereFunctor func = {prcsg};
        compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context());
      }
    }
  };

  template <class GridT> using UpdateInertiaTmpl = UpdateInertia<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("update_inertia", make_grid_variant_operator<UpdateInertiaTmpl>); }
} // namespace exaDEM
