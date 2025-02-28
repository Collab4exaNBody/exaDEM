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
//#pragma xstamp_cuda_enable //! DO NOT REMOVE THIS LINE

#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exaDEM/set_fields.h>
#include <random>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_vx, field::_vy, field::_vz>> class SetRandVelocity : public OperatorNode
  {

    // Define some fieldsets used in compute_cell_particles
    using ComputeFieldsVx = FieldSet<field::_vx>;
    using ComputeFieldsVy = FieldSet<field::_vy>;
    using ComputeFieldsVz = FieldSet<field::_vz>;
    using ComputeFields = FieldSet<field::_vx, field::_vy, field::_vz>;
    using ComputeRegionFieldsVx = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_vx>;
    using ComputeRegionFieldsVy = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_vy>;
    using ComputeRegionFieldsVz = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_vz>;
    using ComputeRegionFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_vx, field::_vy, field::_vz>;
    static constexpr ComputeFieldsVx compute_field_vx{};
    static constexpr ComputeFieldsVy compute_field_vy{};
    static constexpr ComputeFieldsVz compute_field_vz{};
    static constexpr ComputeFields compute_field_set{};
    static constexpr ComputeRegionFieldsVx compute_region_field_vx{};
    static constexpr ComputeRegionFieldsVy compute_region_field_vy{};
    static constexpr ComputeRegionFieldsVz compute_region_field_vz{};
    static constexpr ComputeRegionFields compute_region_field_set{};

    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(double, var, INPUT, 0, DocString{"Variance (same for all dimensions)"});
    ADD_SLOT(Vec3d, mean, INPUT, Vec3d{0, 0, 0}, DocString{"Average vector value."});
    ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
    ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator generates random velocities using a normal distribution law (var[double], mean[vec3d]).
        )EOF";
    }

    inline void execute() override final
    {
      struct jammy
      {
        jammy(double var) { dist = std::normal_distribution<>(0, var); }

        inline int operator()(double &val)
        {
          val += dist(seed);
          seed();
          return 0;
        }

        std::normal_distribution<> dist;
        std::default_random_engine seed;
      };

      jammy gen(*var);
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

        SetRegionFunctor<double> func_vx = {prcsg, (*mean).x};
        SetRegionFunctor<double> func_vy = {prcsg, (*mean).y};
        SetRegionFunctor<double> func_vz = {prcsg, (*mean).z};
        GenSetRegionFunctor<jammy> func = {prcsg, gen};
        compute_cell_particles(*grid, false, func_vx, compute_region_field_vx, parallel_execution_context());
        compute_cell_particles(*grid, false, func_vy, compute_region_field_vy, parallel_execution_context());
        compute_cell_particles(*grid, false, func_vz, compute_region_field_vz, parallel_execution_context());
        compute_cell_particles(*grid, false, func, compute_region_field_set, parallel_execution_context());
      }
      else
      {
        double vx = (*mean).x;
        double vy = (*mean).y;
        double vz = (*mean).z;
        SetFunctor<double, double, double> func_vxyz = {vx, vy, vz};
        GenSetFunctor<jammy> func = {gen};
        compute_cell_particles(*grid, false, func_vxyz, compute_field_set, parallel_execution_context());
        compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context());
      }
    }
  };

  template <class GridT> using SetRandVelocityTmpl = SetRandVelocity<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("set_rand_velocity", make_grid_variant_operator<SetRandVelocityTmpl>); }
} // namespace exaDEM
