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
#include <memory>
#include <exaDEM/set_fields.h>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_type>> class SetFields : public OperatorNode
  {
    using ComputeFieldsType = FieldSet<field::_type>;
    using ComputeRegionFieldsType = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_type>;
    static constexpr ComputeFieldsType compute_field_set_type{};
    static constexpr ComputeRegionFieldsType compute_region_field_set_type{};
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    // vector version
    ADD_SLOT(std::vector<uint32_t>, types, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<double>, densities, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<double>, radii, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<Vec3d>, velocities, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<Vec3d>, random_velocities, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<Vec3d>, angular_velocities, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<Vec3d>, random_angular_velocities, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<Quaternion>, quaternions, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(std::vector<bool>, random_quaternions, INPUT, OPTIONAL, DocString{""});
    // scalar version
    ADD_SLOT(uint32_t, type, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(double, density, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(double, radius, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(Vec3d, velocity, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(Vec3d, random_velocity, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(Vec3d, angular_velocity, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(Vec3d, random_angular_velocity, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(Quaternion, quaternion, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(bool, random_quaternion, INPUT, OPTIONAL, DocString{""});

    // other
    ADD_SLOT(bool, inertia, INPUT, OPTIONAL, DocString{""});
    ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
    ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator fills type id to all particles. 
        )EOF";
    }

  public:
    inline void execute() override final
    {


      bool set_type(false), set_density(false), set_random_velocity(false), set_radius(false), set_quaternion(false), set_random_quaternion(false), set_random_velocity(false), set_inertia(false), set_angular_velocity(false), set_random_angular_velocity(false);
      bool types(false), set_densities(false), set_random_velocities(false), set_radii(false), set_quaternions(false), set_random_quaternions(false), set_angular_velocities(false), set_random_angular_velocities(false); 

      bool is_region  = region.has_value();

      // scalar
      set_type = type.has_value();
      set_density = density.has_value();
      set_radius = radius.has_value();
      set_velocity = velocity.has_value();
      set_random_velocity = random_velocity.has_value();
      set_angular_velocity = angular_velocity.has_value();
      set_random_angular_velocity = random_angular_velocity.has_value();
      set_quaternion = quaternion.has_value();
      set_random_quaternion = random_quaternion.has_value();
      set_inertia = inertia.has_value();

      // vector
      types = types.has_value();
      set_densities = densities.has_value();
      set_radii = radii.has_value();
      set_velocities = velocities.has_value();
      set_random_velocities = random_velocities.has_value();
      set_angular_velocities = angular_velocities.has_value();
      set_random_angular_velocities= random_angular_velocities.has_value();
      set_quaternions = quaternions.has_value();
      set_random_quaternions = random_quaternions.has_value();

      // checks
      try {
        if(!types && (set_densities || set_radii || set_velocities || set_random_velocities
                                    || set_angular_velocities || set_random_angular_velocities
                                    || set_quaternions || set_random_quaternions)) {
          throw invalid_argument("Types is not defined, whereas a field for different types is defined.");

        }
        if(!type && (set_density || set_radius || set_velocity || set_random_velocity
                                 || set_angular_velocity || set_random_angular_velocity
                                 || set_quaternion || set_random_quaternion)) {
          throw invalid_argument("Type is not defined, whereas a uniform field is defined.");

        }
        if( types == false && set_type == false ) {
          throw invalid_argument("You must define either a type or a vector of types.");
        }
        if( type true && set_type == true) {
          throw invalide_argument("You can't define type and types");
        }
      }

      catch (exception& e) 
      {
        lerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
      } 

      if(types)
      {
        std::vector<uint32_t> _types = *types;
        int n = _types;
      }

      try {
        
      }


      auto cells = grid->cells();
      const IJK dims = grid->dimension();


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

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN (dims, i, loc, schedule(dynamic))
        {
          double *__restrict__ m = cells[i][field::mass];
          double *__restrict__ r = cells[i][field::radius];
          const double d = (*density);
          const double pi = 4 * std::atan(1);
          const double coeff = ((4.0) / (3.0)) * pi * d;
          const size_t n = cells[i].size();
#         pragma omp simd
          for (size_t j = 0; j < n; j++)
          {
            m[j] = coeff * r[j] * r[j] * r[j]; // 4/3 * pi * r^3 * d
          }
        }
        GRID_OMP_FOR_END
      }


      }
      else
      {
        if(type.has_value())
        {
          SetFunctor<uint32_t> func = {*type};
          compute_cell_particles(*grid, false, func, compute_field_set_type, parallel_execution_context());
        }

#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN (dims, i, loc, schedule(dynamic))
        {
          double *__restrict__ m = cells[i][field::mass];
          double *__restrict__ r = cells[i][field::radius];
          const double d = (*density);
          const double pi = 4 * std::atan(1);
          const double coeff = ((4.0) / (3.0)) * pi * d;
          const size_t n = cells[i].size();
#         pragma omp simd
          for (size_t j = 0; j < n; j++)
          {
            m[j] = coeff * r[j] * r[j] * r[j]; // 4/3 * pi * r^3 * d
          }
        }
        GRID_OMP_FOR_END
      }

        if(density.has_value())
      }


      // Polyhedra


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
        SetRegionFunctor<uint32_t> func = {prcsg, t};
        compute_cell_particles(*grid, false, func, compute_region_field_set, parallel_execution_context());
      }
      else
      {
        SetFunctor<uint32_t> func = {t};
        compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context());
      }
    }
  };

  template <class GridT> using SetFieldsTmpl = SetFields<GridT>;

  // === register factories ===
  CONSTRUCTOR_FUNCTION { OperatorNodeFactory::instance()->register_factory("set_fields", make_grid_variant_operator<SetFieldsTmpl>); }

} // namespace exaDEM
