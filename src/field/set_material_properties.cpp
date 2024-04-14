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
#include <exanb/compute/compute_cell_particles.h>
#include <memory>
#include <exaDEM/set_fields.h>

namespace exaDEM
{
  using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_type, field::_radius, field::_mass, field::_orient>
    >
    class SetMaterialProperties : public OperatorNode
    {
      static constexpr double default_radius = 0.5;
      static constexpr double default_density = 1;
      static constexpr Quaternion default_quaternion = {0.0,0.0,0.0,1.0}; 
      using ComputeFields = FieldSet< field::_type, field::_radius, field::_mass, field::_orient>;
      using ComputeRegionFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_type, field::_radius, field::_mass, field::_orient>;
      static constexpr ComputeFields compute_field_set {};
      static constexpr ComputeRegionFields compute_region_field_set {};

      ADD_SLOT( GridT             , grid             , INPUT_OUTPUT );
      ADD_SLOT( uint8_t           , type             , INPUT , REQUIRED   , DocString{"type of particle to setialize"} );
      ADD_SLOT( double            , rad               , INPUT , default_radius  , DocString{"default radius value is 0.5 for all particles"} );
      ADD_SLOT( double            , density           , INPUT , default_density  , DocString{"default density value is 0 for all particles"} );
      ADD_SLOT( Quaternion        , quat              , INPUT , default_quaternion  , DocString{"default quaternion value for all particles "} );
      ADD_SLOT( ParticleRegions   , particle_regions , INPUT , OPTIONAL );
      ADD_SLOT( ParticleRegionCSG , region           , INPUT , OPTIONAL );


      public:

      inline std::string documentation() const override final
      {
        return R"EOF(
        This operator sets material properties for spheres, ie radius, denstiy and quaternion values.
        )EOF";
      }

      inline void execute () override final
      {
        // compute mass
        const double d   = (*density);
        const double r   = (*rad);
        const double pi   = 4*std::atan(1);
        const double coeff  = ((4.0)/(3.0)) * pi * d;    
        const double mass = coeff * r * r * r; // 4/3 * pi * r^3 * d 

        if( region.has_value() )
        {
          if( !particle_regions.has_value() )
          {
            fatal_error() << "GenericVec3Operator: region is defined, but particle_regions has no value" << std::endl;
          }

          if( region->m_nb_operands==0 )
          {
            ldbg << "rebuild CSG from expr "<< region->m_user_expr << std::endl;
            region->build_from_expression_string( particle_regions->data() , particle_regions->size() );
          }

          ParticleRegionCSGShallowCopy prcsg = *region;

          FilteredSetRegionFunctor<double, double, Quaternion> func { prcsg, *type, {r, mass, *quat}};
          compute_cell_particles( *grid , false , func , compute_region_field_set , parallel_execution_context() );
        }
        else
        {
          FilteredSetFunctor<double, double, Quaternion> func { *type, {r, mass, *quat}};
          compute_cell_particles( *grid , false , func , compute_field_set , parallel_execution_context() );
        }
      }
    };

  template<class GridT> using SetMaterialPropertiesTmpl = SetMaterialProperties<GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "set_material_properties", make_grid_variant_operator< SetMaterialPropertiesTmpl > );
  }
}
