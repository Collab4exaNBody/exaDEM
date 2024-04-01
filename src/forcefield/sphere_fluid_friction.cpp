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
#include <onika/memory/allocator.h>

#include <onika/soatl/field_pointer_tuple.h>
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exanb/grid_cell_particles/particle_cell_projection.h>
#include <exanb/compute/fluid_friction.h>

namespace exaDEM
{
  using namespace exanb;

  struct SphereFrictionFunctor
  {
    ONIKA_HOST_DEVICE_FUNC inline Vec3d operator () ( const Vec3d& r, const Vec3d& pv, const Vec3d& fv , const double cx, const double radius ) const
    {
      const Vec3d relative_velocity = fv - pv;
      const double relative_velocity_norm = norm(relative_velocity);
      return cx * (relative_velocity * relative_velocity_norm * M_PI * radius*radius);
    }
  };

  template<class GridT> using SphereFluidFriction = FluidFriction< GridT , SphereFrictionFunctor , FieldSet<field::_radius> >;
  
 // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
   OperatorNodeFactory::instance()->register_factory( "sphere_fluid_friction", make_grid_variant_operator< SphereFluidFriction > );
  }

}

