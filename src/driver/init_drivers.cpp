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
#include "exanb/core/operator.h"
#include "exanb/core/operator_slot.h"
#include "exanb/core/operator_factory.h"
#include "exanb/core/make_grid_variant_operator.h"
#include "exanb/core/parallel_grid_algorithm.h"
#include "exanb/core/grid.h"
#include "exanb/core/domain.h"
#include "exanb/compute/compute_cell_particles.h"
#include <mpi.h>
#include <memory>
#include <exaDEM/driver_base.h>
#include <exaDEM/drivers.h>
#include <exaDEM/surface.h>

namespace exaDEM
{

  using namespace exanb;

  template<typename GridT>
    class InitDrivers : public OperatorNode
    {
      ADD_SLOT( Drivers , drivers , OUTPUT , DocString{"List of Drivers"});

      public:

      inline std::string documentation() const override final
      {
        return R"EOF(
        This operator creates a slot for drivers.
        )EOF";
      }

      inline void execute () override final
      {
        // do nothing
      }
    };

  template<class GridT> using InitDriversTmpl = InitDrivers<GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "init_drivers", make_grid_variant_operator< InitDriversTmpl > );
  }
}
