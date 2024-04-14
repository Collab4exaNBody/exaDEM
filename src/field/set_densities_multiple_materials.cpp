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
#include <exanb/core/operator.h>
#include <exanb/core/operator_slot.h>
#include <exanb/core/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <memory>

namespace exaDEM
{
using namespace exanb;

  template<typename GridT
    , class = AssertGridHasFields< GridT, field::_radius, field::_mass>
    >
    class SetDensities : public OperatorNode
    {
      ADD_SLOT( GridT , grid  , INPUT_OUTPUT );
      ADD_SLOT( std::vector<double>, densities , INPUT, REQUIRED, DocString{"Array of density values"});

      public:

      // -----------------------------------------------
      // ----------- Operator documentation ------------
      inline std::string documentation() const override final
      {
        return R"EOF(
        This operator applies various densities according to their material properties.
        )EOF";
      }

      inline void execute () override final
      {
        auto cells = grid->cells();
        const IJK dims = grid->dimension();
#     pragma omp parallel
        {
          GRID_OMP_FOR_BEGIN(dims,i,loc, schedule(dynamic) )
          {
            double* __restrict__ m = cells[i][field::mass];
            double* __restrict__ r = cells[i][field::radius];
            auto* __restrict__ myType = cells[i][field::type];
            const double* d   = (*densities).data();
            const double pi   = 4*std::atan(1);
            const double coeff  = ((4.0)/(3.0)) * pi;    
            const size_t n = cells[i].size();
#         pragma omp simd
            for(size_t j=0;j<n;j++)
            {
              m[j] = coeff * d[myType[j]] * r[j] * r[j] * r[j]; // 4/3 * pi * r^3 * d 
            }
          }
          GRID_OMP_FOR_END
        }
      }

    };

  template<class GridT> using SetDensitiesTmpl = SetDensities<GridT>;

  // === register factories ===  
  CONSTRUCTOR_FUNCTION
  {
    OperatorNodeFactory::instance()->register_factory( "set_densities_multiple_materials", make_grid_variant_operator< SetDensitiesTmpl > );
  }

}

