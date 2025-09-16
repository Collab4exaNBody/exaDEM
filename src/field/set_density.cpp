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
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/core/grid.h>
#include <exaDEM/color_log.hpp>
#include <memory>
#include <random>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_radius, field::_mass>> class SetDensity : public OperatorNode
  {
    ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
    ADD_SLOT(double, density, INPUT, 1, DocString{"density value applied to all particles, default is 1.0 ."});

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator applies the same density to all particles. If you want to apply various densities according to their material properties, use set_densities_multiple_materials.
        Note: Speres ONLY

        YAML example:

          - set_density:
             density: 0.02
        )EOF";
    }

  public:
    inline void execute() override final
    {
      auto cells = grid->cells();
      const IJK dims = grid->dimension();
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
          for (size_t j = 0; j < n; j++)
          {
            m[j] = coeff * r[j] * r[j] * r[j]; // 4/3 * pi * r^3 * d
            if(m[j] <= 0.0)
            {
              color_log::error("set_density", "Wrong definition of a mass for the particle " + std::to_string(cells[i][field::id][j]) + ".");
            }
          }
        }
        GRID_OMP_FOR_END
      }
    }
  };

  template <class GridT> using SetDensityTmpl = SetDensity<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(set_density) { OperatorNodeFactory::instance()->register_factory("set_density", make_grid_variant_operator<SetDensityTmpl>); }

} // namespace exaDEM
