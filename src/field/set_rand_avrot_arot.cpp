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
#include <memory>
#include <random>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT, field::_vrot, field::_arot>> class SetRandVrotArot : public OperatorNode
  {
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(double, var_vrot, INPUT, 0, DocString{"Variance for angular velocity"});
    ADD_SLOT(Vec3d, mean_arot, INPUT, Vec3d{0, 0, 0}, DocString{"Average values (Vec3d) for angular veloctiy"});
    ADD_SLOT(double, var_arot, INPUT, 0, DocString{"Variance for angular acceleration"});
    ADD_SLOT(Vec3d, mean_vrot, INPUT, Vec3d{0, 0, 0}, DocString{"Average values for angular acceleration"});

  public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator generates random angular velocities and angular accelerations using a normal distribution law (var[double], mean[vec3d]).
        )EOF";
    }

    inline void execute() override final
    {
      auto cells = grid->cells();
      const IJK dims = grid->dimension();

      std::normal_distribution<> dist_vrot(0, *var_vrot);
      std::normal_distribution<> dist_arot(0, *var_arot);
#     pragma omp parallel
      {
        GRID_OMP_FOR_BEGIN (dims, i, loc, schedule(dynamic))
        {
          const auto *__restrict__ id = cells[i][field::id];
          Vec3d *__restrict__ vrot = cells[i][field::vrot];
          Vec3d *__restrict__ arot = cells[i][field::arot];
          const size_t n = cells[i].size();
#         pragma omp simd
          for (size_t j = 0; j < n; j++)
          {
            std::default_random_engine seed;
            seed.seed(id[j]); // TODO : Warning
            vrot[j].x = (*mean_vrot).x + dist_vrot(seed);
            seed.seed(id[j] + 1); // TODO : Warning
            vrot[j].y = (*mean_vrot).y + dist_vrot(seed);
            seed.seed(id[j] + 2); // TODO : Warning
            vrot[j].z = (*mean_vrot).z + dist_vrot(seed);

            seed.seed(n * id[j]); // TODO : Warning
            arot[j].x = (*mean_arot).x + dist_arot(seed);
            seed.seed(n * id[j] + 1); // TODO : Warning
            arot[j].y = (*mean_arot).y + dist_arot(seed);
            seed.seed(n * id[j] + 2); // TODO : Warning
            arot[j].z = (*mean_arot).z + dist_arot(seed);
          }
        }
        GRID_OMP_FOR_END
      }
    }
  };

  template <class GridT> using SetRandVrotArotTmpl = SetRandVrotArot<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(set_rand_avrot_arot) { OperatorNodeFactory::instance()->register_factory("set_rand_vrot_arot", make_grid_variant_operator<SetRandVrotArotTmpl>); }

} // namespace exaDEM
