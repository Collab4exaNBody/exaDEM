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
#include <onika/mpi/data_types.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <exanb/compute/reduce_cell_particles.h>
#include <exanb/core/grid.h>
#include <limits>

namespace exaDEM
{
  using namespace exanb;

  template <typename T, class FieldSetT, typename GridT, class = AssertGridHasFields<GridT, field::_mass>> class ReduceMinFieldOP : public OperatorNode
  {
    using ReduceField = FieldSet<FieldSetT>;
    static constexpr ReduceField reduce_field{};
    ADD_SLOT(GridT, grid, INPUT_OUTPUT);
    ADD_SLOT(double, mass, INPUT);
    ADD_SLOT(double, kn, INPUT);

    // -----------------------------------------------
    // ----------- Operator documentation ------------
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator checks the dt critical. 
        )EOF";
    }

    public:

    void check_slot()
    {
      if(!mass.has_field())
      {
         
      }
    }

    inline void execute() override final
    {
       
    }
  };

  template <class GridT> using MinMassTmpl = ReduceMinFieldOP<double, field::_mass, GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(min_mass) { OperatorNodeFactory::instance()->register_factory("min_mass", make_grid_variant_operator<MinMassTmpl>); }
  ONIKA_AUTORUN_INIT(min_rx)   { OperatorNodeFactory::instance()->register_factory("min_rx", make_grid_variant_operator<MinRxTmpl>); }
  ONIKA_AUTORUN_INIT(min_ry)   { OperatorNodeFactory::instance()->register_factory("min_ry", make_grid_variant_operator<MinRyTmpl>); }
  ONIKA_AUTORUN_INIT(min_rz)   { OperatorNodeFactory::instance()->register_factory("min_rz", make_grid_variant_operator<MinRzTmpl>); }

} // namespace exaDEM
