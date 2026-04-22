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
#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>

#include <exaDEM/interface/interface.hpp>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/forcefield/contact_parameters.hpp>
#include <exaDEM/forcefield/contact_force.hpp>
#include <exaDEM/forcefield/multimat_parameters.hpp>

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT>>
class BrokenInterfaceFrictionUpdater : public OperatorNode {
  // attributes processed during computation
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(MultiMatParamsT<ContactParams>, multimat_cp, INPUT, REQUIRED,
           DocString{"List of contact parameters for simulations with multiple materials"});
  ADD_SLOT(InterfaceManager, im, INPUT_OUTPUT, DocString{""});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator modify the cumulated friction [stored in interfactions] if an interface is broken.
        These interactions will be transformed in VertexVertex interactions [unclassify operator]

        frition = max(friction, mu * fn)

        YAML example [no option]:

          - broken_interface_friction_updater
      )EOF";
  }

  inline void execute() final {
    auto& interfaces = *im;
    const MultiMatContactParamsTAccessor<ContactParams> cp = multimat_cp->get_multimat_accessor();
    InteractionWrapper<InteractionType::InnerBond> data_wrapper = ic->get_sticked_interaction_wrapper();
    auto [dn_ptr, cp_ptr, fn_ptr, ft_ptr] = ic->buffer_p(InteractionTypeId::InnerBond);
    auto cells = grid->cells();

    // No copy from GPU if the data has not been touuch by the GPU
#pragma omp parallel for
    for (size_t i = 0; i < interfaces.size(); i++) {
      if (interfaces.break_interface[i] == true) {
        auto [offset, size] = interfaces.data[i];
        auto type_a = cells[data_wrapper.cell_i[offset]][field::type][data_wrapper.p_i[offset]];
        auto type_b = cells[data_wrapper.cell_j[offset]][field::type][data_wrapper.p_j[offset]];
        for (size_t j = 0; j < size; j++) {
          size_t idx = j + offset;
          data_wrapper.broke(idx);
          double ft = exanb::norm(ft_ptr[idx]);
          double mu = cp(type_a, type_b).mu;
          double ft_threshold = mu * exanb::norm(fn_ptr[idx]);
          if( ft > ft_threshold && ft > 0 ) {
            data_wrapper.store_ft(ft_ptr[idx] * ft_threshold / ft, idx);
          }
          if (dn_ptr[idx] > 0) {
            data_wrapper.store_ft(Vec3d{0,0,0}, idx);
          }
        }
      }
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(broken_interface_friction_updater) {
  OperatorNodeFactory::instance()->register_factory("broken_interface_friction_updater",
                                                    make_grid_variant_operator<BrokenInterfaceFrictionUpdater>);
}
}  // namespace exaDEM
