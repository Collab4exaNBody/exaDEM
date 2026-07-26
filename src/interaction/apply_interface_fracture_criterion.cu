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

#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interface/apply_interface_fracture_criterion.hpp>
#include <exaDEM/interface/interface.hpp>

namespace exaDEM {
class ApplyInterfaceFractureCriterion : public OperatorNode {
  // attributes processed during computation
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(InterfaceManager, im, INPUT_OUTPUT, DocString{""});
  ADD_SLOT(bool, result, OUTPUT);
  ADD_SLOT(bool, display, INPUT, false, DocString{"Display interface broken."});

 public:
  inline std::string documentation() const final {
    return R"EOF(

        YAML example [no option]:

          - apply_interface_fracture_criterion
      )EOF";
  }

  inline void execute() final {
    auto& interfaces = *im;
    int init_value = 0;
    uint64_t number_of_broken_interfaces = 0;
    InteractionWrapper<InteractionType::InnerBond> data_wrapper = ic->get_sticked_interaction_wrapper();
    auto [dn, cp, fn, ft] = ic->contact_state(InteractionTypeId::InnerBond);

    ApplyAndReduceInterfaceFractureCriterionFunc func = {data_wrapper, fn, dn};

    number_of_broken_interfaces = reduce_interface(interfaces, func, init_value,
                                parallel_execution_context("apply_rupture_criterion"));
#ifdef APPLY_CRITERION_NO_REDUCTION
    ApplyInterfaceFractureCriterionFunc func = {interfaces.data_.data(), interfaces.break_interface_.data(),
                                                data_wrapper, fn, dn};

    onika::parallel::ParallelForOptions opts;
    opts.omp_scheduling = onika::parallel::OMP_SCHED_STATIC;
    parallel_for(interfaces.size(), func, parallel_execution_context(), opts);

    // No copy from GPU if the data has not been touuch by the GPU

#pragma omp parallel for reduction(+ : number_of_broken_interfaces)
    for (size_t i = 0; i < interfaces.size(); i++) {
      if (interfaces.break_interface_[i] == true) {
        /*auto [offset, size] = interfaces.data_[i];
        for (size_t j = 0; j < size; j++) {
          size_t idx = j + offset;
          data_wrapper.broke(idx);
        }*/
        number_of_broken_interfaces++;
      }
    }
#endif
    MPI_Allreduce(MPI_IN_PLACE, &number_of_broken_interfaces, 1, MPI_UINT64_T, MPI_SUM, *mpi);
    if (*display && number_of_broken_interfaces > 0) {
      if (number_of_broken_interfaces == 1) {
        lout << number_of_broken_interfaces << " interface has been broken." << std::endl;
      } else {
        lout << number_of_broken_interfaces << " interfaces have been broken." << std::endl;
      }
    }
    *result = number_of_broken_interfaces > 0;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(apply_interface_fracture_criterion) {
  OperatorNodeFactory::instance()->register_factory("apply_interface_fracture_criterion",
                                                    make_simple_operator<ApplyInterfaceFractureCriterion>);
}
}  // namespace exaDEM
