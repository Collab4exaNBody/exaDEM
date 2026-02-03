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
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interface/interface.hpp>

namespace exaDEM {
class UpdateInterfaces : public OperatorNode {
  ADD_SLOT(Classifier, ic, INPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(InterfaceManager, im, INPUT_OUTPUT, DocString{""});
  ADD_SLOT(InterfaceBuildManager, ibm, PRIVATE, DocString{""});
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator copies interactions from GridCellParticleInteraction to the Interaction Classifier.

        YAML example [no option]:

          - update_interfaces
      )EOF";
  }

  inline void execute() final {
    auto& build_manager = *ibm;
    rebuild_interface_Manager(build_manager, ic->get_data<InteractionType::InnerBond>(InteractionTypeId::InnerBond));

    int n_interfaces = build_manager.data.size();
    int total_interfaces = 0;
    MPI_Reduce(&n_interfaces, &total_interfaces, 1, MPI_INT, MPI_SUM, 0, *mpi);
    ldbg<< "Number of interfaces: " << total_interfaces << std::endl;
    auto& manager = *im;
    manager.resize(build_manager.data.size());
    std::memcpy(manager.data.data(), build_manager.data.data(), build_manager.data.size() * sizeof(Interface));
    assert(check_interface_consistency(*ibm, ic->get_data<InteractionType::InnerBond>(InteractionTypeId::InnerBond)));
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(update_interfaces) {
  OperatorNodeFactory::instance()->register_factory("update_interfaces", make_simple_operator<UpdateInterfaces>);
}
}  // namespace exaDEM
