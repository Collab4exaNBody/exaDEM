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
#include <memory>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interface/interface.hpp>
#include <exaDEM/interface/apply_interface_fracture_criterion.hpp>

namespace exaDEM
{
  using namespace exanb;
  using namespace onika::parallel;

  class ApplyInterfaceFractureCriterion : public OperatorNode
  {
    // attributes processed during computation
		ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
		ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
		ADD_SLOT(InterfaceManager, im, INPUT_OUTPUT, DocString{""});
		ADD_SLOT(bool, result, OUTPUT);

		public:
		inline std::string documentation() const override final
		{
			return R"EOF(

        YAML example [no option]:

          - uapply_interface_fracture_criterion
      )EOF";
		}

		inline void execute() override final
		{
			auto& interfaces = *im;

			long long number_of_broken_interfaces = 0;

      InteractionWrapper<InteractionType::InnerBond> data_wrapper = ic->get_sticked_interaction_wrapper();

      ApplyInterfaceFractureCriterionFunc func = {interfaces.data.data(), interfaces.break_interface.data(), data_wrapper};

      ParallelForOptions opts;
      opts.omp_scheduling = OMP_SCHED_STATIC;
      parallel_for(interfaces.size(), func, parallel_execution_context(), opts);

			// No copy from GPU if the data has not been touuch by the GPU
# pragma omp parallel for reduction(+:number_of_broken_interfaces)
			for(size_t i=0 ; i<interfaces.size() ; i++)
			{
        if( interfaces.break_interface[i] == true )
        {
          auto [offset, size] = interfaces.data[i];
          for(size_t j=0 ; j<size ; j++) data_wrapper.broke(j+offset); 
				  number_of_broken_interfaces += interfaces.break_interface[i]; 
        }
			}

			MPI_Allreduce(MPI_IN_PLACE, &number_of_broken_interfaces, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, *mpi);
      if( number_of_broken_interfaces>0 ) lout << number_of_broken_interfaces << " interfaces have been borken." << std::endl;
			*result = number_of_broken_interfaces>0;
		}
	};

	// === register factories ===
	ONIKA_AUTORUN_INIT(apply_interface_fracture_criterion) { OperatorNodeFactory::instance()->register_factory("apply_interface_fracture_criterion", make_simple_operator<ApplyInterfaceFractureCriterion>); }
} // namespace exaDEM
