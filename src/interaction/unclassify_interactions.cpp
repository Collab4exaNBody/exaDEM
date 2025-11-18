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

#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/compute_contact_force.h>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/classifier/interactionSOA.hpp>
#include <exaDEM/classifier/interactionAOS.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/classifier/classifier.hpp>

namespace exaDEM
{
  using namespace exanb;

  class UnclassifyInteractions : public OperatorNode
  {
    // attributes processed during computation
    using ComputeFields = FieldSet<field::_vrot, field::_arot>;
    static constexpr ComputeFields compute_field_set{};

    ADD_SLOT(GridCellParticleInteraction, ges, INPUT_OUTPUT, REQUIRED, DocString{"Interaction list"});
    ADD_SLOT(Classifier<InteractionSOA>, ic, INPUT, DocString{"Interaction lists classified according to their types"});
    ADD_SLOT(Classifier2, ic2, INPUT_OUTPUT);

    public:
    inline std::string documentation() const override final
    {
      return R"EOF(
        This operator copies data from the Interaction Classifier to the GridCellParticleInteraction.

        YAML example [no option]:

          - unclassify_interactions
        )EOF";
    }

    inline void execute() override final
    {
      /*auto& classi = *ic;
      double ftx = 0;
      double fty = 0;
      double ftz = 0;
      for(int type = 7; type < 13; type++)
      {
      	auto [data, size] = classi.get_info(type);
      	for(int i = 0; i < size; i++)
      	{
      		ftx+= data.ft_x[i];
      		fty+= data.ft_y[i];
      		ftz+= data.ft_z[i];
      		
      	}
      }
      printf("FTX: %f FTY: %f FTZ: %f\n", ftx, fty, ftz);*/
      auto& classifier2 = *ic2;
      printf("APRÈS CONTACT : \n");
      for(int i = 0; i < 4; i++)
      {
      	auto& data = classifier2.waves[i];
      	double* ftx = (double*)malloc(data.size() * sizeof(double));
      	double* fty = (double*)malloc(data.size() * sizeof(double));
      	double* ftz = (double*)malloc(data.size() * sizeof(double));
      	cudaMemcpy(ftx, data.ft_x, data.size() * sizeof(double), cudaMemcpyDeviceToHost);
      	cudaMemcpy(fty, data.ft_y, data.size() * sizeof(double), cudaMemcpyDeviceToHost);
      	cudaMemcpy(ftz, data.ft_z, data.size() * sizeof(double), cudaMemcpyDeviceToHost);
      	int active = 0;
      	for(int j = 0;  j < data.size(); j++)
      	{
      		if(ftx[j]!=0 || fty[j]!=0 || ftz[j]!=0)
      		{
      			active++;
      		}
      	}
      	printf("TYPE%d : %d/%d\n", i, active, data.size());
      	free(ftx);
      	free(fty);
      	free(ftz);
      }
      auto& classi = *ic;
      double ftx = 0;
      double fty = 0;
      double ftz = 0;
      for(int type = 7; type < 13; type++)
      {
      	auto [data, size] = classi.get_info(type);
      	for(int i = 0; i < size; i++)
      	{
      		ftx+= data.ft_x[i];
      		fty+= data.ft_y[i];
      		ftz+= data.ft_z[i];
      		
      	}
      }
      printf("FTX: %f FTY: %f FTZ: %f\n", ftx, fty, ftz);
      printf("\n\n\n");
      if (!ic.has_value())
        return;
      ic->unclassify(*ges);
    }
  };

  // === register factories ===
  ONIKA_AUTORUN_INIT(unclassify_interactions) { OperatorNodeFactory::instance()->register_factory("unclassify_interactions", make_simple_operator<UnclassifyInteractions>); }
} // namespace exaDEM
