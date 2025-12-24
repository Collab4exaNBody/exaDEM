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

#include <memory>
#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_slot.h>
#include <onika/scg/operator_factory.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/grid.h>

#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>

namespace exaDEM
{
  using namespace exanb;

  template <typename GridT, class = AssertGridHasFields<GridT>> class CheckOwnerPartnerGhosts : public OperatorNode
  {
    ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
    ADD_SLOT(GridT, grid, INPUT, REQUIRED);
    ADD_SLOT(GridCellParticleInteraction, ges, INPUT, REQUIRED, DocString{"Interaction list"});

  public:
    inline std::string documentation() const override final { return R"EOF( This operator displays DEM simulation data for a given frequency.)EOF"; }

    inline void execute() override final
    {
      auto &cells = ges->m_data;

      // v vertex, e edge, f face, c cylinder, s surface, b ball, S stl, sp sticked particles
      int partners[14];
      int owners[14];

      for(int i = 0; i<14 ; i++) { partners[i] = 0; owners[i] = 0; }

      for (size_t i = 0; i < cells.size(); i++)
      {
        for (auto &item : cells[i].m_data)
        {
          int type = item.type();
          if( item.pair.ghost == InteractionPair::PartnerGhost ) { partners[type]+=1; }
          else if( item.pair.ghost == InteractionPair::OwnerGhost ) { owners[type]+=1; }
        }
      }

      
      std::vector<int> val;
      val.resize(14*2);
      for(int i = 0; i<14 ; i++)
      {
        val[i] = owners[i];
        val[14+i] = partners[i];
      }

      int rank;
      MPI_Comm_rank(*mpi, &rank);

      if (rank == 0)
        MPI_Reduce(MPI_IN_PLACE, val.data(), val.size(), MPI_INT, MPI_SUM, 0, *mpi);
			else
				MPI_Reduce(val.data(), val.data(), val.size(), MPI_INT, MPI_SUM, 0, *mpi);

			std::vector<std::string> names = {
				"* Vertex - Vertex           = ",
				"* Vertex - Edge             = ",
				"* Vertex - Face             = ",
				"* Edge   - Edge             = ",
				"* Vertex - Cylinder         = ",
				"* Vertex - Surface          = ",
				"* Vertex - Ball             = ",
				"* Vertex - Vertex (STL)     = ",
				"* Vertex - Edge (STL)       = ",
				"* Vertex - Face (STL)       = ",
				"* Edge   - Edge (STL)       = ",
				"* Edge (STL) - Vertex       = ",
				"* Face (STL) - Vertex       = ",
				"* Vertice - Vertex (Stick)  = "
			};


      lout << "=====================================================" << std::endl;
			lout << "* Ghost type of interaction = partner / owner / error" << std::endl;
      for(int i = 0 ; i < 14 ; i++)
      {
        lout << names[i] << val[i] << " / " << val[14+i] << " / " << val[i] - val[14+i] << std::endl;
      }
      lout << "=====================================================" << std::endl;
		}
	};

	template <class GridT> using CheckOwnerPartnerGhostsTmpl = CheckOwnerPartnerGhosts<GridT>;

	// === register factories ===
	ONIKA_AUTORUN_INIT(check_owner_partner_ghosts) { OperatorNodeFactory::instance()->register_factory("check_owner_partner_ghosts", make_grid_variant_operator<CheckOwnerPartnerGhostsTmpl>); }
} // namespace exaDEM
