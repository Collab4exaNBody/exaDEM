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

  template <typename GridT, class = AssertGridHasFields<GridT>> class StatsInteractions : public OperatorNode
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

      int nvv(0), nve(0), nvf(0), nee(0), nsp(0);                                                // interaction counters [particles]
      int an(0), anvv(0), anve(0), anvf(0), anee(0), ansp(0);                                    // active interaction counters
      int nvc(0), nvs(0), nvb(0), nSvv(0), nSve(0), nSvf(0), nSee(0), nSev(0), nSfv(0);          // interaction counters [drivers]
      int anvc(0), anvs(0), anvb(0), anSvv(0), anSve(0), anSvf(0), anSee(0), anSev(0), anSfv(0); // interaction counters [drivers]

      const exanb::Vec3d null = {0., 0., 0.};

      auto incr_interaction_counters = [null](const PlaceholderInteraction &I, int &count, int &active_count, int &active_global_count) -> void
      {
        count++;
        if (I.is_active())
        {
          active_count++;
          active_global_count++;
        }
      };

#     pragma omp parallel for reduction(+:nvv, nve, nvf, nee, nsp, an, anvv, anve, anvf, anee, ansp, nvc, nvs, nvb, nSvv, nSve, nSvf, nSee, nSev, nSfv, anvc, anvs, anvb, anSvv, anSve, anSvf, anSee, anSev, anSfv)
      for (size_t i = 0; i < cells.size(); i++)
      {
        for (auto &item : cells[i].m_data)
        {
          auto type = item.type();
          // particles
          if (type == 0)
            incr_interaction_counters(item, nvv, anvv, an);
          if (type == 1)
            incr_interaction_counters(item, nve, anve, an);
          if (type == 2)
            incr_interaction_counters(item, nvf, anvf, an);
          if (type == 3)
            incr_interaction_counters(item, nee, anee, an);
          // drivers
          // cylinder
          if (type == 4)
            incr_interaction_counters(item, nvc, anvc, an);
          // surface
          if (type == 5)
            incr_interaction_counters(item, nvs, anvs, an);
          // ball
          if (type == 6)
            incr_interaction_counters(item, nvb, anvb, an);
          // stl
          if (type == 7)
            incr_interaction_counters(item, nSvv, anSvv, an);
          if (type == 8)
            incr_interaction_counters(item, nSve, anSve, an);
          if (type == 9)
            incr_interaction_counters(item, nSvf, anSvf, an);
          if (type == 10)
            incr_interaction_counters(item, nSee, anSee, an);
          if (type == 11)
            incr_interaction_counters(item, nSev, anSev, an);
          if (type == 12)
            incr_interaction_counters(item, nSfv, anSfv, an);
          if (type == 13)
            incr_interaction_counters(item, nsp, ansp, an);
        }
      }

      std::vector<int> val = {// particle
                              nvv, nve, nvf, nee, nsp,
                              // driver
                              nvc, nvs, nvb, nSvv, nSve, nSvf, nSee, nSev, nSfv,
                              // total
                              an,
                              // particle
                              anvv, anve, anvf, anee, ansp,
                              // driver
                              anvc, anvs, anvb, anSvv, anSve, anSvf, anSee, anSev, anSfv};

      int rank;
      MPI_Comm_rank(*mpi, &rank);

      if (rank == 0)
        MPI_Reduce(MPI_IN_PLACE, val.data(), val.size(), MPI_INT, MPI_SUM, 0, *mpi);
      else
        MPI_Reduce(val.data(), val.data(), val.size(), MPI_INT, MPI_SUM, 0, *mpi);

      int idx = 0;
      for (auto it : {
          &nvv, &nve, &nvf, &nee, &nsp, 
          &nvc, &nvs, &nvb, &nSvv, &nSve, &nSvf, &nSee, &nSev, &nSfv, 
          &an, &anvv, &anve, &anvf, &anee, &ansp, 
          &anvc, &anvs, &anvb, &anSvv, &anSve, &anSvf, &anSee, &anSev, &anSfv})
        *it = val[idx++];

      lout << "==================================" << std::endl;
      lout << "* Type of interaction    : active / total " << std::endl;
      lout << "* Number of interactions : " << an << " / " << nvv + nve + nvf + nee + nsp + nvc + nvs + nvb + nSvv + nSve + nSvf + nSee + nSev + nSfv << std::endl;
      lout << "* Vertex - Vertex        : " << anvv << " / " << nvv << std::endl;
      lout << "* Vertex - Edge          : " << anve << " / " << nve << std::endl;
      lout << "* Vertex - Face          : " << anvf << " / " << nvf << std::endl;
      lout << "* Edge   - Edge          : " << anee << " / " << nee << std::endl;
      lout << "* Vertex - Cylinder      : " << anvc << " / " << nvc << std::endl;
      lout << "* Vertex - Surface       : " << anvs << " / " << nvs << std::endl;
      lout << "* Vertex - Ball          : " << anvb << " / " << nvb << std::endl;
      lout << "* Vertex - Vertex (STL)  : " << anSvv << " / " << nSvv << std::endl;
      lout << "* Vertex - Edge (STL)    : " << anSve << " / " << nSve << std::endl;
      lout << "* Vertex - Face (STL)    : " << anSvf << " / " << nSvf << std::endl;
      lout << "* Edge   - Edge (STL)    : " << anSee << " / " << nSee << std::endl;
      lout << "* Edge (STL) - Vertex    : " << anSev << " / " << nSev << std::endl;
      lout << "* Face (STL) - Vertex    : " << anSfv << " / " << nSfv << std::endl;
      lout << "* Sticked Vertices       : " << nsp   << " / " << ansp << std::endl;
      lout << "==================================" << std::endl;
    }
  };

  template <class GridT> using StatsInteractionsTmpl = StatsInteractions<GridT>;

  // === register factories ===
  ONIKA_AUTORUN_INIT(stats_interactions) { OperatorNodeFactory::instance()->register_factory("stats_interactions", make_grid_variant_operator<StatsInteractionsTmpl>); }
} // namespace exaDEM
