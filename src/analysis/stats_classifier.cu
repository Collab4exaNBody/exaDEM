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

#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <mpi.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/interaction/interaction.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/shapes.hpp>

namespace exaDEM {

template <InteractionType IT>
inline void count_classifier_type(Classifier& classifier, int typeID, int& count, int& active_count, int& an,
                                  int& ghost_count, int& gn) {
  auto [data, size] = classifier.get_info<IT>(typeID);
  typename ClassifierContainer<IT>::View view = data.view();
#pragma omp parallel for reduction(+ : count, active_count, an, ghost_count, gn)
  for (size_t idx = 0; idx < size; idx++) {
    const auto I = view(idx);
    if (I.pair_.ghost_ == InteractionPair::PartnerGhost) {
      ghost_count++;
      gn++;
    } else {
      count++;
    }
    if (I.active()) {
      active_count++;
      an++;
    }
  }
}

template <typename GridT, class = AssertGridHasFields<GridT>>
class StatsClassifier : public OperatorNode {
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Classifier, ic, INPUT, REQUIRED, DocString{"Interaction lists classified according to their types"});

 public:
  inline std::string documentation() const final {
    return R"EOF( This operator displays DEM simulation data (from the classifier) for a given frequency.)EOF";
  }

  inline void execute() final {
    Classifier& classifier = *ic;

    // v vertex, e edge, f face, c cylinder, s surface, b ball, S rShape, sp sticked particles

    int nvv(0), nve(0), nvf(0), nee(0), nvvib(0);              // interaction counters [particles]
    int an(0), anvv(0), anve(0), anvf(0), anee(0), anvvib(0);  // active interaction counters
    int gn(0), gnvv(0), gnve(0), gnvf(0), gnee(0), gnvvib(0);  // ghost interaction counters
    int nvc(0), nvs(0), nvb(0), nSvv(0), nSve(0), nSvf(0), nSee(0), nSev(0), nSfv(0);  // interaction counters [drivers]
    int anvc(0), anvs(0), anvb(0), anSvv(0), anSve(0), anSvf(0), anSee(0), anSev(0),
        anSfv(0);  // interaction counters [drivers]
    int gnvc(0), gnvs(0), gnvb(0), gnSvv(0), gnSve(0), gnSvf(0), gnSee(0), gnSev(0),
        gnSfv(0);  // interaction counters [drivers]

    count_classifier_type<InteractionType::ParticleParticle>(classifier, InteractionTypeId::VertexVertex, nvv, anvv, an,
                                                             gnvv, gn);
    count_classifier_type<InteractionType::ParticleParticle>(classifier, InteractionTypeId::VertexEdge, nve, anve, an,
                                                             gnve, gn);
    count_classifier_type<InteractionType::ParticleParticle>(classifier, InteractionTypeId::VertexFace, nvf, anvf, an,
                                                             gnvf, gn);
    count_classifier_type<InteractionType::ParticleParticle>(classifier, InteractionTypeId::EdgeEdge, nee, anee, an,
                                                             gnee, gn);

    count_classifier_type<InteractionType::ParticleDriver>(classifier, InteractionTypeId::VertexCylinder, nvc, anvc, an,
                                                           gnvc, gn);
    count_classifier_type<InteractionType::ParticleDriver>(classifier, InteractionTypeId::VertexSurface, nvs, anvs, an,
                                                           gnvs, gn);
    count_classifier_type<InteractionType::ParticleDriver>(classifier, InteractionTypeId::VertexBall, nvb, anvb, an,
                                                           gnvb, gn);
    count_classifier_type<InteractionType::ParticleDriver>(classifier, InteractionTypeId::VertexRshapeDriverVertex,
                                                           nSvv, anSvv, an, gnSvv, gn);
    count_classifier_type<InteractionType::ParticleDriver>(classifier, InteractionTypeId::VertexRshapeDriverEdge, nSve,
                                                           anSve, an, gnSve, gn);
    count_classifier_type<InteractionType::ParticleDriver>(classifier, InteractionTypeId::VertexRshapeDriverFace, nSvf,
                                                           anSvf, an, gnSvf, gn);
    count_classifier_type<InteractionType::ParticleDriver>(classifier, InteractionTypeId::EdgeRshapeDriverEdge, nSee,
                                                           anSee, an, gnSee, gn);
    count_classifier_type<InteractionType::ParticleDriver>(classifier, InteractionTypeId::EdgeRshapeDriverVertex, nSev,
                                                           anSev, an, gnSev, gn);
    count_classifier_type<InteractionType::ParticleDriver>(classifier, InteractionTypeId::FaceRshapeDriverVertex, nSfv,
                                                           anSfv, an, gnSfv, gn);

    count_classifier_type<InteractionType::InnerBond>(classifier, InteractionTypeId::InnerBond, nvvib, anvvib, an,
                                                      gnvvib, gn);

    std::vector<int> val = {// particle
                            nvv, nve, nvf, nee, nvvib,
                            // driver
                            nvc, nvs, nvb, nSvv, nSve, nSvf, nSee, nSev, nSfv,
                            // total
                            an,
                            // particle
                            anvv, anve, anvf, anee, anvvib,
                            // ghost total
                            gn,
                            // ghost particle
                            gnvv, gnve, gnvf, gnee, gnvvib,
                            // driver
                            anvc, anvs, anvb, anSvv, anSve, anSvf, anSee, anSev, anSfv,
                            // ghost driver
                            gnvc, gnvs, gnvb, gnSvv, gnSve, gnSvf, gnSee, gnSev, gnSfv};

    int rank;
    MPI_Comm_rank(*mpi, &rank);

    if (rank == 0) {
      MPI_Reduce(MPI_IN_PLACE, val.data(), val.size(), MPI_INT, MPI_SUM, 0, *mpi);
    } else {
      MPI_Reduce(val.data(), val.data(), val.size(), MPI_INT, MPI_SUM, 0, *mpi);
    }

    int idx = 0;
    for (auto it : {&nvv,   &nve,   &nvf,  &nee,    &nvvib, &nvc,   &nvs,   &nvb,   &nSvv,   &nSve,  &nSvf,
                    &nSee,  &nSev,  &nSfv, &an,     &anvv,  &anve,  &anvf,  &anee,  &anvvib, &gn,    &gnvv,
                    &gnve,  &gnvf,  &gnee, &gnvvib, &anvc,  &anvs,  &anvb,  &anSvv, &anSve,  &anSvf, &anSee,
                    &anSev, &anSfv, &gnvc, &gnvs,   &gnvb,  &gnSvv, &gnSve, &gnSvf, &gnSee,  &gnSev, &gnSfv}) {
      *it = val[idx++];
    }

    lout << "=====================================================" << std::endl;
    lout << "* Type of interaction      = active / total / ghost" << std::endl;
    lout << "* Number of interactions   = " << an << " / "
         << nvv + nve + nvf + nee + nvvib + nvc + nvs + nvb + nSvv + nSve + nSvf + nSee + nSev + nSfv << " / " << gn
         << std::endl;
    lout << "* Vertex - Vertex          = " << anvv << " / " << nvv << " / " << gnvv << std::endl;
    lout << "* Vertex - Edge            = " << anve << " / " << nve << " / " << gnve << std::endl;
    lout << "* Vertex - Face            = " << anvf << " / " << nvf << " / " << gnvf << std::endl;
    lout << "* Edge   - Edge            = " << anee << " / " << nee << " / " << gnee << std::endl;
    lout << "* Vertex - Cylinder        = " << anvc << " / " << nvc << " / " << gnvc << std::endl;
    lout << "* Vertex - Surface         = " << anvs << " / " << nvs << " / " << gnvs << std::endl;
    lout << "* Vertex - Ball            = " << anvb << " / " << nvb << " / " << gnvb << std::endl;
    lout << "* Vertex - Vertex (Driver) = " << anSvv << " / " << nSvv << " / " << gnSvv << std::endl;
    lout << "* Vertex - Edge (Driver)   = " << anSve << " / " << nSve << " / " << gnSve << std::endl;
    lout << "* Vertex - Face (Driver)   = " << anSvf << " / " << nSvf << " / " << gnSvf << std::endl;
    lout << "* Edge   - Edge (Driver)   = " << anSee << " / " << nSee << " / " << gnSee << std::endl;
    lout << "* Edge (Driver) - Vertex   = " << anSev << " / " << nSev << " / " << gnSev << std::endl;
    lout << "* Face (Driver) - Vertex   = " << anSfv << " / " << nSfv << " / " << gnSfv << std::endl;
    lout << "* Vertice - Vertex (Stick) = " << nvvib << " / " << anvvib << " / " << gnvvib << std::endl;
    lout << "=====================================================" << std::endl;
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(stats_classifier) {
  OperatorNodeFactory::instance()->register_factory("stats_classifier", make_grid_variant_operator<StatsClassifier>);
}
}  // namespace exaDEM
