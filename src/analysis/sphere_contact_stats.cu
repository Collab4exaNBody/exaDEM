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
#include <onika/math/basic_types_operators.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

// exaNBody
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>

// exaDEM
#include <exaDEM/analysis_manager.hpp>
#include <exaDEM/classifier/classifier.hpp>

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT>>
class SphereContactStats : public OperatorNode {
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Classifier, ic, INPUT, REQUIRED, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(double, dt, INPUT, REQUIRED);
  ADD_SLOT(long, timestep, INPUT, REQUIRED, DocString{"Time iteration number"});
  ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory, usually defined into io_config."});
  ADD_SLOT(std::string, name, INPUT, "SphereContactStats.txt",
           DocString{"Filename. Default is: SphereContactStats.txt"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Computes micromechanical contact statistics for sphere-sphere contacts:
          - the number of active (geometrically touching) contacts
          - the coordination number Z
          - the mean normal contact force <fn>

        YAML example:

          - sphere_contact_stats
      )EOF";
  }

  inline void execute() final {
    Classifier& classifier = *ic;
    auto [Ip, size] = classifier.get_info<InteractionType::ParticleParticle>(InteractionTypeId::VertexVertex);
    typename ClassifierContainer<InteractionType::ParticleParticle>::View interactions = Ip.view();
    const auto [dnp, cpp, fnp, ftp] = classifier.contact_state(InteractionTypeId::VertexVertex);

    long local_nb_contacts = 0;
    double local_sum_fn = 0.0;

#pragma omp parallel for reduction(+ : local_nb_contacts, local_sum_fn)
    for (size_t idx = 0; idx < size; idx++) {
      const auto I = interactions(idx);
      if (I.pair_.ghost_ == InteractionPair::PartnerGhost) {  // skip the shadow copy owned by a ghost partner
        continue;
      }
      if (dnp[idx] >= 0.0) {
        continue;
      }
      local_nb_contacts++;
      local_sum_fn += exanb::norm(fnp[idx]);
    }

    long local_nb_particles = grid->number_of_particles() - grid->number_of_ghost_particles();
    long global_nb_contacts = 0;
    long global_nb_particles = 0;
    double global_sum_fn = 0.0;

    // Could be juste a MRI_Reduce
    MPI_Allreduce(&local_nb_contacts, &global_nb_contacts, 1, MPI_LONG, MPI_SUM, *mpi);
    MPI_Allreduce(&local_nb_particles, &global_nb_particles, 1, MPI_LONG, MPI_SUM, *mpi);
    MPI_Allreduce(&local_sum_fn, &global_sum_fn, 1, MPI_DOUBLE, MPI_SUM, *mpi);

    double mean_contact_force = (global_nb_contacts > 0) ? (global_sum_fn / global_nb_contacts) : 0.0;
    double coordination_number = (global_nb_particles > 0) ? (2.0 * global_nb_contacts) / global_nb_particles : 0.0;

    analysis::AnalysisFileManager manager = {};
    manager.set_path((*dir_name) + "/ExaDEMAnalyses");
    manager.set_filename(*name);
    double t = (*dt) * (*timestep);
    manager.add_element("Time", t, "%3f");
    double nb_contacts_d = static_cast<double>(global_nb_contacts);
    manager.add_element("NbContacts", nb_contacts_d, "%f");
    manager.add_element("CoordinationNumber", coordination_number, "%f");
    manager.add_element("MeanContactForce", mean_contact_force, "%f");
    manager.endl();
    manager.write();
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(sphere_contact_stats) {
  OperatorNodeFactory::instance()->register_factory("sphere_contact_stats",
                                                    make_grid_variant_operator<SphereContactStats>);
}
}  // namespace exaDEM
