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

// exaDEM

#include <exaDEM/classifier/classifier_compare.hpp>
#include <exaDEM/color_log.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>

namespace exaDEM {
using namespace exanb;

/** @brief Copies an interaction storage (ges) into another slot. */
class CopyInteractions : public OperatorNode {
  ADD_SLOT(GridCellParticleInteraction, ges, INPUT, REQUIRED, DocString{"Interaction list"});
  ADD_SLOT(GridCellParticleInteraction, ges_copy, OUTPUT, DocString{"Copy of the interaction list"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Copies the interaction list (ges) into the slot ges_copy. Use rebind to choose the output slot.

        YAML example:

          - copy_interactions:
             rebind: { ges_copy: ges_gpu }
       )EOF";
  }

  inline void execute() final { *ges_copy = *ges; }
};

/**
 * @brief Compares two classifiers, independently of the order of the interactions in each wave.
 */
class CompareClassifiers : public OperatorNode {
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(Classifier, ic, INPUT, REQUIRED, DocString{"Reference classifier"});
  ADD_SLOT(Classifier, ic_test, INPUT, REQUIRED, DocString{"Classifier compared to the reference"});
  ADD_SLOT(double, atol, INPUT, 1e-12, DocString{"Absolute tolerance on the interaction fields (friction, ...)"});
  ADD_SLOT(double, rtol, INPUT, 1e-9, DocString{"Relative tolerance on the interaction fields (friction, ...)"});
  ADD_SLOT(bool, compare_values, INPUT, true,
           DocString{"Compare the interaction fields, otherwise only the lists of interactions"});
  ADD_SLOT(long, max_details, INPUT, 10, DocString{"Maximum number of reported interactions per type and category"});
  ADD_SLOT(bool, abort_on_error, INPUT, true, DocString{"Stop the simulation if the classifiers differ"});
  ADD_SLOT(bool, verbose, INPUT, false, DocString{"Print the per-type summary even if the classifiers are equal"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Compares two classifiers (ic: reference, ic_test: tested), independently of the order of the interactions.
        Interactions are identified by their pair (ids, cells, positions, sub-elements, type, swap, ghost); the
        specialized fields (friction, moment, weight, inner bond fields) of the matched interactions are compared
        with |ref - test| <= atol + rtol * |ref|.

        Typical use: compare nbh_polyhedron + classify_interactions with nbh_polyhedron_gpu, see
        config_compare_nbh_gpu.msp.

        YAML example:

          - compare_classifiers:
             rebind: { ic_test: ic_gpu }
             atol: 1e-12
             abort_on_error: true
       )EOF";
  }

  inline void execute() final {
    ClassifierCompareOptions opt;
    opt.atol_ = *atol;
    opt.rtol_ = *rtol;
    opt.compare_values_ = *compare_values;
    opt.max_details_ = static_cast<size_t>(std::max(0L, *max_details));
    ClassifierDiff diff = compare_classifiers(*ic, *ic_test, opt);

    int local_ok = diff.ok() ? 1 : 0;
    int global_ok = local_ok;
    MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN, *mpi);

    if (!local_ok) {
      color_log::warning("compare_classifiers", "The classifiers differ:\n" + diff.report());
    } else if (*verbose) {
      lout << "[compare_classifiers] the classifiers are equal:\n" << diff.report() << std::flush;
    }

    if (!global_ok && *abort_on_error) {
      color_log::mpi_error("compare_classifiers", "The classifiers differ (see the reports above).");
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(compare_classifiers) {
  OperatorNodeFactory::instance()->register_factory("copy_interactions", make_simple_operator<CopyInteractions>);
  OperatorNodeFactory::instance()->register_factory("compare_classifiers", make_simple_operator<CompareClassifiers>);
}
}  // namespace exaDEM
