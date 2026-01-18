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
#include <exanb/core/domain.h>
#include <exanb/core/xform.h>

#include <exaDEM/forcefield/inner_bond_parameters.hpp>
#include <exaDEM/forcefield/inner_bond_force.hpp>
#include <exaDEM/forcefield/multimat_parameters.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/classifier/classifier_for_all.hpp>
#include <exaDEM/itools/itools.hpp>
#include <exaDEM/shapes.hpp>
#include <exaDEM/shape_detection.hpp>
#include <exaDEM/polyhedron/vertices.hpp>
#include <exaDEM/polyhedron/inner_bond.hpp>

namespace exaDEM {

template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>>
class ComputeInnerBondForce : public OperatorNode {
  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(CellVertexField, cvf, INPUT, REQUIRED, DocString{"Store vertex positions for every polyhedron"});
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(MultiMatParamsT<InnerBondParams>, multimat_ibp, INPUT, REQUIRED,
           DocString{"List of inner bond parameters for simulations with multiple materials"});
  ADD_SLOT(double, dt, INPUT, REQUIRED);
  ADD_SLOT(Classifier, ic, INPUT_OUTPUT, DocString{"Interaction lists classified according to their types"});
  ADD_SLOT(shapes, shapes_collection, INPUT_OUTPUT, DocString{"Collection of shapes"});
  // analyses
  ADD_SLOT(long, timestep, INPUT, REQUIRED);
  ADD_SLOT(std::string, dir_name, INPUT, REQUIRED, DocString{"Output directory name."});
  ADD_SLOT(std::string, interaction_basename, INPUT, REQUIRED,
           DocString{"Write an Output file containing interactions."});
  // private
  ADD_SLOT(bool, print_warning, PRIVATE, true,
           DocString{"This variable is used to ensure that warning messages are displayed only once."});
  // output
  ADD_SLOT(double, max_kn, INPUT_OUTPUT, 0,
           DocString{"Get the highest value of the input contact force parameters kn (used for dt_critical)"});

 public:
  inline std::string operator_name() { return "inner_bond_polyhedron"; }

  inline std::string documentation() const final {
    return R"EOF(
        This operator computes forces between inner_bonded particles using the contact law.

        YAML example:

          - inner_bond_polyhedron
      )EOF";
  }

  template <int start, int end, template <int, typename> typename FuncT, typename XFormT, typename... Args>
  void loop_contact_force(Classifier& classifier, XFormT& cp_xform, Args&&... args) {
    FuncT<start, XFormT> contact_law;
    if constexpr (start + 1 <= end) {
      loop_contact_force<start + 1, end, FuncT>(classifier, cp_xform, std::forward<Args>(args)...);
    }
  }

  /** fill highest kn */
  void scan_kn() {
    if (!(*print_warning)) {
      return;
    }
    double kn = 0.0;
    auto get_max_kn = [&kn](const InnerBondParams& cp) -> void { kn = std::max(kn, cp.kn); };
    multimat_ibp->apply(get_max_kn);
    *max_kn = kn;
  }

  inline void execute() final {
    using polyhedron::inner_bond_law;
    if (grid->number_of_cells() == 0) {
      return;
    }

    scan_kn();

    /** Get vertices and particles data */
    const auto cells = grid->cells();
    auto* vertex_fields = cvf->data();

    /** Get Shape */
    const shape* const shps = shapes_collection->data();

    /** deform matrice */
    const Mat3d& xform = domain->xform();
    bool is_def_xform = !domain->xform_is_identity();

    const double time = *dt;
    auto& classifier = *ic;

#define __params__ cells, vertex_fields, ibp, shps, time

    const auto& force_law_parameters = *multimat_ibp;
    const MultiMatContactParamsTAccessor<InnerBondParams> ibp = force_law_parameters.get_multimat_accessor();

    if (is_def_xform) {
      inner_bond_law<LinearXForm> func;
      func.xform = LinearXForm{xform};
      run_contact_law<InteractionTypeId::InnerBond>(parallel_execution_context(), classifier, func, __params__);
    } else {
      inner_bond_law<NullXForm> func;
      func.xform = NullXForm{};
      run_contact_law<InteractionTypeId::InnerBond>(parallel_execution_context(), classifier, func, __params__);
    }
#undef __params__
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(inner_bond_force_polyhedron) {
  OperatorNodeFactory::instance()->register_factory("inner_bond_polyhedron",
                                                    make_grid_variant_operator<ComputeInnerBondForce>);
}
}  // namespace exaDEM
