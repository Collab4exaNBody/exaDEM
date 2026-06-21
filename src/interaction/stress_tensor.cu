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

// onika
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>
// order
#include <onika/parallel/parallel_for.h>
// exanb
#include <exanb/core/concurent_add_contributions.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/core/parallel_grid_algorithm.h>
#include <mpi.h>

// exaDEM
#include <exaDEM/atomic.h>

#include <exaDEM/classifier/classifier.hpp>
#include <exaDEM/classifier/classifier_for_all.hpp>

namespace exaDEM {
template <int type, bool sym>
struct ComputeStressTensorFunc {
  /** @brief Compute the stress tensor for a given interaction
   * @param idx The index of the interaction
   * @param I The interaction for which to compute the stress tensor
   * @param cells The grid cells containing the particles
   * @param dnp The normal overlap for the interaction
   * @param fnp The normal force for the interaction
   * @param ftp The tangential force for the interaction
   * @param cpp The contact point position for the interaction
   */
  template <typename TMPLC>
  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint64_t idx, auto& I, TMPLC* const __restrict__ cells,
                                                const double* const __restrict__ dnp,
                                                const Vec3d* const __restrict__ fnp,
                                                const Vec3d* const __restrict__ ftp,
                                                const Vec3d* const __restrict__ cpp) const {
    assert(type == I.type());
    constexpr bool is_innerbond = ConvertToIntertactionType<type>() == InteractionType::InnerBond;
    if (dnp[idx] < 0.0 || is_innerbond) {
      // get fij and cij
      auto& i = I.i();  // id for particle id, cell for cell id, p for position,
                        // sub for vertex id
      auto& cell = cells[i.cell];
      Vec3d fij = fnp[idx] + ftp[idx];
      Vec3d pos_i = {cell[field::rx][i.p], cell[field::ry][i.p], cell[field::rz][i.p]};
      Vec3d cij = cpp[idx] - pos_i;
      exaDEM::mat3d_atomic_add_contribution(cell[field::stress][i.p], exanb::tensor(fij, cij));

      // polyhedron - polyhedron || sphere - sphere
      if constexpr ((type <= 3 && sym == true) || is_innerbond) {
        auto& j = I.j();  // id for particle id, cell for cell id, p for
                          // position, sub for vertex id
        auto& cellj = cells[j.cell];
        Vec3d fji = -fij;
        Vec3d pos_j = {cellj[field::rx][j.p], cellj[field::ry][j.p], cellj[field::rz][j.p]};
        Vec3d cji = cpp[idx] - pos_j;
        exaDEM::mat3d_atomic_add_contribution(cellj[field::stress][j.p], exanb::tensor(fji, cji));
      }
    }
  }
};

/** @brief Functor for computing stress tensors
 * NTypes: number of interaction types
 * Sym: whether the interaction is symmetric
 * Op: the operator class that contains the parallel execution context
 */
template <int NTypes, bool Sym, typename Op>
struct ComputeStressTensorsLoop {
  Op* oper;  //< the operator class that contains the parallel execution context

  /** @brief Compute the stress tensor for all interactions of a given type
   * @param classifier The classifier containing the interactions
   * @param cells The grid cells containing the particles
   * @tparam Type The interaction type for which to compute the stress tensor
   * @tparam TMPLC The type of the grid cells
   */
  template <int Type, typename TMPLC>
  void iteration(Classifier& classifier, TMPLC* const __restrict__ cells) {
    static_assert(Type >= 0 && Type < NTypes);
    constexpr InteractionType IT = ConvertToIntertactionType<Type>();
    auto [Ip, size] = classifier.get_info<IT>(Type);
    // Skip if there are no interactions of this type
    if (size > 0) {
      ParallelForOptions opts;
      opts.omp_scheduling = OMP_SCHED_STATIC;
      const auto [dnp, cpp, fnp, ftp] =
          classifier.contact_state(Type);       // get parameters: get forces (fn, ft) and contact positions
                                                // (cp) computed into the contact force operators.
      InteractionWrapper<IT> interactions(Ip);  // get data: interaction
      ComputeStressTensorFunc<Type, Sym> func;  // get kernel
                                                // pack data, kernel, and interaction in a wrapper
      WrapperForAll wrapper(interactions, func, cells, dnp, fnp, ftp, cpp);
      // launch kernel
      parallel_for(size, wrapper, oper->parallel_execution_context(), opts);
    }
  }

  /** @brief Loop over all interaction types
   * @tparam Type The current interaction type
   * @tparam Args The types of the arguments
   * @param args The arguments
   */
  template <int Type, typename... Args>
  void loop(Args&&... args) {
    iteration<Type>(std::forward<Args>(args)...);
    if constexpr (Type - 1 >= 0) loop<Type - 1>(std::forward<Args>(args)...);
  }

  /** @brief Call the loop for all interaction types
   * @tparam Args The types of the arguments
   * @param args The arguments
   */
  template <typename... Args>
  void operator()(Args&&... args) {
    static_assert(NTypes >= 1);
    loop<NTypes - 1>(std::forward<Args>(args)...);
  }
};

template <typename GridT, class = AssertGridHasFields<GridT, field::_stress>>
class StressTensor : public OperatorNode {
  // attributes processed during computation
  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(Classifier, ic, INPUT, REQUIRED, DocString{"Interaction lists classified according to their types"});

 public:
  inline std::string documentation() const final {
    return R"EOF( 
        This operator computes the total stress tensor and the stress tensor for each particles. 

        YAML example [no option]:

          - stress_tensor
      )EOF";
  }

  inline void execute() final {
    constexpr bool sym = true;
    // check if grid has cells
    if (grid->number_of_cells() == 0) {
      return;
    }
    // check if interaction container is set
    if (!ic.has_value()) {
      return;
    }

    // get slot data
    auto cells = grid->cells();
    Classifier& cf = *ic;
    // get kernel

    ComputeStressTensorsLoop<InteractionTypeId::NTypes, sym, StressTensor> runner = {this};
    runner(cf, cells);
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(stress_tensor) {
  OperatorNodeFactory::instance()->register_factory("stress_tensor", make_grid_variant_operator<StressTensor>);
}
}  // namespace exaDEM
