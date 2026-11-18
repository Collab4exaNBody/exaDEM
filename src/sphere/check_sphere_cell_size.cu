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

#include <exanb/core/domain.h>
#include <exanb/core/grid.h>
#include <exanb/core/make_grid_variant_operator.h>
#include <exanb/compute/reduce_cell_particles.h>

#include <exaDEM/color_log.hpp>
#include <exaDEM/sphere/check_sphere_cell_size.hpp>
#include <exaDEM/traversal.hpp>

#include <algorithm>
#include <sstream>

namespace exaDEM {
using namespace exanb;

template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>>
class CheckSphereCellSize : public OperatorNode {
  static constexpr FieldSet<field::_radius> reduce_field_set{};

  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT, REQUIRED);
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});
  ADD_SLOT(Domain, domain, INPUT, REQUIRED);
  ADD_SLOT(double, rcut_max, INPUT, 0.0, DocString{"Neighbor search distance, in physical space."});
  ADD_SLOT(bool, stop_execution, INPUT, true, DocString{"Stop the simulation if a check fails."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Checks that the neighbor search can find every sphere contact, using the current radii of the
        particles and the current domain transformation (xform). The largest sphere diameter,
        2 * max(radius), must not be larger than:
          - the neighbor search distance rcut_max,
          - the smallest cell size in physical space, cell_size * xform_min_scale.

        YAML example:

          - check_sphere_cell_size
      )EOF";
  }

  inline void execute() final {
    // largest particle radius
    const ReduceCellParticlesOptions rcpo = traversal_real->get_reduce_cell_particles_options();
    ReduceMaxRadiusFunctor func;
    double rmax = 0.0;
    reduce_cell_particles(*grid, false, func, rmax, reduce_field_set, parallel_execution_context(), {}, rcpo);
    MPI_Allreduce(MPI_IN_PLACE, &rmax, 1, MPI_DOUBLE, MPI_MAX, *mpi);

    const double diameter = 2.0 * rmax;
    const double cell_size_lab = domain->cell_size() * domain->xform_min_scale();
    ldbg << "check_sphere_cell_size: largest diameter = " << diameter << ", rcut_max = " << *rcut_max
         << ", cell size (physical space) = " << cell_size_lab << std::endl;

    if (diameter > *rcut_max) {
      std::ostringstream msg;
      msg << "The largest sphere diameter (" << diameter << ") is larger than the neighbor search distance"
          << " rcut_max (" << *rcut_max << "): contacts may be missed. Check the radii of the particles.";
      color_log::error("check_sphere_cell_size", msg.str(), *stop_execution);
    }
    if (diameter > cell_size_lab) {
      std::ostringstream msg;
      msg << "The largest sphere diameter (" << diameter << ") is larger than the cell size in physical"
          << " space (" << domain->cell_size() << " * " << domain->xform_min_scale() << " = " << cell_size_lab
          << "): increase the cell size.";
      color_log::error("check_sphere_cell_size", msg.str(), *stop_execution);
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(check_sphere_cell_size) {
  OperatorNodeFactory::instance()->register_factory("check_sphere_cell_size",
                                                    make_grid_variant_operator<CheckSphereCellSize>);
}
}  // namespace exaDEM
