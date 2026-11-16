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
#include <exanb/core/domain.h>
#include <onika/math/basic_types_operators.h>
#include <onika/math/basic_types_yaml.h>
#include <onika/scg/operator.h>
#include <onika/scg/operator_factory.h>
#include <onika/scg/operator_slot.h>

#include <algorithm>
#include <cmath>
#include <exaDEM/barostat/cell_barostat.hpp>
#include <exaDEM/color_log.hpp>

namespace exaDEM {
using namespace onika::scg;
using exanb::Domain;
using exanb::Mat3d;
using exanb::Vec3d;

class CellBarostatPrologNode : public OperatorNode {
  ADD_SLOT(Domain, domain, INPUT_OUTPUT);
  ADD_SLOT(double, dt, INPUT, REQUIRED);
  ADD_SLOT(Vec3d, cell_vh, INPUT_OUTPUT, Vec3d{0., 0., 0.},
           DocString{"Velocity of the diagonal cell degrees of freedom (Rockable's Cell.vh). Persisted across "
                     "steps; leave at its default (zero) at startup."});
  ADD_SLOT(Vec3d, cell_ah, INPUT_OUTPUT, Vec3d{0., 0., 0.},
           DocString{"Acceleration of the cell degrees of freedom (Rockable's Cell.ah), set by "
                     "cell_barostat_epilog on the previous step. Persisted across steps; leave at its default "
                     "(zero) at startup."});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        First half (drift + first half-kick) of a dynamical diagonal (no shear)
        periodic cell, same design as Rockable's PeriodicCell. Must run once per
        step, before compute_force, paired with cell_barostat_epilog run after the
        stress tensor is recomputed. Non-periodic axes are left untouched.

        YAML example:

          - cell_barostat_prolog
      )EOF";
  }

  inline void execute() final {
    const double h = *dt;
    Vec3d& vh = *cell_vh;
    const Vec3d& ah = *cell_ah;
    const Mat3d& old_xform = domain->xform();

    double hx = old_xform.m11;
    double hy = old_xform.m22;
    double hz = old_xform.m33;

    if (domain->periodic_boundary_x()) hx = cell_barostat_prolog_axis(hx, vh.x, ah.x, h);
    if (domain->periodic_boundary_y()) hy = cell_barostat_prolog_axis(hy, vh.y, ah.y, h);
    if (domain->periodic_boundary_z()) hz = cell_barostat_prolog_axis(hz, vh.z, ah.z, h);

    if (!cell_barostat_scale_valid(hx, hy, hz)) {
      color_log::error("cell_barostat_prolog",
                       "Invalid cell size (new diagonal xform = [" + std::to_string(hx) + ", " + std::to_string(hy) +
                           ", " + std::to_string(hz) +
                           "]): cell_mass/cell_damping are likely too small for this dt, or the periodic cell has "
                           "diverged.");
    }

    // Guard against the *cumulative* scale collapsing over many steps even though
    // each individual step above looked finite (e.g. Sig is unreachable with the
    // current contact stiffness): once a diagonal entry gets too small, its
    // contribution to det(xform) makes the matrix inverse numerically unreliable.
    if (cell_barostat_scale_collapsed(hx, hy, hz)) {
      color_log::error("cell_barostat_prolog",
                       "The domain has collapsed (new diagonal xform = [" + std::to_string(hx) + ", " +
                           std::to_string(hy) + ", " + std::to_string(hz) +
                           "]): Sig is likely unreachable with the current contact stiffness, or cell_mass/"
                           "cell_damping need tuning.");
    }

    domain->set_xform(diag_matrix(Vec3d{hx, hy, hz}));
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(cell_barostat_prolog) {
  OperatorNodeFactory::instance()->register_factory("cell_barostat_prolog",
                                                    make_simple_operator<CellBarostatPrologNode>);
}
}  // namespace exaDEM
