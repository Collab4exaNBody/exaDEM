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

class CellBarostatEpilogNode : public OperatorNode {
  ADD_SLOT(Domain, domain, INPUT_OUTPUT);
  ADD_SLOT(Mat3d, stress_tensor, INPUT, REQUIRED,
           DocString{"Measured stress tensor (Rockable's Cell.Sig), e.g. from avg_stress_tensor."});
  ADD_SLOT(double, volume, INPUT, REQUIRED, DocString{"Domain volume, e.g. from domain_volume."});
  ADD_SLOT(Vec3d, cell_sig, INPUT, REQUIRED,
           DocString{"Imposed normal stress per axis [sig_x, sig_y, sig_z] (Rockable's cellControl.Sig, "
                     "diagonal-only here). Same sign/unit convention as stress_tensor's diagonal: in this "
                     "codebase's virial convention, compressive is positive (opposite of the usual mechanics "
                     "convention)."});
  ADD_SLOT(double, cell_mass, INPUT, REQUIRED,
           DocString{"Fictive mass of the cell degrees of freedom (Rockable's Cell.mass). Rockable's rule of "
                     "thumb: roughly cellMassRatio * total particle mass, with cellMassRatio = O(1). Larger mass "
                     "= slower, more inertial (less jittery) response."});
  ADD_SLOT(double, cell_damping, INPUT, 0.0,
           DocString{"Viscous damping on the cell velocity (Rockable's Cell.damping). 0 = undamped, will "
                     "oscillate around the target stress; increase towards critical damping to settle without "
                     "overshoot."});
  ADD_SLOT(double, dt, INPUT, REQUIRED);
  ADD_SLOT(Vec3d, cell_vh, INPUT_OUTPUT, Vec3d{0., 0., 0.});
  ADD_SLOT(Vec3d, cell_ah, INPUT_OUTPUT, Vec3d{0., 0., 0.});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        Second half (new acceleration + second half-kick) of a dynamical diagonal
        (no shear) periodic cell, same design as Rockable's PeriodicCell. Must run
        once per step, after the stress tensor has been recomputed (e.g. after
        avg_stress_tensor), paired with cell_barostat_prolog run before
        compute_force. Non-periodic axes are left untouched.

        Per periodic axis i, with h_i the cell's diagonal xform entry, V the
        domain volume, and m/c the cell mass/damping:
          ah_i = ( (V / h_i) * (stress_tensor_ii - cell_sig_i) - c * vh_i ) / m
        (compressive stress is positive in this codebase's virial convention, so a
        positive cell_sig pulls the cell towards more compression.)

        YAML example:

          - cell_barostat_epilog:
             cell_sig: [ 1000.0, 1000.0, 1000.0 ]
             cell_mass: 1.0
             cell_damping: 10.0
      )EOF";
  }

  inline void execute() final {
    const Mat3d& stress = *stress_tensor;
    const Vec3d& sig = *cell_sig;
    const Mat3d& xf = domain->xform();
    const Vec3d bounds_size = domain->bounds_size();
    const double vol = *volume;
    const double mass = *cell_mass;
    const double damping = *cell_damping;
    const double h = *dt;

    Vec3d& vh = *cell_vh;
    Vec3d& ah = *cell_ah;

    // hi is the dimensionless xform diagonal entry (domain->xform() starts at 1.0);
    // the absolute box length along that axis is hi * bounds_size_i. The driving
    // force below needs the *absolute* length (Rockable's Vhinv = Vcell * hinv, with
    // h the absolute cell matrix), not the dimensionless hi alone.
    auto axis_accel = [&](bool periodic, double hi, double bounds_size_i, double sig_i, double stress_i,
                          double vh_i) -> double {
      if (!periodic) return 0.0;
      return cell_barostat_epilog_axis_accel(hi, bounds_size_i, vol, sig_i, stress_i, vh_i, mass, damping);
    };

    ah.x = axis_accel(domain->periodic_boundary_x(), xf.m11, bounds_size.x, sig.x, stress.m11, vh.x);
    ah.y = axis_accel(domain->periodic_boundary_y(), xf.m22, bounds_size.y, sig.y, stress.m22, vh.y);
    ah.z = axis_accel(domain->periodic_boundary_z(), xf.m33, bounds_size.z, sig.z, stress.m33, vh.z);

    if (!std::isfinite(ah.x) || !std::isfinite(ah.y) || !std::isfinite(ah.z)) {
      color_log::error("cell_barostat_epilog", "Invalid cell acceleration (" + std::to_string(ah.x) + ", " +
                                                   std::to_string(ah.y) + ", " + std::to_string(ah.z) +
                                                   "): check cell_mass, volume and stress_tensor.");
    }

    if (domain->periodic_boundary_x()) {
      vh.x += 0.5 * h * ah.x;
    }
    if (domain->periodic_boundary_y()) {
      vh.y += 0.5 * h * ah.y;
    }
    if (domain->periodic_boundary_z()) {
      vh.z += 0.5 * h * ah.z;
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(cell_barostat_epilog) {
  OperatorNodeFactory::instance()->register_factory("cell_barostat_epilog",
                                                    make_simple_operator<CellBarostatEpilogNode>);
}
}  // namespace exaDEM
