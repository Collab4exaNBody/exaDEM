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
#include <exanb/compute/compute_cell_particles.h>
#include <exanb/grid_cell_particles/particle_region.h>
#include <exaDEM/color_log.hpp>


namespace exaDEM {
struct RandomizeRadiusFunctor {
  double relative_deviation; 
  bool allow_exceed;
  const ParticleRegionCSGShallowCopy region; /**< Shallow copy of a particle region. */

  // random generator
  double randomize_function(double current_radius) const {
      static std::normal_distribution<double> dist(0.0, relative_deviation * current_radius / 3.0);
      static std::default_random_engine seed;
      double r = current_radius + dist(seed);
      double min_radius = current_radius * (1.0 - relative_deviation);
      double max_radius = allow_exceed ? current_radius * (1.0 + relative_deviation) : current_radius;

      if (r < min_radius) r = min_radius;
      if (r > max_radius) r = max_radius;
      return r;
    return r;
  }

  ONIKA_HOST_DEVICE_FUNC inline
    void compute(double& mass, double& radius, Vec3d& inertia) const {
    double density = mass /(4./3. * (4*atan(1)) * radius * radius * radius);  // deduce the density
    radius = randomize_function(radius);
    double volume = 4./3. * (4*atan(1)) * radius * radius * radius;
    mass = density * volume;
    const double inertia_value = 0.4 * mass * radius * radius;
    inertia = {inertia_value, inertia_value, inertia_value};
  }

  ONIKA_HOST_DEVICE_FUNC inline
      void operator()(double& mass, double& radius, Vec3d& inertia) const {
        compute(mass, radius, inertia);
  }
  // If the region feature is activated
  ONIKA_HOST_DEVICE_FUNC inline void operator()(const double rx, const double ry, const double rz, const uint64_t id,
                                                double& mass, double& radius, Vec3d& inertia) const {
    Vec3d r = {rx, ry, rz};
    if (region.contains(r, id)) {
      compute(mass, radius, inertia);
    }
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ComputeCellParticlesTraits<exaDEM::RandomizeRadiusFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = false;
  static inline constexpr bool CudaCompatible = false;
};
}  // namespace exanb

namespace exaDEM {
using namespace exanb;

template <typename GridT, class = AssertGridHasFields<GridT, field::_inertia, field::_radius, field::_mass>>
class RandomizeRadiusOp : public OperatorNode {
  using ComputeFields = FieldSet<field::_mass, field::_radius, field::_inertia>;
  using ComputeRegionFields =
      FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_mass, field::_radius, field::_inertia>;
  static constexpr ComputeFields compute_field_set{};
  static constexpr ComputeRegionFields compute_region_field_set{};

  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(double, relative_deviation, INPUT, REQUIRED, DocString{"Relative deviation for radius, e.g., 0.2 for ±20%"});
  ADD_SLOT(bool, allow_exceed, INPUT, OPTIONAL, DocString{"Allow the new radius to exceed the current radius"}, true);  ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
  ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);

 public:
  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
This operator randomly modifies the radius of spheres and updates
their mass and inertia accordingly, based on a relative deviation.

The new radius is sampled from a Gaussian distribution centered
on the current radius of each particle. The width of the distribution
is controlled by the 'relative_deviation' slot.

Parameters:

  - relative_deviation (double, required):
      Fraction of the current radius to use for randomization.
      For example, 0.2 means the new radius will vary roughly ±20%
      around the current radius.

  - allow_exceed (bool, optional, default=false):
      If true, the new radius can exceed the current radius.
      Otherwise, it is limited to a maximum equal to the current radius.

Behavior:

  - The minimum allowed radius is always: current_radius * (1 - relative_deviation)
  - The maximum allowed radius is:
      - current_radius if allow_exceed=false
      - current_radius * (1 + relative_deviation) if allow_exceed=true

YAML example:

  - randomize_radius:
      relative_deviation: 0.2
      allow_exceed: true


)EOF";
  }

  std::string operator_name() {
    return "randomize_radius";
  }

bool check_slot() {
    if (*relative_deviation <= 0.0 || *relative_deviation > 1.0) {
        color_log::error(operator_name(),
                         "relative_deviation must be in (0, 1]");
        return false;
    }
    return true;
}

  inline void execute() final {
    if (!check_slot()) {
      return;
    }

    if (region.has_value()) {
      // Do not touch this part
      if (!particle_regions.has_value()) {
        fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
      }
      if (region->m_nb_operands == 0) {
        ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
        region->build_from_expression_string(particle_regions->data(), particle_regions->size());
      }
      ParticleRegionCSGShallowCopy prcsg = *region;

      // here you can touch it if required
      RandomizeRadiusFunctor func = {*relative_deviation, *allow_exceed, prcsg};
      compute_cell_particles(*grid, false, func, compute_region_field_set, parallel_execution_context());
    } else {
      ParticleRegionCSGShallowCopy prcsg;  // no region
      RandomizeRadiusFunctor func = {*relative_deviation, *allow_exceed, prcsg};
      compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context());
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(update_inertia) {
  OperatorNodeFactory::instance()->register_factory("randomize_radius", make_grid_variant_operator<RandomizeRadiusOp>);
}
}  // namespace exaDEM