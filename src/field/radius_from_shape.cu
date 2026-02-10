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
#include <memory>
#include <random>
#include <exaDEM/shapes.hpp>

namespace exaDEM {
using namespace exanb;
struct UpdateRadiusPolyhedronFunctor {
  const ParticleRegionCSGShallowCopy region; /**< Shallow copy of a particle region. */
  double* list_of_radius;                    /** filled in PolyhedraDefineRadius. */

  ONIKA_HOST_DEVICE_FUNC inline void operator()(uint32_t type, double& radius) const { radius = list_of_radius[type]; }
  // If the region feature is activated
  ONIKA_HOST_DEVICE_FUNC inline void operator()(const double rx, const double ry, const double rz, const uint64_t id,
                                                uint32_t type, double& radius) const {
    Vec3d r = {rx, ry, rz};
    if (region.contains(r, id)) {
      radius = list_of_radius[type];
    }
  }
};
}  // namespace exaDEM

namespace exanb {
template <>
struct ComputeCellParticlesTraits<exaDEM::UpdateRadiusPolyhedronFunctor> {
  static inline constexpr bool RequiresBlockSynchronousCall = true;
  static inline constexpr bool CudaCompatible = false;
};
}  // namespace exanb

namespace exaDEM {
template <typename GridT, class = AssertGridHasFields<GridT, field::_radius>>
class PolyhedraDefineRadius : public OperatorNode {
  using ComputeFields = FieldSet<field::_type, field::_radius>;
  using ComputeRegionFields = FieldSet<field::_rx, field::_ry, field::_rz, field::_id, field::_type, field::_radius>;
  static constexpr ComputeFields compute_field_set{};
  static constexpr ComputeRegionFields compute_region_field_set{};

  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(shapes, shapes_collection, INPUT, REQUIRED, DocString{"Collection of shapes"});
  ADD_SLOT(double, rcut_max, INPUT_OUTPUT, 0.0);
  ADD_SLOT(ParticleRegions, particle_regions, INPUT, OPTIONAL);
  ADD_SLOT(ParticleRegionCSG, region, INPUT, OPTIONAL);

  // -----------------------------------------------
  // ----------- Operator documentation ------------
  inline std::string documentation() const final {
    return R"EOF(
        DEPRECIATED [1.1.3], please use set_fields.

        YAML example [no option]:

          - radius_from_shape
        )EOF";
  }

 public:
  inline void execute() final {
    // compute the biggest radius for each polyhedron
    const shapes shps = *shapes_collection;
    const size_t size = shps.size();
    onika::memory::CudaMMVector<double> r;
    r.resize(size);
    double rmax = *rcut_max;
    for (size_t i = 0; i < size; i++) {
      double rad_max = shps[i]->compute_max_rcut();
      r[i] = rad_max;
      rmax = std::max(rmax, 2 * rad_max);  // r * maxrcut
    }
    *rcut_max = rmax;

    // now, fill the radius field
    if (region.has_value()) {
      ParticleRegionCSGShallowCopy prcsg;
      if (!particle_regions.has_value()) {
        fatal_error() << "Region is defined, but particle_regions has no value" << std::endl;
      }
      if (region->m_nb_operands == 0) {
        ldbg << "rebuild CSG from expr " << region->m_user_expr << std::endl;
        region->build_from_expression_string(particle_regions->data(), particle_regions->size());
      }
      prcsg = *region;
      UpdateRadiusPolyhedronFunctor func = {prcsg, onika::cuda::vector_data(r)};
      compute_cell_particles(*grid, false, func, compute_region_field_set, parallel_execution_context());
    } else {
      ParticleRegionCSGShallowCopy prcsg;
      UpdateRadiusPolyhedronFunctor func = {prcsg, onika::cuda::vector_data(r)};
      compute_cell_particles(*grid, false, func, compute_field_set, parallel_execution_context());
    }
  }
};

// === register factories ===
ONIKA_AUTORUN_INIT(radius_from_shape) {
  OperatorNodeFactory::instance()->register_factory("radius_from_shape",
                                                    make_grid_variant_operator<PolyhedraDefineRadius>);
}

}  // namespace exaDEM
