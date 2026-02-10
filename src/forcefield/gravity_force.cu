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
#include <mpi.h>

#include <exaDEM/traversal.hpp>
#include <exaDEM/drivers.hpp>
#include <exaDEM/forcefield/gravity_force.hpp>

namespace exaDEM {

template <typename GridT, class = AssertGridHasFields<GridT, field::_mass, field::_fx, field::_fy, field::_fz>>
class GravityForce : public OperatorNode {
  static constexpr Vec3d default_gravity = {0.0, 0.0, -9.807};
  // attributes processed during computation
  using ComputeFields =
      field_accessor_tuple_from_field_set_t<FieldSet<field::_mass, field::_fx, field::_fy, field::_fz>>;
  static constexpr ComputeFields compute_field_set{};

  ADD_SLOT(MPI_Comm, mpi, INPUT, MPI_COMM_WORLD);
  ADD_SLOT(GridT, grid, INPUT_OUTPUT, REQUIRED);
  ADD_SLOT(Traversal, traversal_real, INPUT, REQUIRED, DocString{"list of non empty cells within the current grid"});
  ADD_SLOT(Vec3d, gravity, INPUT, default_gravity,
           DocString{"define the gravity constant in function of the gravity axis, default value are x axis = 0, y "
                     "axis = 0 and z axis = -9.807"});
  ADD_SLOT(Drivers, drivers, INPUT, OPTIONAL, DocString{"List of Drivers"});

 public:
  inline std::string documentation() const final {
    return R"EOF(
        This operator computes forces related to the gravity (particles and drivers).
        Note that the gravity is only applied to drivers using the motion type: PARTICLE.
 
        YAML example:

          - gravity_force:
             gravity: [0,0,-0.009807]
        )EOF";
  }

  inline void execute() final {
    // Particles
		const ComputeCellParticlesOptions ccpo = traversal_real->get_compute_cell_particles_options();
    GravityForceFunctor funcP{*gravity};
    compute_cell_particles(*grid, false, funcP, compute_field_set, parallel_execution_context(), ccpo);

    // Drivers
		if (drivers.has_value()) {
      int rank;
      MPI_Comm_rank(*mpi, &rank);
      // This operation should be done by only one mpi process.
      // An MPI reduction (sum) is performed on the driver forces after. 
      if (rank == 0) {
		    GravityForceDriverFunctor funcD{*gravity};
        for (size_t id = 0; id < drivers->get_size(); id++) {
				  drivers->apply(id, funcD);
        }
      }
    }
	}
};

// === register factories ===
ONIKA_AUTORUN_INIT(gravity_force) {
  OperatorNodeFactory::instance()->register_factory("gravity_force", make_grid_variant_operator<GravityForce>);
}

}  // namespace exaDEM
