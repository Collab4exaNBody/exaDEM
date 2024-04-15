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

#include <exanb/extra_storage/sim_dump_reader_es.hpp>
#include <exaDEM/interaction/grid_cell_interaction.hpp>

namespace exaDEM
{ 
	using namespace exanb;
  using DumpFieldSet = FieldSet<field::_rx,field::_ry,field::_rz, field::_vx,field::_vy,field::_vz, field::_mass, field::_homothety, field::_radius, field::_orient , field::_mom , field::_vrot , field::_arot, field::_inertia , field::_id , field::_shape>;

	template <typename GridT> using SimDumpReadParticlesInteractionTmpl = SimDumpReadParticlesES <GridT, exaDEM::Interaction, DumpFieldSet>;
	template <typename GridT> using SimDumpReadParticlesDoubleTmpl = SimDumpReadParticlesES <GridT, double, DumpFieldSet>;
  // === register factories ===
  CONSTRUCTOR_FUNCTION
  { 

    OperatorNodeFactory::instance()->register_factory( "read_dump_particle_interaction" , make_grid_variant_operator<SimDumpReadParticlesInteractionTmpl> );
    OperatorNodeFactory::instance()->register_factory( "read_dump_particle_friction" , make_grid_variant_operator<SimDumpReadParticlesDoubleTmpl> );
  }
}
