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
#include <exaDEM/interaction/grid_cell_interaction.hpp>
#include <exanb/extra_storage/sim_dump_reader_es.hpp>

namespace exaDEM {
using DumpFieldSet = FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_mass,
                              field::_homothety, field::_radius, field::_orient, field::_mom, field::_vrot,
                              field::_arot, field::_inertia, field::_id, field::_type, field::_group>;
using DumpFragmentationFieldSet =
    FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_cluster, field::_mass,
             field::_homothety, field::_radius, field::_orient, field::_mom, field::_vrot, field::_arot,
             field::_inertia, field::_id, field::_type, field::_group>;

template <typename GridT>
using SimDumpReadParticlesInteractionTmpl = SimDumpReadParticlesES<GridT, exaDEM::PlaceholderInteraction, DumpFieldSet>;
template <typename GridT>
using SimDumpReadParticlesFragmentationTmpl =
    SimDumpReadParticlesES<GridT, exaDEM::PlaceholderInteraction, DumpFragmentationFieldSet>;

// Deprecated
using DumpFieldSetLegacy122 = FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz,
                                       field::_mass, field::_homothety, field::_radius, field::_orient, field::_mom,
                                       field::_vrot, field::_arot, field::_inertia, field::_id, field::_type>;
using DumpFragmentationFieldSetLegacy122 =
    FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_cluster, field::_mass,
             field::_homothety, field::_radius, field::_orient, field::_mom, field::_vrot, field::_arot,
             field::_inertia, field::_id, field::_type>;

template <typename GridT>
using SimDumpReadParticlesInteractionTmplLegacy122 =
    SimDumpReadParticlesES<GridT, exaDEM::PlaceholderInteraction, DumpFieldSetLegacy122>;
template <typename GridT>
using SimDumpReadParticlesFragmentationTmplLegacy122 =
    SimDumpReadParticlesES<GridT, exaDEM::PlaceholderInteraction, DumpFragmentationFieldSetLegacy122>;
// ! Deprecated

// === register factories ===
ONIKA_AUTORUN_INIT(sim_dump_reader_interaction) {
  OperatorNodeFactory::instance()->register_factory("read_dump_particle_interaction",
                                                    make_grid_variant_operator<SimDumpReadParticlesInteractionTmpl>);
  OperatorNodeFactory::instance()->register_factory("read_dump_particle_fragmentation",
                                                    make_grid_variant_operator<SimDumpReadParticlesFragmentationTmpl>);
  // Deprecated
  OperatorNodeFactory::instance()->register_factory(
      "read_dump_particle_interaction_v122", make_grid_variant_operator<SimDumpReadParticlesInteractionTmplLegacy122>);
  OperatorNodeFactory::instance()->register_factory(
      "read_dump_particle_fragmentation_v122",
      make_grid_variant_operator<SimDumpReadParticlesFragmentationTmplLegacy122>);
  // !Deprecated
}
}  // namespace exaDEM
