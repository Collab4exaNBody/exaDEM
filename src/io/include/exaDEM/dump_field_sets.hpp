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
#pragma once

// ExaNBody
#include <exanb/core/grid.h>
#include <exanb/core/grid_fields.h>

// ExaDEM
#include <exaDEM/fields.h>

namespace exaDEM {
using namespace exanb;

// A .dump's header names its own fields, so which of these 4 known exaDEM field-set
using DumpFieldSet = FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_mass,
                              field::_homothety, field::_radius, field::_orient, field::_mom, field::_vrot,
                              field::_arot, field::_inertia, field::_id, field::_type, field::_group>;
using DumpFragmentationFieldSet =
    FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_cluster, field::_mass,
             field::_homothety, field::_radius, field::_orient, field::_mom, field::_vrot, field::_arot,
             field::_inertia, field::_id, field::_type, field::_group>;
// Deprecated: older dumps (pre-"group" field) written by read_dump_particle_interaction_v122 /
// read_dump_particle_fragmentation_v122.
using DumpFieldSetLegacy122 = FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz,
                                       field::_mass, field::_homothety, field::_radius, field::_orient, field::_mom,
                                       field::_vrot, field::_arot, field::_inertia, field::_id, field::_type>;
using DumpFragmentationFieldSetLegacy122 =
    FieldSet<field::_rx, field::_ry, field::_rz, field::_vx, field::_vy, field::_vz, field::_cluster, field::_mass,
             field::_homothety, field::_radius, field::_orient, field::_mom, field::_vrot, field::_arot,
             field::_inertia, field::_id, field::_type>;

// Superset grid type able to hold particles read with any of the 4 field sets above.
using DumpReaderGridT = GridFromFieldSet<FragmentationDEMFieldSet>;

}  // namespace exaDEM
