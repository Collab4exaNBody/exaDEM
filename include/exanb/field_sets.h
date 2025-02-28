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

#include <exanb/fields.h>
#include <exanb/core/field_set_proto.h>

namespace exanb
{
  // DEM model field set
  using DEMFieldSet = FieldSet<
      // rx, ry and rz are added implicitly
      field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz, field::_mass, field::_homothety, field::_radius, field::_orient, field::_mom, field::_vrot, field::_arot, field::_inertia, field::_id, field::_type, field::_vertices>;

  // the standard set of FieldSet
  // use FieldSetsWith<fields...> (at the bottom of this file) to select a subset depending on required fields
  using StandardFieldSets = FieldSets<DEMFieldSet>;

} // namespace exanb

#include <exanb/core/field_set_utils.h>
