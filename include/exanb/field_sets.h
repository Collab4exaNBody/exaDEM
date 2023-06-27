#pragma once

#include <exanb/fields.h>
#include <exanb/core/field_set_proto.h>

namespace exanb
{
  // DEM model field set
  using DEMFieldSet= FieldSet<
        // rx, ry and rz are added implicitly
	field::_vx,field::_vy,field::_vz,
	field::_fx,field::_fy,field::_fz,
	field::_mass,field::_homothety ,
	field::_radius,
	//field::_orient, field::_angmom, 
	field::_orient, 
	field::_mom, field::_vrot, field::_arot,
	field::_inertia, field::_id , field::_shape,
	field::_friction, field::_type
	>;

  // the standard set of FieldSet
  // use FieldSetsWith<fields...> (at the bottom of this file) to select a subset depending on required fields
  using StandardFieldSets = FieldSets< DEMFieldSet >;

}

#include <exanb/core/field_set_utils.h>

