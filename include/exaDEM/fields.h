#pragma once

//#include <exanb/core/grid_fields.h>
#ifndef XNB_HAS_GRID_FIELDS_DEFINTIONS
#error Cannot be included outside of exanb/core/grid_fields.h
#endif

#include <onika/math/basic_types_def.h>
#include <onika/math/quaternion.h>
#include <cstdint>
#include <onika/oarray.h>

// DEM - reuse orient and angmom
XNB_DECLARE_FIELD(double, mass, "particle mass");
XNB_DECLARE_FIELD(double, homothety, "particle shape homothety");
XNB_DECLARE_FIELD(double, radius, "radius");
XNB_DECLARE_FIELD(::exanb::Quaternion, orient, "angular position");
XNB_DECLARE_FIELD(::exanb::Vec3d, mom, "moment");
XNB_DECLARE_FIELD(::exanb::Vec3d, vrot, "angular velocity");     //
XNB_DECLARE_FIELD(::exanb::Vec3d, arot, "angular acceleration"); //
XNB_DECLARE_FIELD(::exanb::Vec3d, inertia, "inertia values (same value in the diagonal)");
XNB_DECLARE_FIELD(::exanb::Mat3d, stress, "stress tensor"); //

namespace exaDEM
{
  using namespace ::exanb;

  // DEM model field set
  using DEMFieldSet = FieldSet<
      // rx, ry and rz are added implicitly
      field::_vx, field::_vy, field::_vz, field::_fx, field::_fy, field::_fz, field::_mass, field::_homothety, field::_radius, field::_orient, field::_mom, field::_vrot, field::_arot, field::_inertia, field::_id, field::_type, field::_stress
      >;
  
  static inline constexpr exanb::FieldSets<DEMFieldSet> available_field_sets_v = {};
}

#define HAS_POSITION_BACKUP_FIELDS false
#define PositionBackupFieldX ::exanb::unused_field_id_v
#define PositionBackupFieldY ::exanb::unused_field_id_v
#define PositionBackupFieldZ ::exanb::unused_field_id_v

#define XNB_AVAILABLE_FIELD_SETS ::exaDEM::available_field_sets_v

#include <exanb/compute/math_functors.h>
#include <onika/soatl/field_combiner.h>

ONIKA_DECLARE_FIELD_COMBINER( exaDEM, VelocityNorm2Combiner , vnorm2 , exanb::Vec3Norm2Functor , exanb::field::_vx , exanb::field::_vy , exanb::field::_vz )
ONIKA_DECLARE_FIELD_COMBINER( exaDEM, VelocityNormCombiner  , vnorm  , exanb::Vec3NormFunctor  , exanb::field::_vx , exanb::field::_vy , exanb::field::_vz )

