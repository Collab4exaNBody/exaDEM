#pragma once

#include <exanb/core/basic_types_def.h>
#include <exanb/core/quaternion.h>
#include <exanb/core/declare_field.h>
#include <onika/oarray.h>

#include <cstdint>

// exaStamp fields are defined in namespace xstamp::field
// for rx, use xstamp::field::rx as a field descriptor
XSTAMP_DECLARE_FIELD(uint64_t        ,id                ,"particle id");
XSTAMP_DECLARE_FIELD(uint8_t         ,type              ,"particle type");
XSTAMP_DECLARE_FIELD(double          ,rx                ,"particle position X");
XSTAMP_DECLARE_FIELD(double          ,ry                ,"particle position Y");
XSTAMP_DECLARE_FIELD(double          ,rz                ,"particle position Z");
XSTAMP_DECLARE_FIELD(double          ,vx                ,"particle velocity X");
XSTAMP_DECLARE_FIELD(double          ,vy                ,"particle velocity Y");
XSTAMP_DECLARE_FIELD(double          ,vz                ,"particle velocity Z");
XSTAMP_DECLARE_FIELD(double          ,ax                ,"particle acceleration X");
XSTAMP_DECLARE_FIELD(double          ,ay                ,"particle acceleration Y");
XSTAMP_DECLARE_FIELD(double          ,az                ,"particle acceleration Z");

// DEM - reuse orient and angmom
XSTAMP_DECLARE_FIELD(double   ,mass           ,"particle mass");
XSTAMP_DECLARE_FIELD(double   ,homothety      ,"particle shape homothety");
XSTAMP_DECLARE_FIELD(double   ,radius         	,"radius");
XSTAMP_DECLARE_FIELD(uint32_t ,shape         	,"radius");
XSTAMP_DECLARE_FIELD(::exanb::Quaternion      ,orient  ,"angular position");
XSTAMP_DECLARE_FIELD(::exanb::Vec3d    , mom   	,"moment"); 
XSTAMP_DECLARE_FIELD(::exanb::Vec3d    , vrot   	,"angular velocity"); //
XSTAMP_DECLARE_FIELD(::exanb::Vec3d    , arot   	,"angular acceleration"); // 
XSTAMP_DECLARE_FIELD(::exanb::Vec3d    , inertia   	,"inertia values (same value in the diagonal)");
XSTAMP_DECLARE_FIELD(::exanb::Vec3d    , friction   	,"tmp field"); // 
typedef ::onika::oarray_t<::exanb::Vec3d, 8> VerticesType;
XSTAMP_DECLARE_FIELD(VerticesType , vertices   	,"list to compute vertices"); // 


// aliases
XSTAMP_DECLARE_ALIAS( fx, ax )
XSTAMP_DECLARE_ALIAS( fy, ay )
XSTAMP_DECLARE_ALIAS( fz, az )
//XSTAMP_DECLARE_ALIAS( couple, angacc);
//XSTAMP_DECLARE_ALIAS( angmom, angvel);

namespace exanb
{
  struct __unused_field_id {};
  static inline constexpr bool HAS_POSITION_BACKUP_FIELDS = false;
  static constexpr __unused_field_id PositionBackupFieldX = {};
  static constexpr __unused_field_id PositionBackupFieldY = {};
  static constexpr __unused_field_id PositionBackupFieldZ = {};
}

