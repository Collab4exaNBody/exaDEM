#pragma once


#include <exanb/core/basic_types.h>
#include <onika/cuda/cuda.h>
#include <onika/cuda/cuda_math.h>

namespace exanb
{
  ONIKA_HOST_DEVICE_FUNC
  static inline void mat3d_atomic_add_contribution( Mat3d & dst, const Mat3d& src )
  {
    ONIKA_CU_ATOMIC_ADD( dst.m11 , src.m11 );
    ONIKA_CU_ATOMIC_ADD( dst.m12 , src.m12 );
    ONIKA_CU_ATOMIC_ADD( dst.m13 , src.m13 );
    ONIKA_CU_ATOMIC_ADD( dst.m21 , src.m21 );
    ONIKA_CU_ATOMIC_ADD( dst.m22 , src.m22 );
    ONIKA_CU_ATOMIC_ADD( dst.m23 , src.m23 );
    ONIKA_CU_ATOMIC_ADD( dst.m31 , src.m31 );
    ONIKA_CU_ATOMIC_ADD( dst.m32 , src.m32 );
    ONIKA_CU_ATOMIC_ADD( dst.m33 , src.m33 );
  }

  ONIKA_HOST_DEVICE_FUNC
  static inline void mat3d_atomic_add_block_contribution( Mat3d & dst, const Mat3d& src )
  {
    ONIKA_CU_BLOCK_ATOMIC_ADD( dst.m11 , src.m11 );
    ONIKA_CU_BLOCK_ATOMIC_ADD( dst.m12 , src.m12 );
    ONIKA_CU_BLOCK_ATOMIC_ADD( dst.m13 , src.m13 );
    ONIKA_CU_BLOCK_ATOMIC_ADD( dst.m21 , src.m21 );
    ONIKA_CU_BLOCK_ATOMIC_ADD( dst.m22 , src.m22 );
    ONIKA_CU_BLOCK_ATOMIC_ADD( dst.m23 , src.m23 );
    ONIKA_CU_BLOCK_ATOMIC_ADD( dst.m31 , src.m31 );
    ONIKA_CU_BLOCK_ATOMIC_ADD( dst.m32 , src.m32 );
    ONIKA_CU_BLOCK_ATOMIC_ADD( dst.m33 , src.m33 );
  }
}
