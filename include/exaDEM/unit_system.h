#pragma once

#include <onika/physics/units.h>
#include <onika/cuda/cuda.h>


namespace exaDEM
{

# define EXADEM_UNIT_SYSTEM_UNITS        \
    ::onika::physics::meter,             \
    ::onika::physics::gram,              \
    ::onika::physics::second,            \
    ::onika::physics::elementary_charge, \
    ::onika::physics::kelvin,            \
    ::onika::physics::particle,          \
    ::onika::physics::candela,           \
    ::onika::physics::radian
    
  static inline constexpr onika::physics::UnitSystem UNIT_SYSTEM = { { EXADEM_UNIT_SYSTEM_UNITS } };

  ONIKA_HOST_DEVICE_FUNC
  inline constexpr double to_internal_units( const onika::physics::Quantity & q )
  {
    constexpr onika::physics::UnitSystem target_units = { { EXADEM_UNIT_SYSTEM_UNITS } };
    return q.convert( target_units );
  }

}

#define EXADEM_QUANTITY( E ) ::exaDEM::to_internal_units( ONIKA_QUANTITY( E ) )
#define EXADEM_CONST_QUANTITY( E )  ::exaDEM::to_internal_units( ONIKA_CONST_QUANTITY( E ) )

