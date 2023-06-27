#pragma once

#include <exanb/core/basic_types.h>

namespace exaDEM
{
  using namespace exanb;

  struct ParticlePairFriction
  {
    uint64_t m_particle_id = std::numeric_limits<uint64_t>::max();
    Vec3d m_friction = { 0. , 0. , 0. };
    inline bool is_null() const { return m_friction == Vec3d{0.,0.,0.}; }
  };

  template<size_t MaxPairs>
  struct ParticleFrictionArray
  {
    static constexpr size_t MAX_PARTICLE_FRICTION_PAIRS = MaxPairs;
    ParticlePairFriction m_frictions[MAX_PARTICLE_FRICTION_PAIRS];
  };

}

