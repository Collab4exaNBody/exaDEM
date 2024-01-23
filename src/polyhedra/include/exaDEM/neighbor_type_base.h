#pragma once

#include <exanb/core/basic_types.h>

namespace exaDEM
{
  using namespace exanb;

	template<typename Type>
  struct ParticlePairT
  {
    uint64_t m_particle_id = std::numeric_limits<uint64_t>::max();
		Type m_item;
    inline bool is_null() const { return m_item == Type(); }
  };

  template<size_t MaxPairs, typename Type>
  struct ParticleVectorTArray
  {
    static constexpr size_t MAX_PARTICLE_FRICTION_PAIRS = MaxPairs;
    ParticlePairT<Type> m_data[MAX_PARTICLE_FRICTION_PAIRS];
  };
}

