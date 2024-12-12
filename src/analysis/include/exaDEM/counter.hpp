#pragma once

namespace exaDEM
{
  using vector_int = onika::memory::CudaMMVector<int>;


  struct countersWrapper
  {
    int size;
    int* data;
    int* types;
  };

  struct counters
  {
    vector_int m_data;
    vector_int m_types;
    counters(std::vector<int>& types)
    {
      size_t size = types.size();
      m_data.resize(size);
      m_types.resize(size);
      for(int i=0 ; i<size ; i++)
      {
        m_data[i] = 0;
        m_types[i] = types[i];
      }
    }

    countersWrapper get_wrapper()
    {
      return { onika::memory::vector_size(m_data), onika::memory::vector_data(m_data), onika::memory::vector_data(m_size) }
    }
  };


  struct reduceCounterWrapper
  {
    countersWrapper data;
    const ParticleRegionCSGShallowCopy region; /**< Shallow copy of a particle region. */

  }
}
