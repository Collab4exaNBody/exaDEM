#pragma once

#include <exanb/core/basic_types.h>
#include <exanb/compute/compute_cell_particles.h>
#include <onika/flat_tuple.h>


namespace exaDEM
{
  using namespace exanb;

  template<int idx, typename Tuple, typename Arg>
  ONIKA_HOST_DEVICE_FUNC static inline void setter(const Tuple& values, Arg& arg)
  {
    arg = values.get( onika::tuple_index<idx> );
  }

  template<int idx, typename Arg, typename Tuple, typename... Args>
  ONIKA_HOST_DEVICE_FUNC static inline void setter(const Tuple& values, Arg& arg, Args&... args) 
  {
    arg = values.get( onika::tuple_index<idx> );
    setter<idx+1>(values, args...);
  }


  template <typename... Ts>
  struct SetFunctor
  {
    template<typename... Args>
    ONIKA_HOST_DEVICE_FUNC inline void operator () (Args&... args) const
    {
      static constexpr int first = 0;
      setter<first>(m_default_values, args...);
    }
    onika::FlatTuple<Ts...> m_default_values;
  };

  template <typename Func, typename... Ts>
  struct SetFunctorWithProcessing
  {
    template<typename... Args>
    ONIKA_HOST_DEVICE_FUNC inline void operator () (Args&... args) const 
    {
      static constexpr int first = 0;
      setter<first>(m_default_values, m_processing(args)...);
    }
		mutable Func m_processing;
    onika::FlatTuple<Ts...> m_default_values;
  };

  template <typename... Ts>
  struct FilteredSetFunctor
  {
    template<typename... Args>
    ONIKA_HOST_DEVICE_FUNC inline void operator () (uint8_t type, Args&... args) const
    {
      if(type == filtered_type)
      {
        constexpr int first = 0;
        setter<first>(m_default_values, args...);
      }
    }

    uint8_t filtered_type;
    onika::FlatTuple<Ts...> m_default_values;
  };
}

namespace exanb
{
  template<class... Ts> struct ComputeCellParticlesTraits< exaDEM::SetFunctor<Ts...> >
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };
}

