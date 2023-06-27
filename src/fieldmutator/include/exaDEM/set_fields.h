#pragma once

#include <exanb/core/basic_types.h>
#include <exanb/compute/compute_cell_particles.h>
#include <tuple>


namespace exaDEM
{
  using namespace exanb;

  template<int idx, typename Tuple, typename Arg>
    ONIKA_HOST_DEVICE_FUNC inline void setter(const Tuple& values, Arg& arg)
    {
      arg = std::get<idx> (values);
    }

  template<int idx, typename Arg, typename Tuple, typename... Args>
    ONIKA_HOST_DEVICE_FUNC inline void setter(const Tuple& values, Arg& arg, Args&... args) 
    {
      arg = std::get<idx> (values);
      setter<idx+1>(values, args...);
    }


  template <typename... Ts>
    struct setFunctor
    {
      template<typename... Args>
	ONIKA_HOST_DEVICE_FUNC inline void operator () (Args&... args) const
	{
	  constexpr int first = 0;
	  setter<first>(m_default_values, args...);
	}
      std::tuple<Ts...> m_default_values;
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
      std::tuple<Ts...> m_default_values;
    };
}

namespace exanb
{
  template<> struct ComputeCellParticlesTraits<exaDEM::setFunctor<double>>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = false;
  };
}

namespace exanb
{
  template<> struct ComputeCellParticlesTraits<exaDEM::setFunctor<double,double, double, Vec3d>>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };
}

namespace exanb
{
  template<> struct ComputeCellParticlesTraits<exaDEM::setFunctor<Quaternion, double>>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = false;
  };
}

namespace exanb
{
  template<> struct ComputeCellParticlesTraits<exaDEM::FilteredSetFunctor<double, double, Quaternion>>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };
}
