#pragma once 

namespace exaDEM
{
  using namespace exanb;

  template <typename FieldT>
    struct poly_div_field_volume
    {
      const shape* shps;
      ONIKA_HOST_DEVICE_FUNC inline void operator()(const uint32_t type, FieldT& value) const
      {
        const double volume = shps[type].get_volume();
        value = value / volume;
      }
    };

  template <typename FieldT>
    struct sphere_div_field_volume
    {
      double coeff_4t3_pi = (4/3) * std::atan(-1);
      ONIKA_HOST_DEVICE_FUNC inline void operator()(const double radius, FieldT& value) const
      {
        const double volume = coeff_4t3_pi * radius * radius * radius;
        value = value / volume;
      }
    };
}

namespace exanb
{
  template<typename FieldT> struct ComputeCellParticlesTraits<exaDEM::poly_div_field_volume<FieldT>>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };

  template<typename FieldT> struct ComputeCellParticlesTraits<exaDEM::sphere_div_field_volume<FieldT>>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
  };
}
