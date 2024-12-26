#pragma once

#include <exaDEM/classifier/classifier.hpp>

namespace exaDEM
{
  namespace itools
  {
    template <typename T> inline double get_min_dn(const Classifier<T> &classifier)
    {
      // TODO : Implement a GPU version
      double res = 0;
      for (size_t i = 0; i < classifier.number_of_waves(); i++)
      {
        const auto &buffs = classifier.buffers[i];
        const double *const dnp = onika::cuda::vector_data(buffs.dn);
        const size_t size = onika::cuda::vector_size(buffs.dn);
#       pragma omp parallel for reduction(min: res)
        for (size_t j = 0; j < size; j++)
        {
          const double dn = dnp[j];
          if (dn < res)
            res = dn;
        }
      }
      return res;
    }
  } // namespace itools
} // namespace exaDEM
