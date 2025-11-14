/*
   Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.
*/

#pragma once

#include <exaDEM/classifier/classifier.hpp>

namespace exaDEM
{
  namespace itools
  {
    inline double get_min_dn(const Classifier &classifier)
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
