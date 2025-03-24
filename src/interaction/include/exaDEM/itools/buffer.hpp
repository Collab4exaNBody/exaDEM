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

namespace exaDEM
{
  namespace itools
  {
    struct interaction_buffers
    {
      template <typename T> using VectorT = onika::memory::CudaMMVector<T>;
      VectorT<double> dn;
      VectorT<Vec3d> cp;
      VectorT<Vec3d> fn;
      VectorT<Vec3d> ft;

      void resize(const size_t size)
      {
        assert(size < 1e9);
        if( size == dn.size() ) return;

        if (size != 0)
        {
          dn.resize(size);
          cp.resize(size);
          fn.resize(size);
          ft.resize(size);
        }
        else
        {
          dn.clear();
          cp.clear();
          fn.clear();
          ft.clear();
        }
      }
    };
  } // namespace itools
} // namespace exaDEM
