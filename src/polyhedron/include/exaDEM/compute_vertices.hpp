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
#include <cassert>
#include <onika/math/basic_types.h>
#include <exanb/compute/compute_cell_particles.h>

namespace exaDEM
{
  template<bool def_box>
  struct PolyhedraComputeVerticesFunctor
  {
    const shape *shps;
    VertexGPUAccessor * gv;
    const Mat3d xform;
    ONIKA_HOST_DEVICE_FUNC inline void operator()(const size_t cell_idx, const size_t p, const uint32_t type, const double rx, const double ry, const double rz, const double h, const exanb::Quaternion &orient) const
    {
      // get vertices
      WrapperVertexGPUAccessor vertices = { p , gv[cell_idx] };

      // h will be used in a next development
      const auto &shp = shps[type];
      const unsigned int nv = shp.get_number_of_vertices();
      exanb::Vec3d position = {rx, ry, rz};
      if constexpr(def_box) position = xform * position;
      for (size_t i = 0; i < nv; i++)
      {
        Vec3d vertex = shp.get_vertex(i, position, orient);
        vertices.set(vertex, i);
      }
    }
  };
} // namespace exaDEM

namespace exanb
{
  template <bool def_box> struct ComputeCellParticlesTraits<exaDEM::PolyhedraComputeVerticesFunctor<def_box>>
  {
    static inline constexpr bool RequiresBlockSynchronousCall = false;
    static inline constexpr bool CudaCompatible = true;
    static inline constexpr bool ComputeCellParticlesTraitsUseCellIdx = true;
  };
} // namespace exanb
