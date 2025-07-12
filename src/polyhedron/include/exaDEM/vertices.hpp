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

#include <onika/math/basic_types.h>

namespace exaDEM
{
  using namespace onika::math;
	template <typename T> using vector_t = onika::memory::CudaMMVector<T>;
	struct VertexGPUAccessor
	{
		int m_n_particles = 0;
		int m_n_vertices = 0;
		vector_t<double> m_vertices;

    void resize(int np /* number of particels */, int nv /* number of vertices */ ) 
    { 
      m_n_particles = np; 
      m_n_vertices = nv; 
      m_vertices.resize(3 * m_n_particles * m_n_vertices); 
    }

		ONIKA_HOST_DEVICE_FUNC inline Vec3d operator() (int pid, int vid) // particle id, vertex id
		{
      assert(vid < m_n_vertices);
			Vec3d res;
			int i = pid + 3 * m_n_particles * vid;
      double* data = onika::cuda::vector_data(m_vertices);
			res.x = data[i];
			res.y = data[i+m_n_particles];
			res.z = data[i+2*m_n_particles];
      return res;
		}

		ONIKA_HOST_DEVICE_FUNC inline Vec3d operator() (int pid, int vid) const // particle id, vertex id
		{
      assert(vid < m_n_vertices);
      assert(pid < m_n_particles );
			Vec3d res;
			int i = pid + 3 * m_n_particles * vid;
      const double* data = onika::cuda::vector_data(m_vertices);
			res.x = data[i];
			res.y = data[i+m_n_particles];
			res.z = data[i+2*m_n_particles];
      return res;
		}

		ONIKA_HOST_DEVICE_FUNC inline void set (Vec3d& value, int pid, int vid) // particle id, vertex id
		{
      assert(vid < m_n_vertices);
      assert(pid < m_n_particles );
			int i = pid + 3 * m_n_particles * vid;
      double* data = onika::cuda::vector_data(m_vertices);
			data[i]                = value.x;
			data[i+  m_n_particles] = value.y;
			data[i+2*m_n_particles] = value.z;
		}
	};

	struct WrapperVertexGPUAccessor
  {
    size_t pid;
    VertexGPUAccessor& accessor;
    ONIKA_HOST_DEVICE_FUNC inline Vec3d operator[] (int vid) { return accessor(pid, vid); }
    ONIKA_HOST_DEVICE_FUNC inline const Vec3d operator[] (int vid) const { return accessor(pid, vid); }
    ONIKA_HOST_DEVICE_FUNC inline void set (Vec3d& vertex, int vid) { accessor.set(vertex, pid, vid); }
  };
 

  struct GridVertex
  {
    vector_t<VertexGPUAccessor> gv;
    VertexGPUAccessor* data() { return gv.data(); }
    void resize(int size) { gv.resize(size); }
    void resize(int cell, int np, int nv) { gv[cell].resize(np, nv); }
    ONIKA_HOST_DEVICE_FUNC inline VertexGPUAccessor& operator[](int i) { return gv[i]; }
  };
}
