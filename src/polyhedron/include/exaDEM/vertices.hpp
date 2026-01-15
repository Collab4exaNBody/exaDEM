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

  /**
   * @brief Represents a GPU-resident vector field attached to vertices of particles.
   * 
   * The field stores a 3D vector (Vec3d) for each vertex of each particle,
   * using a Structure of Arrays (SoA) layout for performance on GPU.
   */
  struct VertexField
  {
    int m_n_particles = 0;  /**< Number of particles */
    int m_n_vertices  = 0;  /**< Number of vertices per particle */
    vector_t<double> m_vertices; /**< Flattened storage of 3D vertex data in SoA layout (x, y, z) */

    void resize(int np /* number of particles */, int nv /* number of vertices */ ) 
    { 
      m_n_particles = np; 
      m_n_vertices = nv; 
      m_vertices.resize(3 * m_n_particles * m_n_vertices); 
    }


    /**
     * @brief Accesses a vertex value (non-const version).
     * @param pid Particle index
     * @param vid Vertex index (within particle)
     */
    ONIKA_HOST_DEVICE_FUNC inline Vec3d operator() (int pid, int vid) // particle id, vertex id
    {
      assert(vid < m_n_vertices);
      Vec3d res;
      int i = pid + 3 * m_n_particles * vid;
      double* const __restrict__ data = onika::cuda::vector_data(m_vertices);
      res.x = data[i];
      res.y = data[i+m_n_particles];
      res.z = data[i+2*m_n_particles];
      return res;
    }


    /**
     * @brief Accesses a vertex value (const version).
     * @param pid Particle index
     * @param vid Vertex index (within particle)
     */
    ONIKA_HOST_DEVICE_FUNC inline Vec3d operator() (int pid, int vid) const // particle id, vertex id
    {
      assert(vid < m_n_vertices);
      assert(pid < m_n_particles );
      Vec3d res;
      int i = pid + 3 * m_n_particles * vid;
      const double* __restrict__ data = onika::cuda::vector_data(m_vertices);
      res.x = data[i];
      res.y = data[i+m_n_particles];
      res.z = data[i+2*m_n_particles];
      return res;
    }

    /**
     * @brief Sets the value of a vertex.
     * @param value The 3D vector to assign
     * @param pid Particle index
     * @param vid Vertex index (within particle)
     */
    ONIKA_HOST_DEVICE_FUNC inline void set (const Vec3d& value, int pid, int vid) // particle id, vertex id
    {
      assert(vid < m_n_vertices);
      assert(pid < m_n_particles );
      int i = pid + 3 * m_n_particles * vid;
      double* const __restrict__ data = onika::cuda::vector_data(m_vertices);
      data[i]                 = value.x;
      data[i+  m_n_particles] = value.y;
      data[i+2*m_n_particles] = value.z;
    }
  };

  /**
   * @brief Provides a convenient view to access and modify vertex data for a single particle.
   * This is a wrapper around a `VertexField` instance for a fixed particle ID.
   * It allows simplified access to all vertices associated with the given particle.
   */
  struct ParticleVertexView
  {
    size_t pid;              /**< Particle ID */
    VertexField& buffer;     /**< Reference to the vertex field buffer */
    ONIKA_HOST_DEVICE_FUNC inline Vec3d operator[] (int vid) { return buffer(pid, vid); }
    ONIKA_HOST_DEVICE_FUNC inline const Vec3d operator[] (int vid) const { return buffer(pid, vid); }
    ONIKA_HOST_DEVICE_FUNC inline void set (Vec3d& vertex, int vid) { buffer.set(vertex, pid, vid); }
  };

  /**
   * @brief Container for per-cell vertex data on GPU.
   * Stores a vector of `VertexField` instances, one per cell.
   * Each cell holds vertex fields for multiple particles.
   */
  struct CellVertexField
  {
    vector_t<VertexField> buffers;  /**< Array of vertex field buffers (one per cell) */
    VertexField* data() { return buffers.data(); }
    void resize(int size) { buffers.resize(size); }
    void resize(int cell, int np, int nv) { buffers[cell].resize(np, nv); }
    ONIKA_HOST_DEVICE_FUNC inline VertexField& operator[](int i) { return buffers[i]; }
  };
}
