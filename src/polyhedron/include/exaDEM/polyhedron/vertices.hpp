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

namespace exaDEM {
template <typename T>
using vector_t = onika::memory::CudaMMVector<T>;

/**
 * @brief Represents a GPU-resident vector field attached to vertices of particles.
 *
 * The field stores a 3D vector (Vec3d) for each vertex of each particle,
 * using a Structure of Arrays (SoA) layout for performance on GPU.
 */
struct VertexField {
  int n_particles_ = 0;       /**< Number of particles */
  int n_vertices_ = 0;        /**< Number of vertices per particle */
  vector_t<double> vertices_; /**< Flattened storage of 3D vertex data in SoA layout (x, y, z) */

  void resize(int np /* number of particles */, int nv /* number of vertices */) {
    n_particles_ = np;
    n_vertices_ = nv;
    vertices_.resize(3 * n_particles_ * n_vertices_);
  }

  /**
   * @brief Accesses a vertex value (non-const version).
   * @param pid Particle index
   * @param vid Vertex index (within particle)
   */
  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator()(int pid, int vid)
#if defined(__GNUC__)
      __attribute__((always_inline))
#endif
  {
    assert(vid < n_vertices_);
    int i = pid + 3 * n_particles_ * vid;
    double* const __restrict__ data = onika::cuda::vector_data(vertices_);
    return Vec3d{data[i], data[i + n_particles_], data[i + 2 * n_particles_]};
  }

  /**
   * @brief Accesses a vertex value (const version).
   * @param pid Particle index
   * @param vid Vertex index (within particle)
   */
  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator()(int pid, int vid) const
#if defined(__GNUC__)
      __attribute__((always_inline))
#endif
  {
    assert(vid < n_vertices_);
    assert(pid < n_particles_);
    int i = pid + 3 * n_particles_ * vid;
    const double* __restrict__ data = onika::cuda::vector_data(vertices_);
    return Vec3d{data[i], data[i + n_particles_], data[i + 2 * n_particles_]};
  }

  /**
   * @brief Sets the value of a vertex.
   * @param value The 3D vector to assign
   * @param pid Particle index
   * @param vid Vertex index (within particle)
   */
  ONIKA_HOST_DEVICE_FUNC inline void set(const Vec3d& value, int pid, int vid) {
    assert(vid < n_vertices_);
    assert(pid < n_particles_);
    int i = pid + 3 * n_particles_ * vid;
    double* const __restrict__ data = onika::cuda::vector_data(vertices_);
    data[i] = value.x;
    data[i + n_particles_] = value.y;
    data[i + 2 * n_particles_] = value.z;
  }
};

/**
 * @brief Provides a convenient view to access and modify vertex data for a single particle.
 * This is a wrapper around a `VertexField` instance for a fixed particle ID.
 * It allows simplified access to all vertices associated with the given particle.
 */
struct ParticleVertexView {
  size_t pid_;          /**< Particle ID */
  VertexField& buffer_; /**< Reference to the vertex field buffer */

  ONIKA_HOST_DEVICE_FUNC inline Vec3d operator[](int vid) __attribute__((always_inline))
#if defined(__GNUC__)
  __attribute__((always_inline))
#endif
  {
    return buffer_(pid_, vid);
  }

  ONIKA_HOST_DEVICE_FUNC inline const Vec3d operator[](int vid) const __attribute__((always_inline))
#if defined(__GNUC__)
  __attribute__((always_inline))
#endif
  {
    return buffer_(pid_, vid);
  }

  ONIKA_HOST_DEVICE_FUNC inline void set(Vec3d& vertex, int vid) { buffer_.set(vertex, pid_, vid); }
};

/**
 * @brief Container for per-cell vertex data on GPU.
 * Stores a vector of `VertexField` instances, one per cell.
 * Each cell holds vertex fields for multiple particles.
 */
struct CellVertexField {
  vector_t<VertexField> buffers_; /**< Array of vertex field buffers (one per cell) */

  VertexField* data() { return buffers_.data(); }

  void resize(int size) { buffers_.resize(size); }

  void resize(int cell, int np, int nv) { buffers_[cell].resize(np, nv); }

  ONIKA_HOST_DEVICE_FUNC inline VertexField& operator[](int i) { return buffers_[i]; }
};
}  // namespace exaDEM
