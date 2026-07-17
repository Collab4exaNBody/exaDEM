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

#include <exanb/core/particle_type_id.h>

#include <exaDEM/shape.hpp>

namespace exaDEM {
/**
 * @brief Container for geometric shapes used in particle interactions.
 */
struct shapes {
  onika::memory::CudaMMVector<shape> data_;  ///< Shape storage on CPU/GPU
  int max_nv_ = 0;                           ///< Maximum number of vertices among all shapes
  bool use_obb_tree_ = false;                ///< Whether to enable OBB-tree acceleration

  /// @return pointer to device data (const)
  inline const shape* data() const { return onika::cuda::vector_data(data_); }

  /// @return number of shapes stored
  inline size_t size() { return onika::cuda::vector_size(data_); }

  /// @overload const version
  inline size_t size() const { return onika::cuda::vector_size(data_); }

  /// @return maximum number of vertices among all stored shapes
  inline size_t max_number_of_vertices() { return max_nv_; }

  /// @return true if OBB tree acceleration is enabled
  inline bool use_obb_tree() { return use_obb_tree_; }

  /// Enable OBB tree acceleration
  void enable_obb_tree() { use_obb_tree_ = true; }

  /**
   * @brief Access shape by index (const)
   * @param idx index of the shape
   * @return pointer to the shape
   */
  ONIKA_HOST_DEVICE_FUNC
  inline const shape* operator[](const uint32_t idx) const {
    const shape* data = onika::cuda::vector_data(data_);
    return data + idx;
  }

  /**
   * @brief Access shape by index (mutable)
   * @param idx index of the shape
   * @return pointer to the shape
   */
  ONIKA_HOST_DEVICE_FUNC
  inline shape* operator[](const uint32_t idx) {
    shape* const data = onika::cuda::vector_data(data_);
    return data + idx;
  }

  /**
   * @brief Access shape by name
   * @param name shape name
   * @return pointer to the shape, or nullptr if not found
   */
  ONIKA_HOST_DEVICE_FUNC
  inline shape* operator[](const std::string name) {
    for (auto& shp : this->data_) {
      if (shp.name_ == name) {
        return &shp;
      }
    }
    return nullptr;
  }

  /**
   * @brief Add a new shape (copy)
   * @param shp shape to add
   */
  inline void add_shape(shape& shp) {
    this->data_.push_back(shp);  // copy
    max_nv_ = std::max(max_nv_, shp.get_number_of_vertices());
  }

  /// @brief Add a new shape (by pointer)
  inline void add_shape(shape* shp) { add_shape(*shp); }

  /**
   * @brief Check if container already contains a shape with same name
   * @param shp shape to check
   * @return true if found, false otherwise
   */
  inline bool contains(shape& shp) {
    for (auto& s : data_) {
      if (shp.name_ == s.name_) {
        return true;
      }
    }
    return false;
  }
};

/**
 * @brief Registers a shape into the particle type map and shape container.
 *
 * @param ptm   Reference to the particle type map.
 * @param shps  Reference to the shape container.
 * @param shp   shape to register.
 */
inline void register_shape(exanb::ParticleTypeMap& ptm, shapes& shps, shape& shp) {
  if (ptm.find(shp.name_) != ptm.end()) {
    shp.name_ = shp.name_ + "X";
    color_log::warning("register_shape",
                       "This polyhedron name is already taken, exaDEM has renamed it to: " + shp.name_);
  }
  ptm[shp.name_] = shps.size();
  shps.add_shape(&shp);
}

/**
 * @brief Registers a collection of shapes into the particle type map and shape container.
 *
 * @param ptm   Reference to the particle type map.
 * @param shps  Reference to the shape container.
 * @param shp   Vector of shapes to register.
 */
inline void register_shapes(exanb::ParticleTypeMap& ptm, shapes& shps, std::vector<shape>& shp) {
  for (auto& s : shp) {
    if (ptm.find(s.name_) != ptm.end()) {
      s.name_ = s.name_ + "X";
      color_log::warning("register_shapes",
                         "This polyhedron name is already taken, exaDEM has renamed it to: " + s.name_);
    }
    ptm[s.name_] = shps.size();
    shps.add_shape(&s);
  }
}
}  // namespace exaDEM
