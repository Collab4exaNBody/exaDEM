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

# include <exaDEM/shape.hpp>

namespace exaDEM
{
	/**
	 * @brief Container for geometric shapes used in particle interactions.
	 */
	struct shapes
	{
		template <typename T> using VectorT = onika::memory::CudaMMVector<T>;

    VectorT<shape> m_data;       ///< Shape storage on GPU
    int m_max_nv = 0;            ///< Maximum number of vertices among all shapes
    bool m_use_obb_tree = false; ///< Whether to enable OBB-tree acceleration

    /// @return pointer to device data (const)
    inline const shape *data() const { return onika::cuda::vector_data(m_data); }

    /// @return number of shapes stored
    inline size_t size() { return onika::cuda::vector_size(m_data); }

    /// @overload const version
    inline size_t size() const { return onika::cuda::vector_size(m_data); }

    /// @return maximum number of vertices among all stored shapes
    inline size_t max_number_of_vertices() { return m_max_nv; }

    /// @return true if OBB tree acceleration is enabled
    inline bool use_obb_tree() { return m_use_obb_tree; }

    /// Enable OBB tree acceleration
    void enable_obb_tree() { m_use_obb_tree = true; }

    /**
     * @brief Access shape by index (const)
     * @param idx index of the shape
     * @return pointer to the shape
     */
		ONIKA_HOST_DEVICE_FUNC
			inline const shape *operator[](const uint32_t idx) const
			{
				const shape *data = onika::cuda::vector_data(m_data);
				return data + idx;
			}

    /**
     * @brief Access shape by index (mutable)
     * @param idx index of the shape
     * @return pointer to the shape
     */
		ONIKA_HOST_DEVICE_FUNC
			inline shape *operator[](const uint32_t idx)
			{
				shape * const data = onika::cuda::vector_data(m_data);
				return data + idx;
			}

    /**
     * @brief Access shape by name
     * @param name shape name
     * @return pointer to the shape, or nullptr if not found
     */
		ONIKA_HOST_DEVICE_FUNC
			inline shape *operator[](const std::string name)
			{
				for (auto &shp : this->m_data)
				{
					if (shp.m_name == name)
					{
						return &shp;
					}
				}
				return nullptr;
			}

    /**
     * @brief Add a new shape (copy)
     * @param shp shape to add
     */
		inline void add_shape(shape& shp)
		{
			this->m_data.push_back(shp); // copy
			m_max_nv = std::max(m_max_nv, shp.get_number_of_vertices());
		}

    /// @brief Add a new shape (by pointer)
		inline void add_shape(shape *shp) { add_shape(*shp); }


    /**
     * @brief Check if container already contains a shape with same name
     * @param shp shape to check
     * @return true if found, false otherwise
     */
		inline bool contains(shape& shp)
		{
			for(auto& s : m_data) {
				if(shp.m_name == s.m_name) {
					return true;
				}
			}
			return false;
		}
	};
} // namespace exaDEM
