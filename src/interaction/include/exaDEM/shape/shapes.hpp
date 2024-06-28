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

#include<exaDEM/shape/shape.hpp>

namespace exaDEM
{
	// data collection of shapes
	struct shapes
	{
		template <typename T> using VectorT =  onika::memory::CudaMMVector<T>; 
		VectorT<shape> m_data;

		inline const shape* data() const
    {
			return onika::cuda::vector_data( m_data );
    }

		inline size_t get_size()
		{
			return onika::cuda::vector_size( m_data );
		}

		inline size_t get_size() const
		{
			return onika::cuda::vector_size( m_data );
		}

		ONIKA_HOST_DEVICE_FUNC
			inline const shape* operator[] (const uint32_t idx) const
			{
				const shape * data = onika::cuda::vector_data( m_data );
				return data + idx;
			}

		ONIKA_HOST_DEVICE_FUNC
			inline shape* operator[] (const std::string name)
			{
				for (auto& shp : this->m_data)
				{
					if(shp.m_name == name)
					{
						return &shp;
					}
				}
				//std::cout << "Warning, the shape: " << name << " is not included in this collection of shapes. We return a nullptr." << std::endl;
				return nullptr;
			}

		inline void add_shape(shape* shp)
		{
			this->m_data.push_back(*shp); // copy
		}
	};
}
