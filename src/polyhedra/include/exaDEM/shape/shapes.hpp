#pragma once

#include<exaDEM/shape/shape.hpp>

namespace exaDEM
{
	// data collection of shapes
	struct shapes
	{
		template <typename T> using VectorT =  onika::memory::CudaMMVector<T>; 
		VectorT<shape> m_data;

		inline size_t get_size()
		{
			return onika::cuda::vector_size( m_data );
		}

		inline size_t get_size() const
		{
			return onika::cuda::vector_size( m_data );
		}

		ONIKA_HOST_DEVICE_FUNC
			inline const shape* operator[] (const uint8_t idx) const
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
