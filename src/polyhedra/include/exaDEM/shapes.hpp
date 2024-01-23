#pragma once

#include<exaDEM/shape.hpp>

namespace exaDEM
{
	// data collection of shapes
	struct shapes
	{
		std::vector<shape> m_data;

		const size_t get_size()
		{
			return m_data.size();
		}

		inline const size_t get_size() const
		{
			return m_data.size();
		}
/*
		shape* operator[] (const size_t idx)
		{
			return &m_data[idx];
		}
*/

		const shape* operator[] (const uint8_t idx) const
		{
			return &m_data[idx];
		}

		shape* operator[] (const std::string name)
		{
			for (auto& shp : this->m_data)
			{
				if(shp.m_name == name)
				{
					return &shp;
				}
			}
			std::cout << "Warning, the shape: " << name << " is not included in this collection of shapes. We return a nullptr." << std::endl;
			return nullptr;
		}

		void add_shape(shape* shp)
		{
			this->m_data.push_back(*shp); // copy
		}
	};
}
