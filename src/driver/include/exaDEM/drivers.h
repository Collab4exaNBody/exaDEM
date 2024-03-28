

#pragma once

#include <exaDEM/driver_base.h>
#include <exaDEM/cylinder.h>
#include <exaDEM/surface.h>
#include <exaDEM/ball.h>
#include <exaDEM/driver_stl_mesh.h>
#include <exaDEM/undefined_driver.h>
#include <variant>

namespace exaDEM
{
	using namespace exanb;

	struct Drivers
	{
		template <typename T> using vector_t = onika::memory::CudaMMVector<T>;
		using data_t = std::variant<exaDEM::Cylinder, exaDEM::Surface, exaDEM::Ball, exaDEM::Stl_mesh, exaDEM::UndefinedDriver>;
		using driver_t = DRIVER_TYPE;
		vector_t<DRIVER_TYPE> m_type;
		vector_t<data_t> m_data;

    inline size_t get_size () { return m_type.size(); } 

		template <typename T>
		void add_driver(const int idx, T& Driver)
		{
			constexpr DRIVER_TYPE t = get_type<T>();
			static_assert ( t != DRIVER_TYPE::UNDEFINED );
			assert ( m_type.size() == m_data.size() );
			const int size = m_type.size();
			if ( idx < size )
			{
				DRIVER_TYPE current_type = type(idx);
				if ( current_type != DRIVER_TYPE::UNDEFINED )
				{
					std::cout << "You are currently removing a driver at index " << idx << std::endl ;
					Driver.print();
				}
				m_type[idx] = t;
				m_data[idx] = Driver;
			}
			else
			{
				int new_size = idx + 1;
				m_type.resize( new_size );
				m_data.resize( new_size );
				for ( int i = size ;  i < idx ; i++ )
				{
					m_type[i] = DRIVER_TYPE::UNDEFINED;
					m_data[i] = exaDEM::UndefinedDriver();
				}
				m_type[idx] = t;
				m_data[idx] = Driver;
			}
		}

		void clear()
		{
			m_type.clear();
			m_data.clear();
		}

		// Accessors
		ONIKA_HOST_DEVICE_FUNC 
		inline DRIVER_TYPE type(size_t idx)
		{
			assert( idx < m_type.size());
			return m_type[idx];
		}

		ONIKA_HOST_DEVICE_FUNC 
		inline const data_t& data(size_t idx) const
		{
			assert( idx < m_data.size());
			return m_data[idx];
		}

		ONIKA_HOST_DEVICE_FUNC 
		inline data_t& data(size_t idx)
		{
			assert( idx < m_data.size());
			return m_data[idx];
		}

		bool well_defined()
		{
			for( auto& it : m_type )
			{
				if ( it == DRIVER_TYPE::UNDEFINED ) return false; 
			}
		}

		void stats_drivers()
		{
			int Count [DRIVER_TYPE_SIZE]; // defined in driver_base.h
			for( auto& it : m_type ) Count[it]++;
			std::cout << "Drivers Stats" << std::endl;
			std::cout << "Number of drivers: " << m_type.size() << std::endl;
			for( int t = 0 ; t < DRIVER_TYPE_SIZE ; t++)
			{
				DRIVER_TYPE tmp = DRIVER_TYPE(t);
				std::cout << "Number of " << print(tmp) << "s: " << Count[t] << std::endl;
			}
		}
	};
}
