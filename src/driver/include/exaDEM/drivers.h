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
    /**
     * @brief Alias template for a CUDA memory managed vector.
     * @tparam T The type of elements in the vector.
     */
    template <typename T> using vector_t = onika::memory::CudaMMVector<T>;
    using data_t = std::variant<exaDEM::Cylinder, exaDEM::Surface, exaDEM::Ball, exaDEM::Stl_mesh, exaDEM::UndefinedDriver>;
    using driver_t = DRIVER_TYPE;
    vector_t<driver_t> m_type;  /**< Vector storing the types of drivers. */
    vector_t<data_t> m_data;       /**< Vector storing the data of drivers. */

    /**
     * @brief Get the size of the Drivers collection.
     * @return The size of the Drivers collection.
     */
    inline size_t get_size () { return m_type.size(); } 

    /**
     * @brief Adds a driver to the Drivers collection at the specified index.
     * @tparam T The type of driver to be added.
     * @param idx The index at which to add the driver.
     * @param Driver The driver to be added.
     * @details If the specified index is beyond the current size of the collection, it will resize the collection accordingly.
     *          If a driver already exists at the specified index, it will be replaced.
     *          If the type of driver is undefined, it will throw a static assertion error.
     */
    template <typename T>
      void add_driver(const int idx, T& Driver)
      {
        constexpr DRIVER_TYPE t = get_type<T>();
        static_assert ( t != DRIVER_TYPE::UNDEFINED );
        assert ( m_type.size() == m_data.size() );
        const int size = m_type.size();
        if ( idx < size ) // reallocation
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
        else // allocate
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
    /**
     * @brief Clears the Drivers collection, removing all drivers.
     */
    void clear()
    {
      m_type.clear();
      m_data.clear();
    }

    // Accessors

    /**
     * @brief Returns the type of driver at the specified index.
     * @param idx The index of the driver.
     * @return The type of the driver at the specified index.
     */
    ONIKA_HOST_DEVICE_FUNC 
      inline DRIVER_TYPE type(size_t idx)
      {
        assert( idx < m_type.size());
        return m_type[idx];
      }

    /**
     * @brief Returns a constant reference to the data of the driver at the specified index.
     * @param idx The index of the driver.
     * @return A constant reference to the data of the driver at the specified index.
     */
    ONIKA_HOST_DEVICE_FUNC 
      inline const data_t& data(size_t idx) const
      {
        assert( idx < m_data.size());
        const auto* const ptr = onika::cuda::vector_data(m_data);
        return ptr[idx];
      }

    ONIKA_HOST_DEVICE_FUNC 
      inline data_t& data(size_t idx)
      {
        //assert( idx < m_data.size());
        assert( idx < onika::cuda::vector_size(m_data));
        auto* const ptr = onika::cuda::vector_data(m_data);
        return ptr[idx];
      }

		template<typename Driver>
    ONIKA_HOST_DEVICE_FUNC 
      inline Driver* ptr()
      {
        auto* const ptr = onika::cuda::vector_data(m_data);
        return (Driver*)ptr;
      }

    /**
     * @brief Checks if all drivers in the collection are well-defined.
     * @return True if all drivers are well-defined, false otherwise.
     */
    bool well_defined()
    {
      for( auto& it : m_type )
      {
        if ( it == DRIVER_TYPE::UNDEFINED ) return false; 
      }
    }

    /**
     * @brief Prints statistics about the drivers in the collection.
     * @details This function prints the total number of drivers and the count of each driver type.
     */
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
