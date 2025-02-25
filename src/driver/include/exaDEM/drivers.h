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
#include <exaDEM/stl_mesh.h>
#include <exaDEM/undefined_driver.h>
#include <onika/flat_tuple.h>

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

    struct DriverTypeAndIndex
    {
      DRIVER_TYPE m_type = DRIVER_TYPE::UNDEFINED;
      int m_index = -1;
    };
    
    vector_t<DriverTypeAndIndex> m_type_index; /**< Vector storing the types of drivers. */
    
    //vector_t<data_t> m_data;   /**< Vector storing the data of drivers. */
    onika::FlatTuple< vector_t<Cylinder> , vector_t<Surface> , vector_t<Ball> , vector_t<Stl_mesh> > m_data;

    /**
     * @brief Get the size of the Drivers collection.
     * @return The size of the Drivers collection.
     */
    inline size_t get_size() const { return m_type_index.size(); }

    template<size_t driver_type>
    inline const auto & get_driver_vec() const
    {
      static_assert( driver_type != DRIVER_TYPE::UNDEFINED );
      return m_data.get_nth_const< driver_type >();
    }

    template<size_t driver_type>
    inline auto & get_driver_vec()
    {
      static_assert( driver_type != DRIVER_TYPE::UNDEFINED );
      return m_data.get_nth< driver_type >();
    }

    template<class T>
    inline const T& get_typed_driver(const int idx) const
    {
      constexpr DRIVER_TYPE t = get_type<T>();
      static_assert( t != DRIVER_TYPE::UNDEFINED );
      const auto & driver_vec = m_data.get_nth_const<t>();
      assert( idx>=0 && idx<m_type_index.size() );
      assert( m_type_index[idx].m_type == t );
      assert( m_type_index[idx].m_index >= 0 && m_type_index[idx].m_index < driver_vec.size() );
      return driver_vec[m_type_index[idx].m_index];
    }

    template<class T>
    inline T& get_typed_driver(const int idx)
    {
      constexpr DRIVER_TYPE t = get_type<T>();
      static_assert( t != DRIVER_TYPE::UNDEFINED );
      auto & driver_vec = m_data.get_nth<t>();
      assert( idx>=0 && idx<m_type_index.size() );
      assert( m_type_index[idx].m_type == t );
      assert( m_type_index[idx].m_index >= 0 && m_type_index[idx].m_index < driver_vec.size() );
      return driver_vec[m_type_index[idx].m_index];
    }

    template<class FuncT>
    inline auto apply(const int idx , const FuncT& func)
    {
      assert( idx>=0 && idx<m_type_index.size() );
      DRIVER_TYPE t = m_type_index[idx].m_type;
      assert( t != DRIVER_TYPE::UNDEFINED );
           if (t == DRIVER_TYPE::CYLINDER) return func( m_data.get_nth<DRIVER_TYPE::CYLINDER>()[ m_type_index[idx].m_index ] );
      else if (t == DRIVER_TYPE::SURFACE)  return func( m_data.get_nth<DRIVER_TYPE::SURFACE >()[ m_type_index[idx].m_index ] );
      else if (t == DRIVER_TYPE::BALL)     return func( m_data.get_nth<DRIVER_TYPE::BALL    >()[ m_type_index[idx].m_index ] );
      else if (t == DRIVER_TYPE::STL_MESH) return func( m_data.get_nth<DRIVER_TYPE::STL_MESH>()[ m_type_index[idx].m_index ] );
      fatal_error() << "Internal error: unsupported driver type encountered"<<std::endl;
      static Cylinder tmp;
      return func( tmp );
    }

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
    inline void add_driver(const int idx, T &Driver)
    {
      constexpr DRIVER_TYPE t = get_type<T>();
      static_assert(t != DRIVER_TYPE::UNDEFINED);
      //assert(m_type_index.size() == m_data.size());
      const int size = m_type_index.size();
      if (idx < size) // reallocation
      {
        DRIVER_TYPE current_type = type(idx);
        if (current_type != DRIVER_TYPE::UNDEFINED)
        {
          lout << "You are currently removing a driver at index " << idx << std::endl;
          Driver.print();
        }
      }
      else // allocate
      {
        m_type_index.resize(idx+1);
      }
      m_type_index[idx].m_type = t;
      auto & driver_vec = get_driver_vec<t>();
      m_type_index[idx].m_index = driver_vec.size();
      driver_vec.push_back( Driver );
    }
    
    /**
     * @brief Clears the Drivers collection, removing all drivers.
     */
    void clear()
    {
      m_type_index.clear();
      m_data.get_nth<DRIVER_TYPE::CYLINDER>().clear();
      m_data.get_nth<DRIVER_TYPE::SURFACE>().clear();
      m_data.get_nth<DRIVER_TYPE::BALL>().clear();
      m_data.get_nth<DRIVER_TYPE::STL_MESH>().clear();
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
      assert(idx < m_type_index.size());
      return m_type_index[idx].m_type;
    }

    /**
     * @brief Checks if all drivers in the collection are well-defined.
     * @return True if all drivers are well-defined, false otherwise.
     */
    inline bool well_defined() const
    {
      for (const auto &it : m_type_index)
      {
        if (it.m_type == DRIVER_TYPE::UNDEFINED)
          return false;
      }
      return true;
    }

    /**
     * @brief Prints Drivers informations.
     */
    inline void print_drivers() const
    {
      for (size_t i = 0; i < this->get_size(); i++)
      {
        auto t = m_type_index[i].m_type;
        if (t != DRIVER_TYPE::UNDEFINED)
        {
          lout << "Driver [" << i << "]:" << std::endl;
               if (t == DRIVER_TYPE::CYLINDER) m_data.get_nth_const<DRIVER_TYPE::CYLINDER>()[ m_type_index[i].m_index ].print();
          else if (t == DRIVER_TYPE::SURFACE)  m_data.get_nth_const<DRIVER_TYPE::SURFACE >()[ m_type_index[i].m_index ].print();
          else if (t == DRIVER_TYPE::BALL)     m_data.get_nth_const<DRIVER_TYPE::BALL    >()[ m_type_index[i].m_index ].print();
          else if (t == DRIVER_TYPE::STL_MESH) m_data.get_nth_const<DRIVER_TYPE::STL_MESH>()[ m_type_index[i].m_index ].print();
        }
      }
    }

    /**
     * @brief Prints statistics about the drivers in the collection.
     * @details This function prints the total number of drivers and the count of each driver type.
     */
    inline void stats_drivers() const
    {
      std::array<int, DRIVER_TYPE_SIZE> Count; // defined in driver_base.h
      for (auto &it : Count)
      {
        it = 0;
      }
      for (const auto &it : m_type_index)
      {
        ++ Count[it.m_type];
      }
      lout << "Drivers Stats" << std::endl;
      lout << "Number of drivers: " << m_type_index.size() << std::endl;
      for (size_t t = 0; t < DRIVER_TYPE_SIZE; t++)
      {
        lout << "Number of " << print(DRIVER_TYPE(t)) << "s: " << Count[t] << std::endl;
      }
    }
  };
  
  // read only proxy for drivers list
  struct DriversGPUAccessor
  { 
    size_t m_nb_drivers = 0;
    Drivers::DriverTypeAndIndex * const __restrict__ m_type_index = nullptr;
    onika::FlatTuple< Cylinder * __restrict__ , Surface * __restrict__ , Ball * __restrict__ , Stl_mesh* __restrict__ > m_data = { nullptr, nullptr, nullptr, nullptr };
    onika::FlatTuple< size_t , size_t , size_t , size_t > m_data_size = { 0, 0, 0, 0 };

    DriversGPUAccessor() = default;
    DriversGPUAccessor(const DriversGPUAccessor &) = default;
    DriversGPUAccessor(DriversGPUAccessor &&) = default;
    inline DriversGPUAccessor(Drivers& drvs)
      : m_nb_drivers( drvs.m_type_index.size() )
      , m_type_index( drvs.m_type_index.data() )
      , m_data( { drvs.m_data.get_nth<0>().data() , drvs.m_data.get_nth<1>().data() , drvs.m_data.get_nth<2>().data() , drvs.m_data.get_nth<3>().data() } )
      , m_data_size( { drvs.m_data.get_nth<0>().size() , drvs.m_data.get_nth<1>().size() , drvs.m_data.get_nth<2>().size() , drvs.m_data.get_nth<3>().size() } )
    {}

    template<class T>
    ONIKA_HOST_DEVICE_FUNC inline T& get_typed_driver(const int idx) const
    {
      constexpr DRIVER_TYPE t = get_type<T>();
      static_assert( t != DRIVER_TYPE::UNDEFINED );
      auto * __restrict__ driver_vec = m_data.get_nth_const<t>();
      const size_t driver_vec_size = m_data_size.get_nth_const<t>();
      assert( idx>=0 && idx<m_nb_drivers );
      assert( m_type_index[idx].m_type == t );
      assert( m_type_index[idx].m_index >= 0 && m_type_index[idx].m_index < driver_vec_size );
      return driver_vec[m_type_index[idx].m_index];
    }
    
  };
  
} // namespace exaDEM
