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

#include <exaDEM/contact_force_parameters.h>
#include <exaDEM/contact_force_accessor.h>
#include <exanb/core/particle_type_id.h>

namespace exaDEM
{
  template <typename T> using vector_t = onika::memory::CudaMMVector<T>;
  using ReverseParticleTypeMap = std::map< uint64_t, std::string >;

  /**
   * @brief Stores contact parameters for multiple material pairs and driver types.
   *
   * @tparam ContactParamsT The type used to represent contact parameters (e.g., ContactParams).
   */
  template<typename ContactParamsT>
    struct ContactParamsMultiMat
    {
      vector_t<ContactParamsT> multimat_cp;         //< Contact parameters between material pairs
      vector_t<ContactParamsT> drivers_cp;          //< Contact parameters between drivers and materials
      ParticleTypeMap type_map;                     //< Maps particle types to material indices
      ParticleTypeMap driver_map;                   //< Maps driver types to driver indices
      ReverseParticleTypeMap reverse_type_map;      //< Maps material indices back to particle type names
      ReverseParticleTypeMap reverse_driver_map;    //< Maps driver indices back to driver type names
      int number_of_materials;                      //< Number of unique material types in the simulation
      int number_of_drivers;                        //< Number of unique driver types in the simulation

      /**
       * @brief Returns the number of material types.
       */
      int size_types()
      {
        assert(type_map.size() == reverse_type_map.size());
        assert(type_map.size() == multimat_cp.size());
        assert(type_map.size() == number_of_materials);
        return number_of_materials;
      }

      /**
       * @brief Returns the number of driver types.
       */

      int size_drivers()
      {
        assert(driver_map.size() == reverse_driver_map.size());
        assert(driver_map.size() == drivers_cp.size());
        assert(driver_map.size() == number_of_drivers);
        return number_of_drivers;
      }

      /**
       * @brief Returns an accessor for material-to-material contact parameters.
       * @return Accessor to the material-to-material contact parameters.
       */
      MultiMatContactParamsTAccessor<ContactParamsT> get_multimat_accessor() const
      {
        MultiMatContactParamsTAccessor<ContactParamsT> res = { multimat_cp.data() , number_of_materials };
        return res;
      }

      /**
       * @brief Returns an accessor for driver-to-material contact parameters.
       * @return Accessor to the driver-to-material contact parameters.
       */
      MultiMatContactParamsTAccessor<ContactParamsT> get_drivers_accessor() const
      {
        MultiMatContactParamsTAccessor<ContactParamsT> res = { drivers_cp.data() , number_of_drivers };
        return res;
      }

      /**
       * @brief Applies a function to each element of `multimat_cp` and `drivers_cp`.
       *
       * This generic method accepts a functor (or lambda) and applies it
       * to all elements contained in `multimat_cp` followed by those in `drivers_cp`.
       *
       * @tparam Func The type of the functor or lambda to be applied.
       * @param func A reference to the functor or lambda to apply to each element.
       */
      template<typename Func>
        void apply(Func& func)
        {
          for(size_t i = 0 ; i < multimat_cp.size() ; i++) func(multimat_cp[i]); 
          for(size_t i = 0 ; i < drivers_cp.size() ; i++) func(drivers_cp[i]); 
        }

      /**
       * @brief Initializes the material-to-material contact parameters.
       * 
       * Sets up the type maps (`type_map` and `reverse_type_map`) and allocates the
       * array of contact parameters (`multimat_cp`) based on the provided input map.
       * Initializes the contact parameters for each material pair.
       * 
       * @param input_map A map of particle types to material indices.
       * @param cp Initial contact parameters for each material pair (default is a default-constructed object).
       */
      void setup_multimat(const ParticleTypeMap& input_map, ContactParamsT cp = {}) 
      {
        /** Define type map and reverse type map */
        type_map = input_map;
        for(auto it : type_map)
        {
          reverse_type_map[it.second] = it.first;
        }
        // Allocate array of Contact Force Parameters
        number_of_materials = input_map.size();
        multimat_cp.resize(number_of_materials * number_of_materials, cp);
      }

      /**
       * @brief Initializes the driver-to-material contact parameters.
       * 
       * Sets up the type maps (`driver_map` and `reverse_driver_map`) and allocates the
       * array of contact parameters (`drivers_cp`) based on the provided input map.
       * Initializes the contact parameters for interactions between drivers and materials.
       * 
       * @param input_map A map of particle types to driver indices.
       * @param cp Initial contact parameters for each driver-material interaction (default is a default-constructed object).
       */
      void setup_drivers(const ParticleTypeMap& input_map, ContactParamsT cp = {}) 
      {
        driver_map = input_map;
        for(auto it : driver_map)
        {
          reverse_driver_map[it.second] = it.first;
        }
        number_of_drivers = input_map.size();
        assert(number_of_materials > 0);
        drivers_cp.resize(number_of_drivers * number_of_materials, cp);
      }

      /**
       * @brief Returns the index for material-to-material contact parameters.
       * 
       * @param mat1 The index of the first material.
       * @param mat2 The index of the second material.
       * @return The calculated index for the material-to-material contact parameter.
       */
      ONIKA_HOST_DEVICE_FUNC inline int get_idx_multimat(int mat1, int mat2) 
      { 
        assert(mat1 < size_types());
        assert(mat2 < size_types());
        return mat1*number_of_materials + mat2; 
      }

      /**
       * @brief Returns the index for material-to-driver contact parameters.
       * 
       * Computes the linear index in the contact parameter array for the interaction between a material and a driver.
       * Ensures the material and driver indices are valid before performing the calculation.
       * 
       * @param mat The index of the material.
       * @param driver The index of the driver.
       * @return The calculated index for the driver-to-material contact parameter.
       */
      ONIKA_HOST_DEVICE_FUNC inline int get_idx_drivers(int mat, int driver)  
      { 
        assert(mat < number_of_materials);
        assert(driver < number_of_drivers);
        return mat*number_of_drivers + driver; 
      }

      /**
       * @brief Registers the contact parameters for a material-to-material interaction.
       * 
       * Sets the contact parameters for both directions of the material-to-material interaction
       * (mat1, mat2) and (mat2, mat1), ensuring the contact parameters are symmetric.
       * 
       * @param mat1 The index of the first material.
       * @param mat2 The index of the second material.
       * @param cp The contact parameters to be registered for the interaction.
       */
      void register_multimat(int mat1, int mat2, ContactParamsT& cp)
      {
        int id = get_idx_multimat(mat1, mat2);
        multimat_cp[id] = cp;
        id = get_idx_multimat(mat2, mat1);
        multimat_cp[id] = cp;
      }

      /**
       * @brief Registers the contact parameters for a driver-to-material interaction.
       * 
       * Sets the contact parameters for the interaction between a material and a driver.
       * 
       * @param mat The index of the material.
       * @param driver The index of the driver.
       * @param cp The contact parameters to be registered for the interaction.
       */
      void register_driver(int mat, int driver, ContactParamsT& cp)
      {
        int id = get_idx_drivers(mat, driver);
        drivers_cp[id] = cp;
      }

      /**
       * @brief Checks the completeness of the contact parameters for materials and drivers.
       * 
       * This function verifies that the contact parameters for all material-to-material and
       * driver-to-material interactions are defined (i.e., not equal to the default contact parameters).
       * If any contact parameters are missing, a warning is printed to the output.
       * 
       * @param check_multimat If true, checks the completeness of material-to-material contact parameters.
       * @param check_driver If true, checks the completeness of driver-to-material contact parameters.
       * @return `true` if all contact parameters are defined, `false` otherwise.
       */
      bool check_completeness(bool check_multimat = true, bool check_driver = true)
      {
        bool check = true;
        ContactParamsT default_cp = {};
        for(int m1 = 0 ; m1 < number_of_materials ; m1++)
        {
          if(check_multimat)
          {
            // Check material-to-material parameters
            for(int m2 = 0 ; m2 <= m1 ; m2++) /** symtric definition */
            {
              if(multimat_cp[this->get_idx_multimat(m1, m2)] == default_cp)
              {
                lout << "\033[1;31m[WARNING] Contact force parameters for the pair (mat: " 
                  << m1 << ", mat: " << m2 << ") are not defined.\033[0m" << std::endl;
                check = false;
              }
            }
          }
          // Check driver-to-material parameters
          if(check_driver)
          {
            for(int drv = 0 ; drv < number_of_drivers ; drv++)
            {
              if(drivers_cp[this->get_idx_drivers(m1, drv)] == default_cp)
              {
                lout << "\033[1;31m[WARNING] Contact force parameters for the pair (mat: " 
                  << m1 << ", driver: " << drv << ") are not defined.\033[0m" << std::endl;
                check = false;
              }
            }
          }
        }
        return check;
      }

      /**
       * @brief Displays the contact parameters for materials and drivers.
       * 
       * Depending on the flags `multimat_mode` and `drivers_mode`, this function displays
       * the contact parameters for material-to-material and/or driver-to-material interactions.
       * It prints a formatted table with the contact parameters for the specified modes.
       * 
       * @param multimat_mode If true, displays material-to-material contact parameters.
       * @param drivers_mode If true, displays driver-to-material contact parameters.
       */
      void display(bool multimat_mode = true, bool drivers_mode = true)
      {
        display_header<ContactParamsT>();
        // Loop through all materials
        for(int m1 = 0 ; m1 < number_of_materials ; m1++)
        {
          // Display material-to-material parameters
          if(multimat_mode)
          {
            for(int m2 = 0 ; m2 <= m1 ; m2++) /** symtric definition */
            {
              std::string type1 = reverse_type_map[m1];
              std::string type2 = reverse_type_map[m2];
              ContactParamsT& cp = multimat_cp[this->get_idx_multimat(m1, m2)];
              display_multimat(type1, type2, cp);
            }
          }
          // Display driver-to-material parameters
          if(drivers_mode)
          {
            for(int drv = 0 ; drv < number_of_drivers ; drv++)
            {
              display_multimat(reverse_type_map[m1], reverse_driver_map[drv], drivers_cp[this->get_idx_drivers(m1, drv)]);
            }
          }
        }
        display_end_table<ContactParamsT>();
      }

      /**
       * @brief Retrieves the contact parameters for a material-to-material interaction.
       * 
       * @param mat1 The index of the first material.
       * @param mat2 The index of the second material.
       * @return A reference to the contact parameters for the material-to-material interaction.
       */
      ONIKA_HOST_DEVICE_FUNC inline ContactParamsT& get_multimat_cp(int mat1, int mat2) const
      {
        return multimat_cp[this->get_idx_multimat(mat1, mat2)];
      }

      /**
       * @brief Retrieves the contact parameters for a driver-to-material interaction.
       * 
       * @param mat The index of the material.
       * @param driver The index of the driver.
       * @return A reference to the contact parameters for the driver-to-material interaction.
       */
      ONIKA_HOST_DEVICE_FUNC inline ContactParamsT& get_drivers_cp(int mat, int driver) const
      {
        return drivers_cp[this->get_idx_driver(mat, driver)];
      }
    };
} // namespace exaDEM
